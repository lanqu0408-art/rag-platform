from utils.wrapper import Wrapper
from services.rag.oss import Oss_Process
from haystack import Document, component, logging
from ..external_api.aizt_api import aizt_chat, aizt_embedding
import os
import traceback
from .lindorm.lindorm import LindormDocumentStore
from .splitter.chinese_document_splitter import chinese_DocumentSpliter
from .splitter.chinese_recrucuit_splitter import chinese_RecrucuitSplitter
# from client.Client import Client
from core.config import config
from haystack.components.writers import DocumentWriter
from .lindorm.knowledge_base_ab import KnowledgeBaseABDocumentStore
import time
import io
import socket, urllib.parse
import requests


class FaqApiSyncUpdate:
    """Faq api sync update Service"""
    def __init__(self):
        env = os.environ['env']
        self.lindorm_url = config.get("lindorm_url")
        self.lindorm_key = config.get("lindorm_key")
        self.lindorm_value = config.get("lindorm_value")
        self.kb_info_client = KnowledgeBaseABDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            index="kbab_" + env + "_new_3"
        )


    def process(self, 
                info_carrier):
            self.kb_id = str(info_carrier['id'])
            cur_env = os.environ.get("env")
            operation_type = info_carrier['operationType'] # 暂时只是create
            kb_used = self.kb_info_client.get(str(self.kb_id)) # 判断库状态  # 这个拿的是可使用的，，要拿到之后存入另一个库
            if kb_used == -1 or kb_used == "b":
                # 证明 这个库是新建的，向a中写
                kb_write = "a"  # 写如a
                kb_used = "b"
                delete_a = False # 整个update结束，删除b
            elif kb_used == "a":
                kb_write = "b" 
                delete_a = True # 整个update 结束 删除 a
            self.delete_a = delete_a
            self.kb_write = kb_write
            self.kb_used = kb_used

            # 这个这一次更新过程不会被改变
            self.document_store_read= LindormDocumentStore(
                hosts=[{"host": self.lindorm_url, "port": 30070}],
                http_auth=(self.lindorm_key, self.lindorm_value),
                index= "kb_" + str(kb_used) + "_" + cur_env,
                embedding_dim=1024,
                ) # 读的库
            self.document_store_write= LindormDocumentStore(
                hosts=[{"host": self.lindorm_url, "port": 30070}],
                http_auth=(self.lindorm_key, self.lindorm_value),
                index="kb_" + kb_write + "_" + cur_env,
                embedding_dim=1024,
                ) # 写的库
            # 上面其实是在更新一些信息，可能独立出去更好一点
            delete_a, doc_write = self.construct_update_pip_faq(self.kb_id)

            query_list = []
            answer_list = []
            doc_num = 0
            try:

                if operation_type == "create":
                    
                    # try:
                        # 获取query_list 和 answer_list

                    faq_info = info_carrier['faq']
                    for i in range(len(faq_info)):
                        query_list.append(faq_info[i]['question'])
                        answer_list.append(faq_info[i]['answer'])
                    # query_list, answer_list, title = self.feishu_processor.get_faq_content(faq_token, sheet_name, question_name, answer_name)


                else:
                    # update  模式
                    # 从相关的知识库中拿到信息，然后写进去
                    # docs = self.document_store_read.filter_documents(
                    #     filters={"kb_id": self.kb_id},
                    #     all=True)   
                    docs = self.document_store_read.filter_documents(filters={"kb_id": str(self.kb_id)})
                    for doc in docs:
                        related_answer = doc.meta.get("related_answer")
                        question = doc.meta.get("question")
                        query_list.append(question)
                        answer_list.append(related_answer)

                keywordNum = info_carrier.get('faqPreprocessConfig', {}).get("keywordCount", 0)
                constructType = info_carrier.get('faqPreprocessConfig', {}).get("constructType", "QUESTION")
                qa_pairs_num = self.faq2kb(query_list, answer_list, delete_a, doc_write, self.kb_id, keywordNum, constructType)
                doc_num += qa_pairs_num
                self.post_process_kb(self.kb_id, delete_a) # 删掉之前备份
                return {"code": 0, "message": "", "data":{"documentCount": doc_num}}

            except Exception as e:
                print(f"处理出错: {e}")
                traceback.print_exc()
                return {"code": 1, "message": "更新出错，联系工作人员", "data":{"documentCount": 0}}


    def post_process_kb(self, kb_name, delete_a, construct_info={}, doc_info={}, update_res={}):
        cur_env = os.environ.get("env")
        # 删除

        print("当前的update_res 是什么")
        print(update_res)
        print(delete_a)
        print(doc_info)
        print(kb_name)
        doc_info_list = [
            {"id": k, **v} for k, v in doc_info.items()
        ]
        # self.kb_info_client.write(str(kb_name), kb_used, construct_info, doc_info_list, update_res)
        if delete_a:  
            
            self.kb_info_client.write(str(kb_name), "b", construct_info, doc_info_list, update_res)
            document_store = LindormDocumentStore(
                hosts=[{"host": self.lindorm_url, "port": 30070}],
                http_auth=(self.lindorm_key, self.lindorm_value),
                index="kb_" + "a" + "_" + cur_env,
                embedding_dim=1024,
            )
        else:
            # self.kb_info_client.write(str(kb_name), "a") #改成a，删除b
            self.kb_info_client.write(str(kb_name), "a", construct_info, doc_info_list, update_res) 
            document_store = LindormDocumentStore(
                hosts=[{"host": self.lindorm_url, "port": 30070}],
                http_auth=(self.lindorm_key, self.lindorm_value),
                index="kb_" + "b" + "_" + cur_env,
                embedding_dim=1024,
            ) 
        filter_doc = document_store.filter_documents(filters={"kb_id": str(kb_name)})
        filter_document_ids = [doc.id for doc in filter_doc]  # 提取所有文档的 ID

        if filter_document_ids:  # 避免空列表导致的无效操作
            document_store.delete_documents(document_ids=filter_document_ids)
            print(f"已删除 {len(filter_document_ids)} 个文档")
        else:
            print("当前文档库中没有文档可删除")



    def faq2kb(self, query_list, answer_list, delete_a, doc_write, kb_id, keywordNum, constructType):
        documents_faq = []
        for i in range(len(query_list)):
            cur_query = str(query_list[i]) if query_list[i] is not None else ""
            cur_answer = str(answer_list[i]) if answer_list[i] is not None else ""
            if not cur_query or not cur_answer:
                continue

            query = cur_query

            if constructType != "QUESTION":
                query = "问题是：" + query + ", 答案是：" + cur_answer + "。"

            if keywordNum != 0:
                # prompt = "根据下面这段文字，提取" + str(keywordNum) + "个关键词，只返回关键词，不用返回其他内容，返回格式为：关键词1/关键词2/关键词3。文字内容为:"
                
                if str(keywordNum) == "1":
                    prompt = f"""请从下面文字中严格提取且仅提取 1 个关键词。

                        硬性要求：
                        1. 只能输出 1 个关键词
                        2. 不允许输出多个关键词
                        3. 不允许使用 / 、， 、, 、； 等任何分隔符
                        4. 不要解释，不要补充，不要重复原文
                        5. 只输出一个最核心的关键词本身


                        文字内容："""
                else:

                    prompt = f"""请根据下面文字提取指定个数的关键词。

                        要求：
                        1. 本次必须提取 {str(keywordNum)} 个关键词
                        2. 返回结果中必须且只能包含 {str(keywordNum)} 个关键词
                        3. 关键词之间用 / 分隔
                        4. 不要返回解释，不要返回多余内容
                        5. 如果内容中可提取的关键词不足，也要尽量概括出 {str(keywordNum)} 个
                        6. 输出格式示例仅指分隔形式，例如：关键词1/关键词2/.../关键词{str(keywordNum)}

                        文字内容："""
                
                prompt += query
                response = aizt_chat(prompt, model_type="qwen3")
                pre_info = "<key_word>" + response + "<key_word_end>"
            else:
                pre_info = "<key_word> "  + "<key_word_end>"
            query = pre_info + query


            cur_doc = Document(content=query,
                               meta={
                                    "constructType": constructType,
                                    "related_answer": cur_answer,
                                    "query_name": "Question",
                                    "answer_name" : "Answer",
                                    "source_kb": str(kb_id),
                                    "question" : cur_query})
            
            cur_doc.kb_id = str(kb_id)
            documents_faq.append(cur_doc)

        for idx in range(len(documents_faq)):
            sub_doc = documents_faq[idx].content
            embedding = aizt_embedding(sub_doc) # 这是在文档入库
            documents_faq[idx].embedding = embedding
        res = doc_write.run(documents=documents_faq)
        print("res")
        print(res)
        # time.sleep(10)
        return len(documents_faq)



    def construct_update_pip_faq(self, kb_id):

        cur_env = os.environ.get("env")
        kb_used = self.kb_info_client.get(str(kb_id)) # 判断库状态  # 这个拿的是可使用的，，要拿到之后存入另一个库
        if kb_used == -1 or kb_used == "b":
            # 证明 这个库是新建的，向a中写
            kb_write = "a"  # 写如a
            delete_a = False # 整个update结束，删除b
        elif kb_used == "a":
            kb_write = "b" 
            delete_a = True # 整个update 结束 删除 a


        # 向kb_write中写入
        document_store= LindormDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            index="kb_"+ kb_write +  "_" +cur_env,  
            embedding_dim=1024,
        )

        filter_doc = document_store.filter_documents(filters={"kb_id": str(kb_id)})
        if len(filter_doc) != 0:
            filter_document_ids = [doc.id for doc in filter_doc]  # 提取所有文档的 ID
            if filter_document_ids:  # 避免空列表导致的无效操作
                document_store.delete_documents(document_ids=filter_document_ids)
        # if document_store.count_documents() != 0:
        #     all_documents = document_store.filter_documents(filters=None)
        #     all_document_ids = [doc.id for doc in all_documents]  # 提取所有文档的 ID
        #     if all_document_ids:  # 避免空列表导致的无效操作
        #         document_store.delete_documents(document_ids=all_document_ids)
        document_writer = DocumentWriter(document_store)
        return delete_a, document_writer



class FaqApiSyncWrapper(Wrapper):
    def __init__(self, config=None):
        self.instance = FaqApiSyncUpdate()