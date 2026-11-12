
import traceback
from typing import Optional, Union, Dict, List
from haystack import Document, component, logging
from utils.wrapper import Wrapper
from services.external_api import ark_chat
import argparse
import lark_oapi as lark
from lark_oapi.api.docx.v1 import *
import json
from .lindorm.lindorm import LindormDocumentStore
from haystack import Document, component, logging
import os
import lark_oapi as lark
import time
from lark_oapi.api.drive.v1 import *
from lark_oapi.api.wiki.v2 import *
from lark_oapi.api.sheets.v3 import *
import copy
import socket, urllib.parse
from collections import deque
from .feishu_process import Feishu_Process
from elasticsearch import Elasticsearch
from getpass import getpass
from pathlib import Path
import re
from haystack import Document, Pipeline
from haystack.dataclasses import ChatMessage
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.writers import DocumentWriter
from haystack.components.builders import ChatPromptBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
# from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.converters import MarkdownToDocument, PyPDFToDocument, TextFileToDocument, XLSXToDocument
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner, RecursiveDocumentSplitter
from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
from haystack.components.converters.docx import DOCXToDocument, DOCXTableFormat
# from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
# from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from .splitter.chinese_document_splitter import chinese_DocumentSpliter
from .splitter.chinese_recrucuit_splitter import chinese_RecrucuitSplitter
from ..external_api.aizt_api import aizt_chat, aizt_embedding, aizt_ocr, check_aizt_chat, check_aizt_embedding, check_aizt_ocr, check_aizt_rerank
import time
import logging
from .lindorm.knowledge_base_ab import KnowledgeBaseABDocumentStore
from .oss import Oss_Process
import base64
from core.config import config
import os, json, tempfile, subprocess
from pathlib import Path
# from client.Client import Client
import threading
import requests
from services.rag.pdf_update import Local_File_Update

logger = logging.getLogger(__name__)
map_dict = {
    1:"page",       # 页面（根） Block
    2:"text",       # 文本 Block
    3:"heading1",   # 一级标题 Block
    4:"heading2",   # 二级标题 Block
    5:"heading3",   # 三级标题 Block
    6:"heading4",   # 四级标题 Block
    7:"heading5",   # 五级标题 Block
    8:"heading6",   # 六级标题 Block
    9:"heading7",   # 七级标题 Block
    10:"heading8",  # 八级标题 Block
    11:"heading9",  # 九级标题 Block
    12:"bullet",    # 无序列表 Block
    13:"ordered",   # 有序列表 Block
}

class Kb_Update:
    """Kb update Service"""
    def __init__(self):
        self.feishu_processor = Feishu_Process()
        self.oss_processor = Oss_Process()
        env = os.environ['env']
        # client = Client()
        self.lindorm_url = config.get("lindorm_url")
        self.lindorm_key = config.get("lindorm_key")
        self.lindorm_value = config.get("lindorm_value")
        self.kb_info_client = KnowledgeBaseABDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            index="kbab_" + env + "_new_3"
        )


    def process(self, 
                info_carrier, 
                model_chat=None, 
                chat_tokenizer=None, 
                chinese_tokenizer_coarse=None,
                chinese_tokenizer_fine = None,
                document_embedder = None,
                text_embedder = None):
        try:
            self.chart_mapping = {}
            self.ocr_mapping = {}
            # self.kb_info_client.get_doc_info(str(999997)).get("cancel", "0") == "1":
            self.kb_info_client.write(str(999997), str(999997), kb_construct_type={'cancel': str(0)})
            self.kb_info_client.write(str(999996), str(999996), kb_construct_type={'finish': str(1)}) # 写在kb_used中
            kb_name = info_carrier['id']
            # 更新逻辑开始时候就需要写入正在更新的知识库
            self.kb_info_client.write(str(999999), str(999999), kb_construct_type={'updating_kb': str(kb_name)}) # 写在kb_used中
            # 然后写入正在更新的知识库
            # 如果点击的过快，不是当前的知识库，显示暂停失败，等下再暂停
            all_fail_doc = []
            cur_env = os.environ.get("env")
            image_fail_all_doc = []
            
            self.kb_name = kb_name
            kb_type = info_carrier['type'] # 知识类型,用于判断是文档还是faq类型
            kb_used = self.kb_info_client.get(str(kb_name)) # 判断库状态  # 这个拿的是可使用的，，要拿到之后存入另一个库
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
            self.max_update_num = int(config.get("max_update_num"))
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

            if kb_type == "DOC":
                time_start_count = time.time()
                doc_info = info_carrier['documentSources'] # 这个可能是多个
                doc_cfg = info_carrier['documentPreprocessConfig']
                process_cfg = doc_cfg
                index_cfg = info_carrier['indexConstructConfig']
                # 对于一次调用任务，所有的pipeline一致
                delete_a, document_writer, document_splitter = self.construct_update_pip(kb_name, 
                                                                                         doc_cfg = doc_cfg, 
                                                                                         index_cfg = index_cfg,
                                                                                         model_chat = model_chat,
                                                                                         chat_tokenizer = chat_tokenizer,
                                                                                         chinese_tokenizer_coarse = chinese_tokenizer_coarse,
                                                                                         chinese_tokenizer_fine = chinese_tokenizer_fine,
                                                                                     )
                
                pre_construct_type = self.kb_info_client.get_construct_type(str(kb_name)) # 整体判断即可
                if pre_construct_type == info_carrier['documentPreprocessConfig']:  # 前后两次状态相同
                    self.pre_lindorm_doc = self.kb_info_client.get_doc_info(str(kb_name))
                    # key 是文件的token
                    # value 是一个二元组
                    # (version, path)
                else: # 前后两次不同，没有需要从之前库获得的信息
                    self.pre_lindorm_doc = {}

                update_num = 0
                for sub_doc_info in doc_info:
                    wiki_token = sub_doc_info["url"].split("https://fintopia.feishu.cn/wiki/")[1].split('?from')[0]
                    if not sub_doc_info["syncRecursive"]: # 不递归
                        node_info, node_status = self.feishu_processor.get_node(wiki_token)
                        if not node_status:
                            # 如果权限不足，update_num不再加这篇文章，继续处理其他文章
                            continue
                        doc_version = self.feishu_processor.get_document_info(wiki_token)['data']['document']['revision_id']
                        if self.pre_lindorm_doc.get(wiki_token, {}) != {"version" : doc_version, "path" : node_info['title']}:
                            update_num += 1
                    else: # 递归
                        cur_recursive_num, cancled= self.get_recursive_num(wiki_token, doc_cfg['documentAugment'], kb_name)
                        if cancled:
                            self.post_process_kb_by_cancel(kb_name)
                            return {}, [], [], True
                        update_num += cur_recursive_num
                        
                        if update_num > self.max_update_num:
                            update_res =  {
                                "code": 1,
                                "message": "单次更新文档数量过多，请增量更新或联系寇梅如",
                                "data": {
                                    "documentCount": -1,
                                    "status": "FINISH",
                                    "details": []
                                }
                            }
                            self.post_process_kb_by_cancel(kb_name, update_res)
                            return {}, [], [], False
                time_end_count = time.time()
                print(time_end_count - time_start_count)
                if update_num > self.max_update_num:
                    update_res =  {
                        "code": 1,
                        "message": "单次更新文档数量过多，请增量更新或联系寇梅如",
                        "data": {
                            "documentCount": -1,
                            "status": "FINISH",
                            "details": []
                        }
                    }
                    self.post_process_kb_by_cancel(kb_name, update_res)
                    return {}, [], [], False

                doc_num = 0
                cur_lindorm_doc_info = {} # 这个是最后要存入lindorm的info
                for sub_doc_info in doc_info:
                    wiki_token = sub_doc_info["url"].split("https://fintopia.feishu.cn/wiki/")[1].split('?from')[0]
                    if not sub_doc_info["syncRecursive"]: # 不递归
                        node_info, node_status  = self.feishu_processor.get_node(wiki_token)
                        if not node_status:
                            all_fail_doc.append(wiki_token)
                            continue
                        doc_version = self.feishu_processor.get_document_info(wiki_token)['data']['document']['revision_id']
                        if self.pre_lindorm_doc.get(wiki_token, {}) == {"version" : doc_version, "path" : node_info['title']}:
                            self.get_chunk_from_lindorm(wiki_token)
                            cur_lindorm_doc_info[wiki_token] = self.pre_lindorm_doc[wiki_token]
                            doc_num += 1
                        else:
                            try:
                                image_fail = self.doc2txt(node_info['obj_token'], node_info['title'], kb_name, wiki_token, delete_a, document_writer, document_splitter,node_info['title'])
                                if self.kb_info_client.get_construct_type(str(999997)).get("cancel", "0") == "1":
                                    self.post_process_kb_by_cancel(kb_name)
                                    return {}, [], [], True
                                if image_fail:
                                    image_fail_all_doc.append(wiki_token)

                                # 这个是错的，入库会失败，要改一下
                                cur_lindorm_doc_info[wiki_token] = {"version":doc_version, "path" : node_info['title']}
                                doc_num += 1
                            except Exception as e:
                                all_fail_doc.append(wiki_token)
                                print(f"处理 {wiki_token} 时出错: {e}")
                                traceback.print_exc()

                    else: # 递归
                        # doc_num, retry_doc, image_fail_doc = self.doc2txt_Recursive(wiki_token, kb_name, doc_cfg['documentAugment'], doc_num, delete_a, document_writer, document_splitter)
                        doc_num, retry_doc, image_fail_doc , temp_lindorm_doc_info= self.doc2txt_Recursive(wiki_token, kb_name, doc_cfg['documentAugment'], doc_num, delete_a, document_writer, document_splitter)
                        if self.kb_info_client.get_construct_type(str(999997)).get("cancel", "0") == "1":
                            self.post_process_kb_by_cancel(kb_name)
                            return {}, [], [], True
                        all_fail_doc += retry_doc
                        image_fail_all_doc += image_fail_doc
                        cur_lindorm_doc_info.update(temp_lindorm_doc_info)
                # 这个函数要注意
                # if doc_num != 0:
                #     # 证明入库成功，否则为是吧is s
                #     # self.post_process_kb(kb_name, delete_a) # 删掉之前备份库
                #     self.post_process_kb(kb_name, delete_a, info_carrier['documentPreprocessConfig'], cur_lindorm_doc_info) # 删掉之前备份库
            else:
                # faq类型数据
                    faq_info = info_carrier['faqSources'] # 可能多个
                    faq_cfg = info_carrier['faqPreprocessConfig']
                    process_cfg = faq_cfg
                    # 在这里需要把知识库处理好
                    delete_a, doc_write = self.construct_update_pip_faq(kb_name)
                    pre_construct_type = self.kb_info_client.get_construct_type(str(kb_name)) # 整体判断即可
                    if pre_construct_type == info_carrier['faqPreprocessConfig']:  # 前后两次状态相同
                        self.pre_lindorm_doc = self.kb_info_client.get_doc_info(str(kb_name))
                        # key 是文件的token
                    else: # 前后两次不同，没有需要从之前库获得的信息
                        self.pre_lindorm_doc = {}


                    doc_num = 0
                    cur_lindorm_doc_info = {}
                    for sub_faq_info in faq_info:
                        try:
                            if self.kb_info_client.get_construct_type(str(999997)).get("cancel", "0") == "1":
                                self.post_process_kb_by_cancel(kb_name)
                                return {}, [], [], True
                            question_name = sub_faq_info['questionColumn']
                            answer_name = sub_faq_info['answerColumn']
                            if 'sheet' in sub_faq_info['url']:
                                sheet_name = sub_faq_info['url'].split('sheet=')[1]
                                if '?from' in sub_faq_info["url"]:
                                    faq_token = sub_faq_info["url"].split("https://fintopia.feishu.cn/wiki/")[1].split('?from')[0]
                                else:
                                    faq_token = sub_faq_info["url"].split("https://fintopia.feishu.cn/wiki/")[1].split('?sheet')[0]
                            else:
                                if '?from' in sub_faq_info["url"]:
                                    faq_token = sub_faq_info["url"].split("https://fintopia.feishu.cn/wiki/")[1].split('?from')[0]
                                else:
                                    faq_token = sub_faq_info["url"].split("https://fintopia.feishu.cn/wiki/")[1]
                                sheet_name = None
                            # query_list, answer_list, title = self.feishu_processor.get_faq_content(faq_token, sheet_name, question_name, answer_name)
                            # # 判断q入库，还是qa入库
                            # qa_pairs_num = self.faq2kb(faq_token, sheet_name, query_list, answer_list, faq_cfg ,model_chat,chat_tokenizer, title, delete_a, doc_write, kb_name)
                            # doc_num += qa_pairs_num
                            # 先判断是否在内，在内的话获取版本


                            faq_node_info, node_status = self.feishu_processor.get_node(faq_token)
                            if not node_status:
                                all_fail_doc.append(faq_token)
                                continue
                            version = faq_node_info.get('obj_edit_time', -1) # 把这个当作是version
                            if self.pre_lindorm_doc.get(faq_token, {}) == {"version" : version, "path" : None}: # 只需要比对版本
                                

                                faq_url = "https://fintopia.feishu.cn/wiki/" + faq_token
                                if sheet_name:
                                    faq_url += "?sheet="
                                    faq_url += sheet_name
                                qa_pairs_num = self.get_chunk_from_lindorm(faq_token, faq_url)
                                cur_lindorm_doc_info[faq_token] = self.pre_lindorm_doc[faq_token]
                                # 复制一份过去
                                doc_num += qa_pairs_num
                            
                            else:
                                
                                query_list, answer_list, title = self.feishu_processor.get_faq_content(faq_token, sheet_name, question_name, answer_name)
                                # 判断q入库，还是qa入库

                                # 处理这里


                                qa_pairs_num = self.faq2kb(faq_token, sheet_name, query_list, answer_list, faq_cfg ,model_chat,chat_tokenizer, title, delete_a, doc_write, kb_name)
                                doc_num += qa_pairs_num
                                # 也要维护
                                cur_lindorm_doc_info[faq_token] = {"version" : version, "path" : None}
                        except Exception as e:
                            all_fail_doc.append(faq_token)
                            print(f"处理 {faq_token} 时出错: {e}")
                            traceback.print_exc()
                        # if doc_num != 0:
                        #     # 证明入库成功，否则为是吧is s
                        #     self.post_process_kb(kb_name, delete_a, info_carrier['faqPreprocessConfig'], cur_lindorm_doc_info) # 删掉之前备份
                        #     # self.post_process_kb(kb_name, delete_a, info_carrier['faqPreprocessConfig'], cur_lindorm_doc_info)
                        #     # self.post_process_kb(kb_name, delete_a) # 删掉之前备份库
            response = {"outputs":
                         {
                            "documentCount": doc_num,
                            "status": "FINISH"
                         }
                       }
            # 只有在success的时候才需要update

            # 不对，整个update过程可能随时随地结束 
            # 直接暂停的话后处理都走不到，单独写一个info进去
            update_res = self.get_update_info(response, all_fail_doc, image_fail_all_doc)
            # 通过后处理写入
            if doc_num != 0: # 不等于0才要后处理吗，后处理是删除之前备份，所以如果等于0的话，相当于更新失败
                self.post_process_kb(kb_name, delete_a, process_cfg, cur_lindorm_doc_info, update_res) # 删掉之前备份
            else:
                self.post_process_kb_by_cancel(kb_name, update_res)
            return response, all_fail_doc, image_fail_all_doc, False
        except Exception as e:
            print(f"\n{traceback.format_exc()}\n")  # 测试用
            info_carrier['error'] = f"{self.__class__.__name__} error: {str(e)}"
            raise Exception(f"{self.__class__.__name__} error: {str(e)} ")




    def get_update_info(self, processed_data=None, failure_doc=None, image_failure_doc=None):
        """准备输出数据"""
        output_data = processed_data.get('outputs', None)
        if output_data is None:
            output_data = processed_data
        output_data['details'] = []
        if output_data['documentCount'] == 0:
            output_data['status'] = "FINISH"
            processed_data = {
                "code": 1,
                "message": "文档入库全部失败，请检查权限或联系管理人员寇梅如",
                "data":output_data
            }
            return processed_data


        if failure_doc is not None:
            for i in range(len(failure_doc)):
                output_data['details'].append({})
                url = os.path.join("https://fintopia.feishu.cn/wiki",failure_doc[i] )
                output_data['details'][i]["url"] = url
                output_data['details'][i]["reason"] = "权限不足，注意权限"
        if image_failure_doc is not None:
            for i in range(len(image_failure_doc)):
                output_data['details'].append({})
                url = os.path.join("https://fintopia.feishu.cn/wiki",image_failure_doc[i] )
                output_data['details'][i]["url"] = url
                output_data['details'][i]["reason"] = "文档中图片未处理成功，可能未加入协作组"  

        processed_data = {
            "code": 0,
            "message": "",
            "data":output_data

        }
        return processed_data

    def get_chunk_from_lindorm(self, wiki_token, faq_url=None):
        # 这里也要处理图片
        # 
        if faq_url:
            docs = self.document_store_read.filter_documents(filters={"meta.doc_url.keyword": faq_url}, all=True)
        else:
            docs = self.document_store_read.filter_documents(filters={"meta.doc_url.keyword": "https://fintopia.feishu.cn/wiki/"+wiki_token}, all=True)

        for doc in docs:
            # 如果 filter_documents 返回的是 haystack.Document 对象

            body = {
                "content": doc.content,
                "meta": doc.meta,
                "kb_id": doc.meta.get('source_kb')
            }

            self.document_store_write.client.index(
                index=self.document_store_write.index,
                id=doc.id,         # 保持相同 _id，或者去掉让 Lindorm 自动生成
                body=body
            )
        cur_env  = os.environ.get("env")
        oss_src_prefix = cur_env + str(self.kb_name) + "_kb_" + self.kb_used + wiki_token
        oss_des_prefix = cur_env + str(self.kb_name) + "_kb_" + self.kb_write + wiki_token
        # 要加上wiki信息
        self.oss_processor.cp_objects_by_prefix(oss_src_prefix, oss_des_prefix)
        return len(docs)


    def faq2kb(self,faq_token, sheet_name, query_list, answer_list, faq_cfg, model_chat,chat_tokenizer, title, delete_a, doc_write, kb_name):
        query_nmae = str(query_list[0][0]) if query_list[0][0] is not None else ""
        answer_name = str(answer_list[0][0]) if answer_list[0][0] is not None else ""
        faq_url = "https://fintopia.feishu.cn/wiki/" + faq_token
        if sheet_name:
            faq_url += "?sheet="
            faq_url += sheet_name
        documents_faq = []
        for i in range(1, len(query_list)):
            cur_query = str(query_list[i][0]) if query_list[i][0] is not None else ""
            cur_answer = str(answer_list[i][0]) if answer_list[i][0] is not None else ""
            if not cur_query or not cur_answer:
                continue
            if faq_cfg['constructType'] == 'QUESTION':
                # 单q入库
                query = cur_query
            else:
                # qa 入库
                query = query_nmae + ":" + cur_query + ", " + answer_name + ":" + cur_answer + "."
            if faq_cfg['keywordCount'] != 0:

                if str(faq_cfg['keywordCount']) == "1":
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
                        1. 本次必须提取 {str(faq_cfg['keywordCount'])} 个关键词
                        2. 返回结果中必须且只能包含 {str(faq_cfg['keywordCount'])} 个关键词
                        3. 关键词之间用 / 分隔
                        4. 不要返回解释，不要返回多余内容
                        5. 如果内容中可提取的关键词不足，也要尽量概括出 {str(faq_cfg['keywordCount'])} 个
                        6. 输出格式示例仅指分隔形式，例如：关键词1/关键词2/.../关键词{str(faq_cfg['keywordCount'])}

                        文字内容："""

                # 
                # prompt = "根据下面这段文字，提取" + str(faq_cfg['keywordCount']) + "个关键词，只返回关键词，不用返回其他内容，返回格式为：关键词1/关键词2/关键词3。文字内容为:"
                prompt += query
                response = aizt_chat(prompt, model_type="qwen3")
                pre_info = "<key_word>" + response + "<key_word_end>"
            else:
                pre_info = "<key_word> "  + "<key_word_end>"
            query = pre_info + query

            cur_doc = Document(content=query,
                               meta={"constructType": faq_cfg['constructType'],
                                    "related_answer": cur_answer,
                                    "query_name":query_nmae,
                                    "answer_name" : answer_name,
                                    "question": cur_query,
                                    "doc_url": faq_url,
                                    "title" : title,
                                    "source_kb": str(kb_name)})
            cur_doc.kb_id = str(kb_name)
            documents_faq.append(cur_doc)

        for idx in range(len(documents_faq)):
            sub_doc = documents_faq[idx].content
            embedding = aizt_embedding(sub_doc) # 这是在文档入库
            documents_faq[idx].embedding = embedding
        doc_write.run(documents=documents_faq)

        return len(documents_faq)


    def construct_update_pip_faq(self, kb_name):

        cur_env = os.environ.get("env")
        kb_used = self.kb_info_client.get(str(kb_name)) # 判断库状态  # 这个拿的是可使用的，，要拿到之后存入另一个库
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

        filter_doc = document_store.filter_documents(filters={"kb_id": str(kb_name)})
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

    def construct_update_pip(self, 
                           kb_name, 
                           doc_cfg = None, 
                           faq_cfg = None,
                           index_cfg = None, # 目前只支持0.6b模型，这个之后再改
                           model_chat = None,
                           chat_tokenizer = None,
                           chinese_tokenizer_coarse = None,
                           chinese_tokenizer_fine = None,
                           document_embedder = None):
        cur_env = os.environ.get("env")
        split_length = doc_cfg['splitLength']
        if doc_cfg['documentAugment']:
            split_length += 50
        if doc_cfg['paragraphAugment']:
            split_length *= 1.2
        if doc_cfg['keywordCount']:
            split_length += 50
        split_length = int(split_length)
        if doc_cfg['splitType'] == "RECURSIVE_DOCUMENT_SPLITTER":
            # 先把递归处理掉
            document_splitter = chinese_RecrucuitSplitter(split_length = split_length, 
                                                          split_overlap = doc_cfg['splitOverlap'], 
                                                          separators = doc_cfg['separators'],
                                                          paragraphAugment = doc_cfg['paragraphAugment'],
                                                          documentAugment = doc_cfg['documentAugment'],
                                                          keywordCount = doc_cfg['keywordCount'],
                                                          model_chat = model_chat,
                                                          chat_tokenizer = chat_tokenizer,
                                                          chinese_tokenizer_coarse = chinese_tokenizer_coarse,
                                                          chinese_tokenizer_fine = chinese_tokenizer_fine)
        else:

            document_splitter = chinese_DocumentSpliter(split_length = split_length, 
                                                        split_overlap = doc_cfg['splitOverlap'], 
                                                        separators = doc_cfg['separators'],
                                                        paragraphAugment = doc_cfg['paragraphAugment'],
                                                        documentAugment = doc_cfg['documentAugment'],
                                                        keywordCount = doc_cfg['keywordCount'],
                                                        # respectSentenceBoundary = doc_cfg['respectSentenceBoundary'],
                                                        respectSentenceBoundary = False,
                                                        splitThreshold = doc_cfg['splitThreshold'],
                                                        model_chat = model_chat,
                                                        chat_tokenizer = chat_tokenizer,
                                                        chinese_tokenizer_coarse = chinese_tokenizer_coarse,
                                                        chinese_tokenizer_fine = chinese_tokenizer_fine)


        kb_used = self.kb_info_client.get(str(kb_name)) # 判断库状态  # 这个拿的是可使用的，，要拿到之后存入另一个库
        if kb_used == -1 or kb_used == "b":
            # 证明 这个库是新建的，向a中写
            kb_write = "a"  # 写如a
            delete_a = False # 整个update结束，删除b
        elif kb_used == "a":
            kb_write = "b" 
            delete_a = True # 整个update 结束 删除 a

        document_store= LindormDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            index="kb_" + kb_write + "_" + cur_env,
            # index=str(kb_name) + "_1" + kb_write + cur_env,
            embedding_dim=1024,
        )

        # 先吧所有的index 切过去

        filter_doc = document_store.filter_documents(filters={"kb_id": str(kb_name)})
        if len(filter_doc) != 0:
            filter_document_ids = [doc.id for doc in filter_doc]  # 提取所有文档的 ID
            if filter_document_ids:  # 避免空列表导致的无效操作
                document_store.delete_documents(document_ids=filter_document_ids)
        document_writer = DocumentWriter(document_store)
        return delete_a, document_writer, document_splitter
    
    def replace_chart(self, content):
        if "<chart_s>" in content and "<chart_e>" in content:
            # 找到所有的<code>的位置
            positions_start = [m.start() for m in re.finditer(r"<chart_s>", content)]
            positions_end = [m.start() for m in re.finditer(r"<chart_e>", content)]
            if len(positions_start) != len(positions_end):
                return content
                # 这里先跳过，保底，证明切分过程出现问题
            for i in range(len(positions_start)):
                if positions_start[i] > positions_end[i]:
                    return content
            positions_start = positions_start[::-1]
            positions_end = positions_end[::-1]
            for sub_idx in range(len(positions_start)):  # 有几个东西
                chart_token = content[positions_start[sub_idx]+9:positions_end[sub_idx]].strip()
                chart_content = self.chart_mapping[chart_token]
                content = content[:positions_start[sub_idx]] + "相关Mermaid: " +  chart_content  +  content[positions_end[sub_idx]+9:]
        return content



    def replace_table(self, content, node_token):
        if "<tab_s>" in content and "<tab_e>" in content:
            # 找到所有的<code>的位置
            positions_start = [m.start() for m in re.finditer(r"<tab_s>", content)]
            positions_end = [m.start() for m in re.finditer(r"<tab_e>", content)]
            if len(positions_start) != len(positions_end):
                return content
                # 这里先跳过，保底，证明切分过程出现问题
            for i in range(len(positions_start)):
                if positions_start[i] > positions_end[i]:
                    return content
            positions_start = positions_start[::-1]
            positions_end = positions_end[::-1]
            for sub_idx in range(len(positions_start)):  # 有几个东西
                table_id = content[positions_start[sub_idx]+7:positions_end[sub_idx]].strip()
                cur_table = self.feishu_processor.get_table_content(node_token, table_id)
                content = content[:positions_start[sub_idx]] + "<tab_s>" +  cur_table +  "<tab_e>" +  content[positions_end[sub_idx]+7:]

                # 最后是没有 <code>的标识位的
        return content


    def replace_ocr(self, content):
        if "<ocr_s>" in content and "<ocr_e>" in content:
            # 找到所有的<code>的位置
            positions_start = [m.start() for m in re.finditer(r"<ocr_s>", content)]
            positions_end = [m.start() for m in re.finditer(r"<ocr_e>", content)]
            if len(positions_start) != len(positions_end):
                return content
                # 这里先跳过，保底，证明切分过程出现问题
            for i in range(len(positions_start)):
                if positions_start[i] > positions_end[i]:
                    return content
            positions_start = positions_start[::-1]
            positions_end = positions_end[::-1]
            for sub_idx in range(len(positions_start)):  # 有几个东西
                ocr_id = content[positions_start[sub_idx]+7:positions_end[sub_idx]].strip()
                ocr_content = self.ocr_mapping[ocr_id]
                # 这个最后要能删掉
                content = content[:positions_start[sub_idx]] + "<ocr_s>" + "图片ocr内容为"  + ocr_content +  "<ocr_e>" +  content[positions_end[sub_idx]+7:]
        return content


    def replace_code(self, content, node_token):
        if "<code>" in content and "<code_end>" in content:
            # 找到所有的<code>的位置
            positions_start = [m.start() for m in re.finditer(r"<code>", content)]
            positions_end = [m.start() for m in re.finditer(r"<code_end>", content)]
            if len(positions_start) != len(positions_end):
                return content
                # 这里先跳过，保底，证明切分过程出现问题
            for i in range(len(positions_start)):
                if positions_start[i] > positions_end[i]:
                    return content
            positions_start = positions_start[::-1]
            positions_end = positions_end[::-1]
            for sub_idx in range(len(positions_start)):  # 有几个东西
                code_id = content[positions_start[sub_idx]+6:positions_end[sub_idx]].strip()
                cur_code = self.feishu_processor.get_code_content(node_token, code_id)
                content = content[:positions_start[sub_idx]] + "相关code:" + "\n```\n" +  cur_code + "\n```\n  " +  content[positions_end[sub_idx]+10:]
        return content


    def post_process_kb_by_cancel(self, kb_name, update_res=None):
        # 被cancel 的数据库其实只需要更新一个update_res字段就好了
        # 或者就是先获取，再写入覆盖
        # 如果是首次更新呢
        # 
        
        # 用户手动停止
        # 手动停止的原因是什么
        # 怎么透传

        print("这个函数是否被调用了呢")
        construct_info = self.kb_info_client.get_construct_type(str(kb_name))
        doc_info = self.kb_info_client.get_doc_info(str(kb_name))
        if construct_info == -1:
            construct_info = {}
        if doc_info == -1:
            doc_info = {}
        kb_used = self.kb_info_client.get(str(kb_name)) # 判断库状态  # 这个拿的是可使用的，，要拿到之后存入另一个库
        if kb_used == -1:
            kb_used = "a"

        if not update_res:
            cancel_reason = self.kb_info_client.get_construct_type(str(999998))["cancel_reason"]
            update_res = {
                        "code": 1,
                        "message": cancel_reason,
                        "data": {
                            "documentCount": -1,
                            "status": "FINISH",
                            "details": []
                        }
                    }

        # 这里会写信息
        doc_info_list = [
            {"id": k, **v} for k, v in doc_info.items()
        ]
        self.kb_info_client.write(str(kb_name), kb_used, construct_info, doc_info_list, update_res)
        self.kb_info_client.write(str(999999), str(999999), kb_construct_type={'updating_kb': str(-1)}) # 写在kb_used中

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
                # index=str(kb_name) + "_1" + "a" + cur_env,
                embedding_dim=1024,
            )
            oss_prefix = cur_env + str(kb_name) + "_kb_" + "a"
        else:
            # self.kb_info_client.write(str(kb_name), "a") #改成a，删除b
            self.kb_info_client.write(str(kb_name), "a", construct_info, doc_info_list, update_res) 
            document_store = LindormDocumentStore(
                hosts=[{"host": self.lindorm_url, "port": 30070}],
                http_auth=(self.lindorm_key, self.lindorm_value),
                index="kb_" + "b" + "_" + cur_env,
                # index=str(kb_name) + "_1" + "b" + cur_env,
                embedding_dim=1024,
            ) 
            oss_prefix = cur_env + str(kb_name) + "_kb_" + "b"
        self.oss_processor.delete_objects_by_prefix(oss_prefix)
        filter_doc = document_store.filter_documents(filters={"kb_id": str(kb_name)})
        filter_document_ids = [doc.id for doc in filter_doc]  # 提取所有文档的 ID
        self.kb_info_client.write(str(999999), str(999999), kb_construct_type={'updating_kb': str(-1)}) # 写在kb_used中
        if filter_document_ids:  # 避免空列表导致的无效操作
            document_store.delete_documents(document_ids=filter_document_ids)
            print(f"已删除 {len(filter_document_ids)} 个文档")
        else:
            print("当前文档库中没有文档可删除")

    def doc2txt(self, wiki_token, title, kb_name, node_token, delete_a, document_writer, document_splitter,doc_title ):
        time_start = time.time()
        # block_content = self.feishu_processor.get_block_content(wiki_token)
        block_content = {}
        page_content = self.feishu_processor.get_block_content(wiki_token)
        block_content['items'] = page_content


        content = "<file_path> " + title + " <file_path_end>"
        content += block_content['items'][0][map_dict[block_content['items'][0]['block_type']]]['elements'][0]['text_run']['content']
        chunk_title_stack = []
        chunk_title_stack.append(doc_title)
        image_fail = False
        # 这个应该是文章名字
        processed_block = []
        for i in range(1, len(block_content['items'])):
            print(block_content['items'][i]['block_type'])
            if self.kb_info_client.get_construct_type(str(999997)).get("cancel", "0") == "1":
                self.post_process_kb_by_cancel(kb_name)
                return image_fail 
            # if block_content['items'][i]['block_id'] in processed_block:
            #     continue
            if block_content['items'][i]['parent_id'] in processed_block:
                processed_block.append(block_content['items'][i]['block_id'])
                continue
            if block_content['items'][i]['block_type'] < 14:
                # 都是文本部分
                if block_content['items'][i]['block_type'] > 2 and block_content['items'][i]['block_type'] < 10:  # 是几级表题的部分
                    # chunk_title_stack

                    cur_chunk_title = block_content['items'][i][map_dict[block_content['items'][i]['block_type']]]['elements'][0]['text_run']['content']
                    while len(chunk_title_stack) < (block_content['items'][i]['block_type'] - 2):
                        chunk_title_stack.append('')
                    while len(chunk_title_stack) > (block_content['items'][i]['block_type'] - 2):
                        chunk_title_stack.pop()
                    chunk_title_stack.append(cur_chunk_title)
                    chunk_info = '/'.join(chunk_title_stack)
                    content += "\n\n"  # 确定下什么时候会用到\n\n
                    content += "<chunk_info> " + chunk_info + " <chunk_info_end>"
                else:
                    content += "\n"
                # 这里调整过
                elements = block_content['items'][i][map_dict[block_content['items'][i]['block_type']]]['elements']
                # for j in range(len(elements)):
                #     if 'text_run' in elements[j]:
                #         content += elements[j]['text_run']['content']

                for j in range(len(elements)):
                    if "mention_user" in elements[j]:
                        content += " @" 
                        name = self.feishu_processor.get_user_name(elements[j]['mention_user']['user_id'])
                        content += name 
                    if 'text_run' in elements[j]:
                        content += elements[j]['text_run']['content']


            elif block_content['items'][i]['block_type'] == 14: # 处理代码块
                # 代码块部分的处理，存入id，然后可以根据id构建字典，拿到相应的代码块，一个代码块只会用到一次，用完delete掉key和相应的value
                content += "<code>"
                content += block_content['items'][i]['block_id'] 
                content += "<code_end>"

            elif block_content['items'][i]['block_type'] == 40:  # 流程图
                # 流程图第一种：由于code转换拿到的流程图
                try:
                    # record_str = block_content['items'][i]['add_ons']['record']
                    # clean_str = self.feishu_processor.extract_flowchart_from_block(record_str)
                    # block_id = block_content['items'][i]['block_id']
                    # self.chart_mapping[block_id] = clean_str
                    # content += "<chart_s>"
                    # content += block_id
                    # content += "<chart_e>"
                    record_str = block_content['items'][i]['add_ons']['record']
                    clean_str = self.feishu_processor.extract_flowchart_from_block(record_str)
                    block_id = block_content['items'][i]['block_id']
                    self.chart_mapping[block_id] = clean_str
                    content += "<chart_s>"
                    content += block_id
                    content += "<chart_e>"



                    cfg_path = "/tmp/puppeteer-config.json"

                    # ✅ 原子写配置，避免并发写入导致 JSON 半截
                    if not os.path.exists(cfg_path):
                        tmp_cfg = cfg_path + ".tmp"
                        with open(tmp_cfg, "w", encoding = "utf-8") as f:
                            json.dump({"args": ["--no-sandbox", "--disable-setuid-sandbox"]}, f)
                        os.replace(tmp_cfg, cfg_path)  # 原子替换


                    def mermaid_to_png(mermaid_text: str):
                        # 1) 写入临时 .mmd 文件
                        with tempfile.TemporaryDirectory() as tmpdir:
                            mmd_path = Path(tmpdir) / "diagram.mmd"
                            png_path = Path(tmpdir) / "diagram.png"

                            # 你的字符串里有中文，utf-8 必须
                            mmd_path.write_text(mermaid_text, encoding = "utf-8")

                            # 2) 用 npx 运行 mermaid-cli 生成 PNG
                            subprocess.run(
                                [
                                    "mmdc",
                                    "-p", cfg_path,  # ⭐ 新增：puppeteer 配置
                                    "-i", str(mmd_path),
                                    "-o", str(png_path),
                                    "-b", "white",
                                    "--width", "2000",
                                    "--height", "1200",
                                    "--scale", "1",
                                ],
                                check = True,
                            )

                            image_bin = png_path.read_bytes()
                        return image_bin

                    # mermaid_to_png(clean_str, "diagram.png")
                    image_bin = mermaid_to_png(clean_str)
                    b64_str = base64.b64encode(image_bin).decode("utf-8")
                    if delete_a:
                        temp_used = "b"
                    else:
                        temp_used = "a"
                    content += "相关流程图为<image>"
                    content += block_content['items'][i]['block_id']
                    cur_env = os.environ.get("env")
                    obj_name = cur_env + str(kb_name) + "_kb_" + temp_used + node_token + block_content['items'][i][
                        'block_id']
                    self.oss_processor.upload_file(obj_name, b64_str)
                    content += "<image_end>"
                except:
                    # image_fail = True
                    # 流程图报错不影响整体
                    continue

            
            elif block_content['items'][i]['block_type'] == 43: # 流程图
                try:
                    whiteboard_token = block_content['items'][i]['board']['token']

                    # API地址：https://open.feishu.cn/open-apis/board/v1/whiteboards/{whiteboard_id}/nodes
                    board_info = self.feishu_processor.get_board_content(whiteboard_token)
                    # 本质是清洗一下board_info
                    board_prompt = self.get_board_prompt(board_info)
                    board_content = aizt_chat(board_prompt, model_type="qwen3")
                    block_id = block_content['items'][i]['block_id']
                    self.chart_mapping[block_id] = board_content
                    content += "<chart_s>"
                    content += block_id
                    content += "<chart_e>"
                    image_bin = self.feishu_processor.download_whiteboard_png(whiteboard_token)
                    b64_str = base64.b64encode(image_bin).decode("utf-8")
                    if delete_a:
                        temp_used = "b"
                    else:
                        temp_used = "a"
                    content += "相关流程图为<image>"
                    content += block_content['items'][i]['block_id']
                    cur_env = os.environ.get("env")
                    obj_name = cur_env + str(kb_name) + "_kb_" + temp_used + node_token + block_content['items'][i]['block_id']
                    self.oss_processor.upload_file(obj_name, b64_str)
                    content += "<image_end>"
                except:
                    # image_fail = True
                    # 流程图报错不影响整体
                    continue

            elif block_content['items'][i]['block_type'] == 27: # 处理图片
                try:
                    image_bin = self.feishu_processor.get_image_content(node_token, block_content['items'][i]['block_id'])
                    
                    b64_str = base64.b64encode(image_bin).decode("utf-8")
                    if delete_a:
                        temp_used = "b"
                    else:
                        temp_used = "a"

                    content += "<image>"
                    content += block_content['items'][i]['block_id']
                    cur_env = os.environ.get("env")
                    obj_name = cur_env + str(kb_name) + "_kb_" + temp_used + node_token + block_content['items'][i]['block_id']
                    self.oss_processor.upload_file(obj_name, b64_str)
                    content += "<image_end>"
                    caption = block_content['items'][i].get("image", {}).get('caption', {}).get('content', None)
                    if caption:
                        content += "图片的caption为："
                        content += caption
                    try:
                        ocr_content = aizt_ocr(b64_str)
                    except:
                        print("获取ocr出错，但是程序继续运行")
                        ocr_content = ""
                    self.ocr_mapping[block_content['items'][i]['block_id']] = ocr_content
                    content += "<ocr_s>"
                    content += block_content['items'][i]['block_id']
                    content += "<ocr_e>"
                    # 如果处理失败，content 不加这段内容
                except:
                    image_fail = True
                    continue
            elif block_content['items'][i]['block_type'] == 31:
                processed_block.append(block_content['items'][i]['block_id'])
                content += "<tab_s>"
                content += block_content['items'][i]['block_id'] # 如何判断一个表格结束了呢
                content += "<tab_e>"

            else:
                continue


            if len(content) >= 50000:
                # 设置1w
                # 直接切分并写入
                # 把当前的content 去处理
                self.split_content(content, doc_title, node_token, kb_name, document_splitter, document_writer)
                # 重置contnet
                # 
                content = ""
                if i < len(block_content['items']) - 1:
                    
                    content = "<file_path> " + title + " <file_path_end>"
                    # 这个信息是啥啊
                    # content += block_content['items'][0][map_dict[block_content['items'][0]['block_type']]]['elements'][0]['text_run']['content']
                    if block_content['items'][i + 1]['block_type'] > 2 and block_content['items'][i + 1]['block_type'] < 10:
                        # 证明后续会拼接标题，不用管了
                        pass
                    else:
                        chunk_info = '/'.join(chunk_title_stack)

                        content += "<chunk_info> " + chunk_info + " <chunk_info_end>"
        if content:
            self.split_content(content, doc_title, node_token, kb_name, document_splitter, document_writer)
        return image_fail



    def split_content(self, content, doc_title, node_token, kb_name, document_splitter, document_writer):
        doc = Document(content=content)
        doc.meta['title'] = doc_title
        # 这个title的问题
        doc.meta['doc_url'] = "https://fintopia.feishu.cn/wiki/" + node_token # 加一点信息进去，看看能不能到最后        
        doc.meta['source_kb'] = str(kb_name)
        splitted_doc = document_splitter.run([doc])

        for idx in range(len(splitted_doc['documents'])):
            sub_doc = splitted_doc['documents'][idx].content
            new_content = self.replace_table(sub_doc, node_token)
            new_content = self.replace_ocr(new_content)
            new_content = self.replace_chart(new_content)
            new_content = self.replace_code(new_content, node_token)
            splitted_doc['documents'][idx].content = new_content
            embedding = aizt_embedding(new_content) # 这是在文档入库
            splitted_doc['documents'][idx].embedding = embedding
            splitted_doc['documents'][idx].kb_id = str(kb_name)
        # time_embedding= time.time()
        # logger.info('time：embedding的耗时: %s' % (time_embedding - time_feishu ))
        document_writer.run(documents=splitted_doc['documents'])


    def get_board_prompt(self, board_info):

        prompt = """请帮我将传入的数据转换为Mermaid flowchart LR格式，要求如下：

        **步骤一：获取白板节点数据**

        1. 解析传入中的 board_info：
        - 提取所有 type=composite_shape 的节点（文本框节点）
        - 识别所有 type=image 的节点（图片节点）
        - 提取所有 type=connector 的节点（连接线）

        **步骤二：处理节点关系**

        1. 对于图片节点（type=image）：
        - 跳过图片节点本身，不在flowchart中显示
        - 但要保持图片节点的父子关系继承
        - 即：如果 A → 图片 → B，则转换为 A → B

        2. 对于文本节点（type=composite_shape）：
        - 提取 text.text 字段作为节点内容
        - 保持所有原始文本，不截断

        3. 对于连接线（type=connector）：
        - 提取 connector.start_object.id 作为起点
        - 提取 connector.end_object.id 作为终点
        - 提取 connector.captions.data[0].text 作为连接线标签

        4. **连接去重优化**（重要）：
        - 如果同一对节点有多条连接（有的有标签、有的无标签）
        - 只保留带标签的连接，去除无标签的重复连接
        - 例如：node10 --> node11（无标签）和 node10 -->|"实验组1"|node11（有标签）
        - 只保留：node10 -->|"实验组1"| node11

        **步骤四：生成Mermaid flowchart LR格式**

        1. 节点定义格式：
        - 将原始文本中的换行符 \n 替换为 <br/>（实现自动换行）
        - 将引号 " 替换为 #quot;（避免语法冲突）
        - 格式：node1["文本内容<br/>第二行<br/>第三行"]

        2. 连接定义格式：
        - 有标签：nodeA -->|"标签文字"| nodeB
        - 无标签：nodeA --> nodeB

        3. 输出要求：
        - 第一行：flowchart LR
        - 节点定义：所有节点的定义
        - 连接关系：所有节点间的连接

        **最终输出要求**

        生成一个纯文本文件，内容为完整的flowchart LR代码，格式如下：

        "flowchart LR 为
            node1["完整文本内容<br/>可以<br/>自动<br/>换行"]
            node2["另一个节点"]
            ...
            node1 -->|"标签"| node2
            ...
        "

        **质量检查**

        -  所有composite_shape节点的文本都完整提取
        -  所有连接线的标签都正确提取
        -  图片节点已跳过但关系已保持
        -  使用<br/>实现文本自动换行
        -  引号正确转义
        -  如果某个节点或数据缺失，无需提示
        -  输出内容不允许出现形如「三个连续双引号」或 「三个连续单引号」的分隔符。
        -  输出纯文字，不要出现 ``` 这种字符
        -  如果解析不出来流程图，不要抱歉，返回 "无法解析该流程图" 即可
        -  即使数据量很大，也一定要展示所有节点和连接，不可以折叠
        -  不可省略任何节点和连接，给出完整输出
        传入的board_info值为
        """
        if len(str(board_info)) > 50000:
            board_info = self.shrink_board_info_strict(board_info)
        prompt += str(board_info)        
        return prompt   



    # def drop_geometry_fields(self, obj: dict) -> dict:
    #     """递归删除几何/样式字段（保守版）"""

    #     DROP_KEYS = {
    #         "x","y","z_index","width","height","angle",
    #         "style","rotation","transform","matrix"
    #     }
    #     if isinstance(obj, dict):
    #         new = {}
    #         for k, v in obj.items():
    #             if k in DROP_KEYS:
    #                 continue
    #             # turning_points 也很大，通常不需要
    #             if k == "turning_points":
    #                 continue
    #             new[k] = self.drop_geometry_fields(v)
    #         return new
    #     elif isinstance(obj, list):
    #         return [self.drop_geometry_fields(x) for x in obj]
    #     else:
    #         return obj

    # def shrink_board_info(self, board_info: list[dict]) -> list[dict]:
    #     return [self.drop_geometry_fields(o) for o in board_info]




    def shrink_board_obj_strict(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        更激进的清洗：
        - 顶层只保留 id/type/(locked)
        - composite_shape 只保留 {"type": "..."}
        - text 只保留 "text" 字符串（即 obj["text"]["text"]）
        - connector 保留最关键：caption 文本 + start/end 连接对象 id + shape(可选)
        """

        KEEP_TOP_LEVEL = {"id", "type", "locked"}   # locked 不需要可删
        # KEEP_TEXT = {"text"}                        # 这里只是语义说明，实际只取 text.text
        # KEEP_COMPOSITE_SHAPE = {"type"}             # composite_shape 只留 type
        t = obj.get("type")

        out: Dict[str, Any] = {
            k: obj.get(k) for k in KEEP_TOP_LEVEL if k in obj
        }

        # 1) text：只保留文字内容
        if isinstance(obj.get("text"), dict):
            txt = obj["text"].get("text")
            if isinstance(txt, str) and txt.strip():
                out["text"] = txt.strip()

        # 2) composite_shape：只保留其类型
        if isinstance(obj.get("composite_shape"), dict):
            ctype = obj["composite_shape"].get("type")
            if ctype:
                out["composite_shape"] = {"type": ctype}

        # 3) connector：保留连接关系 + caption 文本（其余几何信息全部扔掉）
        if t == "connector" and isinstance(obj.get("connector"), dict):
            conn = obj["connector"]

            # caption 文本（如果有）
            cap_texts: List[str] = []
            captions = (conn.get("captions") or {}).get("data") or []
            if isinstance(captions, list):
                for c in captions:
                    if isinstance(c, dict):
                        txt = c.get("text")
                        if isinstance(txt, str) and txt.strip():
                            cap_texts.append(txt.strip())
            if cap_texts:
                out["caption"] = " ".join(cap_texts)

            # start/end 挂载对象 id（最关键）
            start_id = (
                (conn.get("start_object") or {}).get("id")
                or (conn.get("start") or {}).get("attached_object", {}).get("id")
            )
            end_id = (
                (conn.get("end_object") or {}).get("id")
                or (conn.get("end") or {}).get("attached_object", {}).get("id")
            )
            if start_id:
                out["start_id"] = start_id
            if end_id:
                out["end_id"] = end_id

            # 可选：保留 shape（polyline/straight）
            shape = conn.get("shape")
            if shape:
                out["shape"] = shape

        return out

    def shrink_board_info_strict(self, board_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            self.shrink_board_obj_strict(o)
            for o in board_info
            if isinstance(o, dict)
        ]



    def get_recursive_num(self, wiki_token, documentAugment, kb_name):
        doc_num = 0
        stack = [(wiki_token, [])]
        while stack:
            if doc_num > 200:
                # 如果单个递归超过上限，无需继续计算
                return doc_num, False
            if self.kb_info_client.get_construct_type(str(999997)).get("cancel", "0") == "1":
                canceled = True
                return doc_num, canceled
                # self.post_process_kb_by_cancel(kb_name)
                # return {}, [], [], True
            # 是否父子增强体现在名字上
            cur_wiki_token, parent_res_list = stack.pop()  # 弹出栈顶元素（替代 popleft）

            cur_node_info, node_status = self.feishu_processor.get_node(cur_wiki_token)
            if not node_status:
                # 如果权限不足，则不处理这个wiki_token，父文档权限不足，则不处理子文档
                continue
            if cur_node_info['node_type'] == 'shortcut':
                cur_node_info , node_status = self.feishu_processor.get_node(cur_node_info['origin_node_token'])
            if not node_status:
                # 如果权限不足，则不处理这个wiki_token，父文档权限不足，则不处理子文档
                continue
            # 这里要有一处替换，如果判断这个node



            file_type = cur_node_info['obj_type']
            if file_type != "docx":
                continue # 递归过程只处理docx数据
            doc_version = self.feishu_processor.get_document_info(cur_wiki_token)['data']['document']['revision_id']
            title = cur_node_info['title']
            cur_title = title
            if documentAugment: 
                if parent_res_list:
                    parent_path = os.path.join(*parent_res_list)
                    title = os.path.join(parent_path, title)
            print(title)
            if self.pre_lindorm_doc.get(cur_wiki_token, {}) != {"version":doc_version, "path" : title }:
                doc_num += 1
                print("当前的doc_num是")
                print(doc_num)
            if cur_node_info['has_child']:
                parent_res_list.append(cur_node_info['title'])
                cur_space_id = cur_node_info['space_id']
                parent_node_token = cur_node_info['node_token']
                sub_node_list = self.feishu_processor.search_subnode_info(cur_space_id, parent_node_token)
                for sub_node in sub_node_list:
                    if sub_node['obj_type'] != 'docx':
                        continue
                    stack.append((sub_node['node_token'], copy.deepcopy(parent_res_list)))  # 入栈替代入队
            # stack中存储的永远都应该是  wiki_token wiki_token之前的parent_res_list
        return doc_num, False

    def doc2txt_Recursive(self, wiki_token, kb_name, documentAugment, doc_num, delete_a, document_writer, document_splitter):
        retry_doc = []
        image_fail_doc = []
        temp_lindorm_doc_info = {}
        stack = [(wiki_token, [])]
        while stack:
            # 是否父子增强体现在名字上
            cur_wiki_token, parent_res_list = stack.pop()  # 弹出栈顶元素（替代 popleft）
            cur_node_info, node_status = self.feishu_processor.get_node(cur_wiki_token)
            if not node_status:
                # 如果权限不足，则不处理这个wiki_token，父文档权限不足，则不处理子文档
                retry_doc.append(cur_wiki_token)
                continue
            if cur_node_info['node_type'] == 'shortcut':
                cur_node_info, node_status = self.feishu_processor.get_node(cur_node_info['origin_node_token'])
            if not node_status:
                retry_doc.append(cur_wiki_token)
                continue
            # 这里要有一处替换，如果判断这个node



            file_type = cur_node_info['obj_type']
            if file_type != "docx":
                continue # 递归过程只处理docx数据
            doc_version = self.feishu_processor.get_document_info(cur_wiki_token)['data']['document']['revision_id']
            title = cur_node_info['title']
            cur_title = title
            if documentAugment: 
                if parent_res_list:
                    parent_path = os.path.join(*parent_res_list)
                    title = os.path.join(parent_path, title)
            if self.pre_lindorm_doc.get(cur_wiki_token, {}) == {"version":doc_version, "path" : title }:
                self.get_chunk_from_lindorm(cur_wiki_token)
                doc_num += 1
                temp_lindorm_doc_info[cur_wiki_token] = self.pre_lindorm_doc[cur_wiki_token]
            else:
                try:

                    # f cur_wiki_token in pre_lindorm_doc and pre_lindorm_doc[cur_wiki_token][0] == doc_version and pre_lindorm_doc[cur_wiki_token][1] == title:
                    image_fail = self.doc2txt(cur_node_info['obj_token'], title, kb_name, cur_node_info['node_token'], delete_a, document_writer, document_splitter,cur_title)
                    if self.kb_info_client.get_construct_type(str(999997)).get("cancel", "0") == "1":
                        self.post_process_kb_by_cancel(kb_name)
                        
                        # 标志位设置之后，直接返回
                        return doc_num, retry_doc, image_fail_doc, temp_lindorm_doc_info
                    # 图片的存储以及获取可能会出问题
                    if image_fail:
                        image_fail_doc.append(cur_node_info['node_token'])
                    doc_num += 1
                    # 图片失败不算整体失败，进入图片失败的字典
                    temp_lindorm_doc_info[cur_wiki_token] = {"version":doc_version, "path" : title }
                except Exception as e:
                    # 整个失败直接进入这retry
                    retry_doc.append(cur_node_info['node_token'])
                    print(f"处理 {cur_node_info['node_token']} 时出错: {e}")
                    traceback.print_exc()

            if cur_node_info['has_child']:
                parent_res_list.append(cur_node_info['title'])
                cur_space_id = cur_node_info['space_id']
                parent_node_token = cur_node_info['node_token']
                sub_node_list = self.feishu_processor.search_subnode_info(cur_space_id, parent_node_token)
                for sub_node in sub_node_list:
                    if sub_node['obj_type'] != 'docx':
                        continue


                    # 这个地方append ori token
                    
                    stack.append((sub_node['node_token'],  copy.deepcopy(parent_res_list)))   # 入栈替代入队
        return doc_num, retry_doc, image_fail_doc, temp_lindorm_doc_info

class KbUpdateWrapper(Wrapper):
    def __init__(self, config=None):
        self.instance = Kb_Update()


class Cancel_Update:
    """Kb update Service"""
    def __init__(self):
        env = os.environ['env']
        # client = Client()
        self.lindorm_url = config.get("lindorm_url")
        self.lindorm_key = config.get("lindorm_key")
        self.lindorm_value = config.get("lindorm_value")
        self.kb_info_client = KnowledgeBaseABDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            index="kbab_" + env + "_new_3"
        )

        self.graph_info_client = KnowledgeBaseABDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            index="kbab_" + env + "_graph"
        )

    def process(self, info_carrier):
        print("任务被暂停，走暂停逻辑")
        kb_name = info_carrier['id']
        graph = info_carrier.get('isGraph', False)
        if not graph:
            cancel_reason = info_carrier.get("reason", "未知原因暂停")
            # 把cancel_reason存入lindorm
            self.kb_info_client.write(str(999998),str(999998), kb_construct_type = {"cancel_reason": cancel_reason}) # 写在kb_used中
            # cancel reason 写进去了99998，以备后续查询状态使用
            # 先写入应该是没问题的


            # -2 里面存着cancel_reason 这个信息

            updateing_kb = self.kb_info_client.get_construct_type(str(999999))['updating_kb']
            # 获取正在更新的知识库信息

            # 如果一致的话 cancel_event被set一下
            # 如果不一致的话，就直接返回response吧

            # 999997 用于存cancel event
            # 999996 用于存finish_event
            if str(kb_name) == str(updateing_kb):
                # 这里更新cancel
                self.kb_info_client.write(str(999997), str(999997), kb_construct_type={'cancel': str(1)}) # 写在kb_used中

                # 如果是1 的话，证明被cancel 了
                # 如果是0 的话  证明没被cancel
                print(f"[A] after set")
                # cancel_event.set() # 如果相等，则信号被设置
                return {"code": 0, "reason": "停止事件设置成功"}
            elif updateing_kb == "-1":
                return {"code": 1, "reason": "当前不存在知识库更新任务"}
            else:
                return {"code": 1, "reason": "暂停知识库和正在更新知识库不匹配"}
        else:

            cancel_reason = info_carrier.get("reason", "未知原因暂停")
            # 把cancel_reason存入lindorm
            self.graph_info_client.write(str(999998),str(999998), kb_construct_type = {"cancel_reason": cancel_reason}) # 写在kb_used中
            # cancel reason 写进去了99998，以备后续查询状态使用
            # 先写入应该是没问题的


            # -2 里面存着cancel_reason 这个信息

            updateing_kb = self.graph_info_client.get_construct_type(str(999999))['updating_kb']
            # 获取正在更新的知识库信息

            # 如果一致的话 cancel_event被set一下
            # 如果不一致的话，就直接返回response吧

            # 999997 用于存cancel event
            # 999996 用于存finish_event
            if str(kb_name) == str(updateing_kb):
                # 这里更新cancel
                self.graph_info_client.write(str(999997), str(999997), kb_construct_type={'cancel': str(1)}) # 写在kb_used中

                print(f"[A] after set")
                # cancel_event.set() # 如果相等，则信号被设置
                return {"code": 0, "reason": "graph停止事件设置成功"}
            elif updateing_kb == "-1":
                return {"code": 1, "reason": "graph当前不存在知识库更新任务"}
            else:
                return {"code": 1, "reason": "graph暂停知识库和正在更新知识库不匹配"}

class CancelUpdateWrapper(Wrapper):
    def __init__(self, config=None):
        self.instance = Cancel_Update()





class Get_Update_Status:
    """Kb update Service"""
    def __init__(self):
        env = os.environ['env']
        # client = Client()
        self.lindorm_url = config.get("lindorm_url")
        self.lindorm_key = config.get("lindorm_key")
        self.lindorm_value = config.get("lindorm_value")

        self.kb_info_client = KnowledgeBaseABDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            index="kbab_" + env + "_new_3"
        )

        self.graph_info_client = KnowledgeBaseABDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            index="kbab_" + env + "_graph"
        )


    def process(self, info_carrier):
        # 先获取状态
        kb_name = info_carrier['id']
        graph = info_carrier.get('isGraph', False)
        if not graph:
            if self.kb_info_client.get_construct_type(str(999996)).get("finish", "0") == "1":
                # 更新未完成的话，通过finish进行判断
                response_data = {
                                    "code": 0,
                                    "message": "",
                                    "data": {
                                        "documentCount": -1,
                                        "status": "UPDATEING",
                                        "details": []
                                    }
                                }
                # 我会直接返回一些结果
            else:
                response_data = self.kb_info_client.get_update_res(str(kb_name)) # 整体判断即可
            return response_data
            
        else:
            # 不会判断库信息吗
            # self.graph_info_client.get_construct_type(str(kb_name))
            if self.graph_info_client.get_construct_type(str(999996)).get("finish", "0") == "1":
                response_data = {
                                    "code": 0,
                                    "message": "",
                                    "data": {
                                        "documentCount": -1,
                                        "status": "UPDATEING",
                                        "details": []
                                    }
                                }
                # 没有结束时候的返回时这样子的
            else:
                # import  pdb
                # pdb.set_trace()
                response_data = self.graph_info_client.get_update_res(str(kb_name)) # 整体判断即可
                if response_data == -1:
                    # 证明从这个update_res拿不到，可能是因为delete
                    # 所以要从99995拿
                    response_data = self.graph_info_client.get_construct_type(str(999995))
                # response_data['operationType'] = "CREATE" # 这个是hard的
            return response_data

class GetUpdateStatusWrapper(Wrapper):
    def __init__(self, config=None):
        self.instance = Get_Update_Status()




class Get_Health_Status:
    def __init__(self):
        env = os.environ['env']
        # client = Client()
        self.lindorm_url = config.get("lindorm_url")
        self.lindorm_key = config.get("lindorm_key")
        self.lindorm_value = config.get("lindorm_value")
        self.kb_info_client = KnowledgeBaseABDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            index="kbab_" + env + "_new_3"
        )
        
        self.app_id = config.get("app_id")
        self.app_secret = config.get("app_secret")
        self.client = lark.Client.builder() \
            .app_id(self.app_id ) \
            .app_secret(self.app_secret) \
            .log_level(lark.LogLevel.DEBUG) \
            .build()


        self.pdf_endpoint=config.get("endpoint_pdf")

    # 其实就是3个ping
    def process(self):
        lindrom_health, error_message_lindorm = self.ping_lindorm()
        feishu_health, error_message_feishu = self.ping_feishu()
        pdf_parse_health, error_message_pdf = self.ping_pdf()
        chat_model_health, error_message_chat = check_aizt_chat()
        rerank_model_health, error_message_rerank = check_aizt_rerank()
        embedding_model_health, error_message_embedding = check_aizt_embedding()
        ocr_model_health, error_message_ocr = check_aizt_ocr()

        health_status = {
            "code": 0,
            "message": "",
            "data": {
            "status" : {
                "feishu": feishu_health,
                "lindorm": lindrom_health,
                "aliParse": pdf_parse_health,
                "chat_model": chat_model_health,
                "rerank_model": rerank_model_health,
                "embedding_model": embedding_model_health,          
                "ocr_model": ocr_model_health
            },
            "error": {
                "feishu": error_message_feishu,
                "aliParse": error_message_pdf,
                "lindorm": error_message_lindorm,
                "chat_model": error_message_chat,
                "rerank_model": error_message_rerank,
                "embedding_model": error_message_embedding,
                "ocr_model": error_message_ocr

            }
            }
        }

        return health_status
    
    def ping_lindorm(self):
        try:
            return self.kb_info_client.client.ping(), ""
            # 这个本身就可能会给false
            # 底下的是链接报错
        except Exception as e:
            print(f"[ping_lindorm] Connection failed: {type(e).__name__} - {e}")
            false_reasion = f"[ping_lindorm] Connection failed: {type(e).__name__} - {e}"
            import traceback
            print(traceback.format_exc())
            return False, false_reasion


    def ping_feishu(self):
        """
        检查飞书接口是否可用（仅请求一次）。
        Returns:
            bool: True 表示连通且凭证有效；False 表示失败。
        """
        url = "https://open.feishu.cn/open-apis/auth/v3/app_access_token/internal/"
        payload = {
            "app_id": self.app_id,
            "app_secret": self.app_secret
        }

        try:
            resp = requests.post(url, json=payload, timeout=5)
            data = resp.json()
            if data.get("code") == 0:
                print("✅ 飞书连接成功，凭证有效")
                return True, ""
            else:
                error_message = f"❌ 飞书连接失败: code={data.get('code')}, msg={data.get('msg')}"
                print(error_message)
                return False, error_message
        except Exception as e:
            error_message = f"[ping_feishu] 请求失败: {type(e).__name__} - {e}"
            print(error_message)
            import traceback
            print(traceback.format_exc())
            return False, error_message


    def ping_pdf(self):
        """
        检查 DocMind SDK 是否能成功初始化 + endpoint 网络可达。
        不调用实际解析接口。
        """
        try:
            # Step 1: 初始化 SDK client

            client = Local_File_Update().create_client()
            if not client:
                error_message = "❌ Client 初始化失败（None）"
                print(error_message)
                return False, error_message
            print("✅ DocMind Client 初始化成功")

            # Step 2: 检查 endpoint 网络连通性（简单 TCP 连接）
            
            u = urllib.parse.urlparse(self.pdf_endpoint)
            host = u.hostname or self.pdf_endpoint
            port = u.port or (443 if u.scheme == "https" else 80)

            with socket.create_connection((host, port), timeout=3):
                print(f"✅ Endpoint 可连接: {host}:{port}")
                return True, ""

        except Exception as e:
            error_message = f"❌ ping_pdf 检测失败: {type(e).__name__} - {e}"
            print(error_message)
            import traceback
            print(traceback.format_exc())
            return False, error_message


class GetHealthStatusWrapper(Wrapper):
    def __init__(self, config=None):
        self.instance = Get_Health_Status()
