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
from alibabacloud_docmind_api20220711.client import Client as DocmindClient
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_docmind_api20220711 import models as docmind_models
from alibabacloud_tea_util import models as util_models
import io

import socket, urllib.parse
import requests


class Local_File_Update:
    """Kb update Service"""
    def __init__(self):
        self.oss_processor = Oss_Process(bucket_name="ai-rag-files")
        env = os.environ['env']

        # client = Client()
        self.lindorm_url = config.get("lindorm_url")
        self.lindorm_key = config.get("lindorm_key")
        self.lindorm_value = config.get("lindorm_value")

        self.kb_info_client = KnowledgeBaseABDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            index="kbab_local_file_" + env
        )  # 这个用于文件层面的判断

        self.kb_info_client_cur_kb = KnowledgeBaseABDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            index="kbab_" + env + "_new_3"
        )


        self.document_store= LindormDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            index="kb_a_" + env,
            embedding_dim=1024,
        )

        # 这个对吗

        self.access_key_id=config.get("ak_pdf")
        self.access_key_secret=config.get("sk_pdf")
        self.pdf_endpoint=config.get("endpoint_pdf")



    def process(self,
                info_carrier,
                chinese_tokenizer_coarse = None,
                chinese_tokenizer_fine = None):
        # # 先获取状态
        try:
            kb_id = info_carrier['id'] # 要存入的知识库

            # 在这里判断，然后删除


            localfilesources = info_carrier['localFileSources']
            # file_url = localfilesources['url']
            file_id = localfilesources['fileId']
            self.post_process(kb_id, file_id)
            object_name = localfilesources['url']
            file_name = localfilesources['fileName']
            doc_cfg = info_carrier['documentPreprocessConfig']
            self.kb_info_client_cur_kb.write(kb_id, "a") # 所有的pdf文件都存入a
            self.document_writer, self.document_splitter = self.construct_update_pip(doc_cfg,
                                                                        chinese_tokenizer_coarse = chinese_tokenizer_coarse,
                                                                        chinese_tokenizer_fine = chinese_tokenizer_fine)
            # 写到相应的知识库里面去

            # 这个有问题
            # import pdb
            # pdb.set_trace()
            self.kb_info_client.write(file_id, str(kb_id), kb_construct_type={'update_status': str("UPDATING")})
            # self.kb_info_client.write_or_append(file_id, kb_id, kb_construct_type={'update_status': str("UPDATING")}) # 写在kb_used中
            # file = self.oss_processor.download_file(file_url)
            file_url=self.oss_processor.get_url(object_name)
            # pdf_bytes = self.oss_processor.download_pdf_content(file_url)
            # 先尝试在这两步停下来
            update_status, cancel_reason = self.kb_info_client.get_local_file_status(str(kb_id), file_id)
            if update_status == "CANCEL":
                self.post_process(kb_id, file_id)
                response = {
                    "code": 1,
                    "message": cancel_reason,
                    "data": {
                        "documentCount": -1,
                        "status": "FINISH",
                        "details": []
                    }
                }
                self.kb_info_client.write(file_id, kb_id, kb_construct_type={'update_status': str("FINISH")})
                return response

            cancel_job = self.parse_update_pdf(file_url, file_name, kb_id, file_id) # 这里应该是提取出来了


            print(file_name, "parse_update_pdf 函数成功")
            print(file_name)
            # if len(file_content) > 100:
            #     print(file_content[:100])
            # else:
            #     print(file_content)

            # cancel_job 需要做一些后处理
            # 把之前加载的当篇文章删掉
            if cancel_job:
                update_status, cancel_reason = self.kb_info_client.get_local_file_status(str(kb_id), file_id)
                self.post_process(kb_id, file_id)
                response = {
                    "code": 1,
                    "message": cancel_reason,
                    "data": {
                        "documentCount": -1,
                        "status": "FINISH",
                        "details": []
                    }
                }
                self.kb_info_client.write(file_id, kb_id, kb_construct_type={'update_status': str("FINISH")})

                return response


            # content = '<file_path>' + file_name.split(".")[0] + '<file_path_end>' + file_content

            # print(file_name, "construct_update_pip 函数成功")
            # # print(file_name)
            # self.split_content(content, file_name, kb_id, document_splitter, document_writer, file_id)
            # print(file_name, "split_content 函数成功")
            response = {
                            "code": 0,
                            "message": "",
                            "data": {}
                     }



            self.kb_info_client.write(file_id, kb_id, kb_construct_type={'update_status': str("FINISH")})
            return response
        except Exception as e:
            print("进入except函数")
            kb_id = info_carrier['id'] # 要存入的知识库
            file_id = localfilesources['fileId']
            self.post_process(kb_id, file_id)
            self.kb_info_client.write(file_id, kb_id, kb_construct_type={'update_status': str("FINISH")})
            print(f"\n{traceback.format_exc()}\n")
            info_carrier['error'] = f"{self.__class__.__name__} error: {str(e)}"
            raise Exception(f"{self.__class__.__name__} error: {str(e)} ")


    def post_process(self, kb_id, file_id):
        filters = {
            "kb_id": str(kb_id),
            "meta.doc_url.keyword": file_id
        }
        filter_doc = self.document_store.filter_documents(filters = filters)
        filter_document_ids = [doc.id for doc in filter_doc]  # 提取所有文档的 ID
        if filter_document_ids:  # 避免空列表导致的无效操作
            self.document_store.delete_documents(document_ids=filter_document_ids)



    def ping_pdf(self) -> bool:
        """
        检查 DocMind SDK 是否能成功初始化 + endpoint 网络可达。
        不调用实际解析接口。
        """
        try:
            # Step 1: 初始化 SDK client
            client = self.create_client()
            if not client:
                print("❌ Client 初始化失败（None）")
                return False
            print("✅ DocMind Client 初始化成功")

            # Step 2: 检查 endpoint 网络连通性（简单 TCP 连接）

            u = urllib.parse.urlparse(self.pdf_endpoint)
            host = u.hostname or self.pdf_endpoint
            port = u.port or (443 if u.scheme == "https" else 80)

            with socket.create_connection((host, port), timeout=3):
                print(f"✅ Endpoint 可连接: {host}:{port}")
                return True

        except Exception as e:
            print(f"❌ ping_pdf 检测失败: {type(e).__name__} - {e}")
            import traceback
            print(traceback.format_exc())
            return False


    def create_client(self):
        config=open_api_models.Config(
            access_key_id=self.access_key_id,
            access_key_secret=self.access_key_secret,
            endpoint=self.pdf_endpoint
        )
        return DocmindClient(config)

    # def submit_pdf_job_from_bytes(self, client,file_bytes, file_name):
    #     """
    #     从内存中的 PDF 字节流提交解析任务（优化版）
    #     """
    #     file_stream=io.BytesIO(file_bytes)
    #     file_size_mb=len(file_bytes)/1024/1024
    #     print(f"[DocMind] 正在提交 PDF: {file_name} ({file_size_mb:.2f} MB)")
    #
    #     request=docmind_models.SubmitDocStructureJobAdvanceRequest(
    #         file_url_object=file_stream,
    #         file_name=file_name,
    #         structure_type="default",
    #         formula_enhancement=True
    #     )
    #
    #     # 增强超时和重试配置
    #     runtime=util_models.RuntimeOptions(
    #         connect_timeout=20000,  # 连接超时 20s
    #         read_timeout=180000,  # 读超时 180s
    #         autoretry=True
    #     )
    #
    #     try:
    #         response=client.submit_doc_structure_job_advance(request, runtime)
    #         job_id=response.body.data.id
    #         print(f"[DocMind] 任务提交成功，Job ID: {job_id}")
    #         return job_id
    #     except Exception as e:
    #         print(f"[Error] 提交 PDF 解析任务失败: {e}")
    #         raise
    def submit_pdf_job_from_bytes(self, client, file_url, file_name):
        """
           提交文档解析任务并轮询获取结果（包含所有阶段的 RequestId 打印）
           """
        print("\n====== 1. 提交任务 Submit ======\n")

        # ==========================
        # 1) 提交任务
        # ==========================
        request=docmind_models.SubmitDocStructureJobRequest(
            file_url=file_url,
            file_name=file_name,
        )

        # 增强超时和重试配置
        runtime=util_models.RuntimeOptions(
            connect_timeout=20000,  # 连接超时 20s
            read_timeout=180000,  # 读超时 180s
            autoretry=True
        )

        try:
            response=client.submit_doc_structure_job_with_options(request, runtime)
            job_id=response.body.data.id
            print(f"[DocMind] 任务提交成功，Job ID: {job_id}")
            return job_id
        except Exception as e:
            print(f"[Error] 提交 PDF 解析任务失败: {e}")
            raise

    def poll_result(self, client, job_id, max_wait=300, kb_id=None, file_id=None):
        """轮询查询结果（增强版：支持异常重试、防止 503）"""
        wait_interval=10  # 秒
        max_retries=int(max_wait*60/wait_interval)
        consecutive_failures=0
        max_failures=5  # 最多连续 5 次失败才放弃

        for i in range(max_retries):
            # 检查是否被用户取消
            update_status, _=self.kb_info_client.get_local_file_status(str(kb_id), file_id)
            if update_status=="CANCEL":
                print("[DocMind] 检测到任务被取消")
                return None, True

            # 构建请求
            request=docmind_models.GetDocStructureResultRequest(
                id=job_id,
                reveal_markdown=True,
                image_strategy="url"
            )

            try:
                response=client.get_doc_structure_result(request)
                body=response.body
            except Exception as e:
                consecutive_failures+=1
                print(f"[Warning] 第 {i+1} 次轮询异常: {e}")
                if consecutive_failures>=max_failures:
                    print("[DocMind] 连续多次请求失败，终止任务。")
                    return None, False
                time.sleep(wait_interval)
                continue

            consecutive_failures=0  # 重置失败计数

            if not body.completed:
                print(f"[DocMind] 任务处理中... ({i+1}/{max_retries})")
                time.sleep(wait_interval)
                continue

            if body.status=="Success":
                print(f"[DocMind] 解析成功！（Job ID: {job_id}）")
                return body.data, False
            else:
                print(f"[DocMind] 解析失败: {body.status} - {body.message}")
                return None, False

        print("[DocMind] 轮询超时未完成解析。")
        return False, False

        # 第二个返回值用于判断cancel

    def concat_update_pdf(self, layout,  file_name, kb_id, file_id):
        """
        拼接 layout 中的正文内容，去除页眉、页脚、页码等。
        """
        exclude_types={"foot_pagenum", "header", "footer", "page_header", "page_footer", "title"}  # 可扩展

        texts=[]

        for block in layout:
            block_type=block.get("type", "").lower()
            text=(block.get("text") or "").strip()

            # 如果是正文类型才拼接
            if block_type not in exclude_types and text:
                # 过滤掉明显的页眉页脚关键词（保险起见）
                if not any(k in text for k in ["页眉", "页脚", "第", "页"]):
                    texts.append(text)
                if len("\n".join(texts)) > 10000:
                    cur_content = "\n".join(texts)
                    print(file_name, "construct_update_pip 函数成功")
                    # print(file_name)
                    self.split_content(cur_content, file_name, kb_id, file_id)
                    texts = []
                    update_status, _=self.kb_info_client.get_local_file_status(str(kb_id), file_id)
                    if update_status=="CANCEL":
                        return True

        if len(texts) > 0:
            cur_content = "\n".join(texts)
            self.split_content(cur_content, file_name, kb_id, file_id)
        return False



    def parse_update_pdf(self, file_url, file_name, kb_id, file_id):
        # 初始化客户端
        client=self.create_client()
        # 1. 提交任务
        job_id=self.submit_pdf_job_from_bytes(client, file_url, file_name)

        # 2. 轮询结果
        result_data, cancel_job=self.poll_result(client, job_id, kb_id=kb_id, file_id=file_id)
        if cancel_job:
            return True
        if not result_data:
            return False

        # 这个名字应该改动一下的，这个其实是把文档已经入库完成了
        cancel_job = self.concat_update_pdf(result_data['layouts'], file_name, kb_id, file_id)
        return cancel_job


    def split_content(self, content, doc_title, kb_id, file_id ):
        content = '<file_path>' + doc_title.split(".")[0] + '<file_path_end>' + content
        doc = Document(content=content)
        doc.meta['title'] = doc_title
        # 这个title的问题
        doc.meta['doc_url'] = file_id # 加一点信息进去，看看能不能到最后
        doc.meta['source_kb'] = str(kb_id)
        splitted_doc = self.document_splitter.run([doc])

        for idx in range(len(splitted_doc['documents'])):
            sub_doc = splitted_doc['documents'][idx].content
            embedding = aizt_embedding(sub_doc) # 这是在文档入库
            splitted_doc['documents'][idx].embedding = embedding
            splitted_doc['documents'][idx].kb_id = str(kb_id)
        self.document_writer.run(documents=splitted_doc['documents'])

    def construct_update_pip(self,
                             doc_cfg = None,
                             model_chat = None,
                             chat_tokenizer = None,
                             chinese_tokenizer_coarse = None,
                             chinese_tokenizer_fine = None):
        cur_env = os.environ.get("env")
        split_length = doc_cfg['splitLength']
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
                                                        respectSentenceBoundary = False,
                                                        splitThreshold = doc_cfg['splitThreshold'],
                                                        model_chat = model_chat,
                                                        chat_tokenizer = chat_tokenizer,
                                                        chinese_tokenizer_coarse = chinese_tokenizer_coarse,
                                                        chinese_tokenizer_fine = chinese_tokenizer_fine)

        document_store= LindormDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            index="kb_a_" + cur_env,
            embedding_dim=1024,
        )

        document_writer = DocumentWriter(document_store)
        return  document_writer, document_splitter

class LocalFileUpdateWrapper(Wrapper):
    def __init__(self, config=None):
        self.instance = Local_File_Update()





class Cancel_Parse:
    """Kb Parse Service"""
    def __init__(self):
        env = os.environ['env']
        self.lindorm_url = config.get("lindorm_url")
        self.lindorm_key = config.get("lindorm_key")
        self.lindorm_value = config.get("lindorm_value")
        self.kb_info_client = KnowledgeBaseABDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            index="kbab_local_file_" + env
        )



    def process(self, info_carrier):
        # logger.info("任务被暂停，走暂停逻辑")
        kb_id = info_carrier['id']
        cancel_reason = info_carrier.get("reason", "未知原因暂停")
        file_id = info_carrier['fileId']
        # 把cancel_reason存入lindorm
        update_status, _ = self.kb_info_client.get_local_file_status(str(kb_id), file_id)
        if update_status == -1:
            # 证明当前文档没在更新或已经更新完成
            return {"code": 1, "message": "当前文档未在更新中"}
        else:
            # self.kb_info_client.write(file_id, str(kb_id), kb_construct_type={'update_status': str("UPDATING")})
            self.kb_info_client.write(file_id, kb_id, kb_construct_type={'update_status': str("CANCEL"), "cancel_reason":cancel_reason}) # 写在kb_used中
            return {"code": 0, "message": "停止事件设置成功"}


class CancelParseWrapper(Wrapper):
    def __init__(self, config=None):
        self.instance = Cancel_Parse()