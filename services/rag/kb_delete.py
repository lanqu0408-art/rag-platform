
import os
import json
import traceback
from typing import Optional, Union, Dict, List
from elasticsearch import Elasticsearch
from utils.wrapper import Wrapper
from services.external_api import ark_chat
# from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from .lindorm.lindorm import LindormDocumentStore
from .oss import Oss_Process
# from client.Client import Client
from core.config import config
# 这个其实是kb_delete
class Kb_Delete:
    """KB Process"""
    def __init__(self):
        # client = Client()
        self.lindorm_url = config.get("lindorm_url")
        self.lindorm_key = config.get("lindorm_key")
        self.lindorm_value = config.get("lindorm_value")
        self.oss_processor = Oss_Process()

    def delete_file(self, kb_id, file_id):
        cur_env = os.environ.get("env")
        document_store = LindormDocumentStore(
                hosts=[{"host": self.lindorm_url, "port": 30070}],
                http_auth=(self.lindorm_key, self.lindorm_value),
                index="kb_a_" + cur_env,
                embedding_dim=1024,
            ) 
        filters = {
            "kb_id": str(kb_id),
            "meta.doc_url.keyword": file_id
        }
        filter_doc = document_store.filter_documents(filters = filters)
        filter_document_ids = [doc.id for doc in filter_doc]  # 提取所有文档的 ID
        # all_documents = document_store.filter_documents(filters=None)
        # all_document_ids = [doc.id for doc in all_documents]  # 提取所有文档的 ID
        if filter_document_ids:  # 避免空列表导致的无效操作
            document_store.delete_documents(document_ids=filter_document_ids)
        # 这个暂时没有报错信息，功能简单，不会报错，后续可以加信息


    def delete(self, kb_name):
        # 删掉a 库 并删掉b库
        cur_env = os.environ.get("env")
        document_store = LindormDocumentStore(
                hosts=[{"host": self.lindorm_url, "port": 30070}],
                http_auth=(self.lindorm_key, self.lindorm_value),
                index="kb_" + "a" + "_" + cur_env,
                embedding_dim=1024,
            ) 
        filter_doc = document_store.filter_documents(filters={"kb_id": str(kb_name)})
        filter_document_ids = [doc.id for doc in filter_doc]  # 提取所有文档的 ID
        # all_documents = document_store.filter_documents(filters=None)
        # all_document_ids = [doc.id for doc in all_documents]  # 提取所有文档的 ID
        if filter_document_ids:  # 避免空列表导致的无效操作
            document_store.delete_documents(document_ids=filter_document_ids)
           

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

        cur_env = os.environ.get("env")
        oss_prefix = cur_env + str(kb_name) + "_kb_"
        self.oss_processor.delete_objects_by_prefix(oss_prefix)
        
    def process(self, info_carrier: dict) -> dict:
        try:
            kb_name = info_carrier["id"]
            task_type = info_carrier['action']

            if task_type == "DELETE":
                self.delete(kb_name)
            elif task_type == "DELETE_FILE":
                # 删掉某个本地上传的文件
                file_id = info_carrier["fileId"]
                self.delete_file(kb_name, file_id) # 之前都是删除某个库，现在是单独文件的删除
            
            # 只有成功，没有失败
            response = {"outputs":
                         {
                            "status": "SUCCESS"
                         }
                       }
            return response

        except Exception as e:
            print(f"\n{traceback.format_exc()}\n")  # 测试用
            info_carrier['error'] = f"{self.__class__.__name__} error: {str(e)}"
            raise Exception(f"{self.__class__.__name__} error: {str(e)} ")


class KbDeleteWrapper(Wrapper):
    def __init__(self, config=None):
        # print("启动服务时被调用")
        self.instance = Kb_Delete()