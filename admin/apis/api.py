# encoding=utf-8
import json
import logging
import traceback
from flask import Blueprint, g, Response
from utils.request_processor import RequestProcessor
from services.rag import KbDeleteWrapper, KbUpdateWrapper, RAGWrapper, RetrieverWrapper, CancelUpdateWrapper,GetUpdateStatusWrapper, KnowledgeBaseABDocumentStore, LocalFileUpdateWrapper, CancelParseWrapper, GetHealthStatusWrapper, GraphUpdateWrapper, FaqApiSyncWrapper
# from transformers import AutoModelForCausalLM, AutoTokenizer
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from typing import Dict, Any
from flask import Response, stream_with_context
import threading
import json
from core.config import config
from core.hanlp_runtime import ensure_hanlp_env, get_chinese_tokenizers
# from client.Client import Client
import os
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=5)

logger = logging.getLogger(__name__)
api_blueprint = Blueprint('api_blueprint', __name__, url_prefix='/ai-rag-platform')
health = Blueprint('health', __name__, url_prefix='/')

ensure_hanlp_env()
tokenizer = None
document_embedder = None
text_embedder = None 
text_embedder = None
model_Qwen_7B = None


# 健康检查，项目上线用不要删除
@health.route('/healthCheck', methods=["GET", "POST"])
def api_get_health():
    result = {'status': 'running'}
    return Response(json.dumps(result), mimetype='application/json')


@api_blueprint.route("/test", methods=['GET', 'POST'])
def get_test_result():
    result_dict = {'status': 'running'}
    return Response(json.dumps(result_dict), mimetype='application/json')

class KbUpdateConstructor(RequestProcessor):
    def __init__(self, operator):
        super().__init__('kb-update', operator)
        env = os.environ['env']
        # self.lindorm_url = config.get("lindorm_url")
        # self.lindorm_key = config.get("lindorm_key")
        # self.lindorm_value = config.get("lindorm_value")
        # client = Client()
        self.lindorm_url = config.get("lindorm_url")
        self.lindorm_key = config.get("lindorm_key")
        self.lindorm_value = config.get("lindorm_value")
        self.kb_info_client = KnowledgeBaseABDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            index="kbab_" + env + "_new_3"
        )

    def process_request(self, raw_request: Dict[str, Any]) -> str:
        """处理请求的主流程"""
        try:
            request_id = raw_request.get('request_id', 'None')
            # 准备请求数据
            request_data = self.prepare_request_data(raw_request)
            chinese_tokenizer_coarse, chinese_tokenizer_fine = get_chinese_tokenizers()

            future = executor.submit(self.operator.process, 
                                    request_data,
                                    model_chat = model_Qwen_7B,
                                    chat_tokenizer = tokenizer,
                                    chinese_tokenizer_coarse = chinese_tokenizer_coarse,
                                    chinese_tokenizer_fine = chinese_tokenizer_fine,
                                    document_embedder = document_embedder,
                                    text_embedder = text_embedder)

            future.add_done_callback(lambda f: self._handle_result(f, request_id))

            response_data = {
                                "code": 0,
                                "message": "",
                                "data": {
                                    "documentCount": -1,
                                    "status": "UPDATEING",
                                    "details": []
                                }
                            }
            return Response(json.dumps(response_data, ensure_ascii=False), mimetype='application/json')
        except Exception as e:
            return self.handle_error(e, request_id)

    def _handle_result(self, future, request_id):
        """异步任务执行完成后自动调用"""
        try:
            processed_data, failure_doc, image_failure_doc, iscancel = future.result()

            # 异步拿到结果的时候处理
            if iscancel:
                print(f"[{request_id}] 用户取消任务")
                # cancel_event.clear()
            else:
            # cancel 之后后续任务无需处理

                # 打印或处理结果
                self.log_processing_result(processed_data)
            self.kb_info_client.write(str(999997), str(999997), kb_construct_type={'cancel': str(0)})
            self.kb_info_client.write(str(999996), str(999996), kb_construct_type={'finish': str(0)})

            # TODO: 这里可以存数据库、写文件、发回调通知等

        except Exception as e:
            self.kb_info_client.write(str(999997), str(999997), kb_construct_type={'cancel': str(0)})
            self.kb_info_client.write(str(999996), str(999996), kb_construct_type={'finish': str(0)})
            print(f"[{request_id}] 异步任务失败 ❌: {e}")


    def prepare_response_data(self, processed_data, failure_doc, image_failure_doc):
        """准备输出数据"""
        print('这个函数应该没用了')


kb_update = KbUpdateConstructor(
    KbUpdateWrapper().get_instance())
@api_blueprint.route(kb_update.server_name, methods=['GET', 'POST'])

def get_kb_update_result():
    return kb_update.process_request(g.request_data)

class KBDeleteConstructor(RequestProcessor):
    def __init__(self, operator):
        super().__init__('kb-delete', operator)

kb_delete = KBDeleteConstructor(
    KbDeleteWrapper().get_instance())
@api_blueprint.route(kb_delete.server_name, methods=['GET', 'POST'])
def get_kb_delete_result():
    return kb_delete.process_request(g.request_data)



class RAGConstructor(RequestProcessor):
    def __init__(self, operator):
        super().__init__('stream', operator)

    def process_request(self, raw_request):
        try:
            request_id = raw_request.get('request_id', 'None')
            request_data = self.prepare_request_data(raw_request)
            if request_data.get('stream', True) == True:

                @stream_with_context
                def generate():
                    try:
                        for chunk in self.operator.process_stream(
                            request_data,
                            model_chat=model_Qwen_7B,
                            chat_tokenizer=tokenizer,
                            text_embedder=text_embedder
                        ):
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    except Exception as e:
                        yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

                return Response(generate(), mimetype='text/event-stream')
            else:
                response_data = self.operator.process_unstream(request_data, model_chat=model_Qwen_7B)
                return Response(json.dumps(response_data, ensure_ascii=False), mimetype='application/json')

        except Exception as e:
            return self.handle_error(e, request_id)


rag = RAGConstructor(
    RAGWrapper().get_instance())
@api_blueprint.route(rag.server_name, methods=['GET', 'POST'])
def get_rag_result():
    return rag.process_request(g.request_data)

class RetrieverConstructor(RequestProcessor):
    def __init__(self, operator):
        super().__init__('retrieval', operator)

    def process_request(self, raw_request):
        try:
                request_id = raw_request.get('request_id', 'None')
                request_data = self.prepare_request_data(raw_request)
                processed_data = self.operator.process(request_data, 
                                                       model_chat=model_Qwen_7B,
                                                       chat_tokenizer=tokenizer,
                                                       text_embedder=text_embedder)
                self.log_processing_result(processed_data)
                
                # 准备返回数据
                response_data = self.prepare_response_data(processed_data)

                return Response(json.dumps(response_data, ensure_ascii=False), mimetype='application/json')
        except Exception as e:
            return self.handle_error(e, request_id)


retriever = RetrieverConstructor(
    RetrieverWrapper().get_instance())








@api_blueprint.route(retriever.server_name, methods=['GET', 'POST'])
def get_retriever_result():
    return retriever.process_request(g.request_data)

class CancelUpdateConstructor(RequestProcessor):
    def __init__(self, operator):
        super().__init__('cancel-update', operator)

    def handle_error(self, error: Exception, request_id: str) -> str:
        """处理过程中发生的错误"""
        error_msg = f'request_id: {request_id}. Failed to call server: {self.server_name}. error: {str(error)}'
        logger.error(f"{error_msg}\n{traceback.format_exc()}\n")
        # response = {
        #     "code": 1,
        #     "detail": error_msg,
        #     "data":{}
        # }

        response = {"code": 1, "reason": "停止失败："+error_msg}
        return response

    def process_request(self, raw_request):
        try:
            request_id = raw_request.get('request_id', 'None') 
            request_data = self.prepare_request_data(raw_request)
            response_data = self.operator.process(request_data)
            return Response(json.dumps(response_data, ensure_ascii=False), mimetype='application/json')
        except Exception as e:
            # 如果停止事件失败的话，那我要返回
            return self.handle_error(e, request_id)

cancel_update = CancelUpdateConstructor(
    CancelUpdateWrapper().get_instance())
@api_blueprint.route(cancel_update.server_name, methods=['GET', 'POST'])
def get_cancel_update_result():
    return cancel_update.process_request(g.request_data)


class GetUpdateStatusConstructor(RequestProcessor):
    def __init__(self, operator):
        super().__init__('get-update-status', operator)


    def process_request(self, raw_request):
        try:
            request_id = raw_request.get('request_id', 'None')
            request_data = self.prepare_request_data(raw_request)
            response_data = self.operator.process(request_data)

            # operationType
            # 主要拿这个response data就行
            return Response(json.dumps(response_data, ensure_ascii=False), mimetype='application/json')

        except Exception as e:
            print('查询状态接口报错')
            return self.handle_error(e, request_id)

get_update_status = GetUpdateStatusConstructor(
    GetUpdateStatusWrapper().get_instance())
@api_blueprint.route(get_update_status.server_name, methods=['GET', 'POST'])
def get_get_update_status_result():
    return get_update_status.process_request(g.request_data)


class LocalFileUpdateConstructor(RequestProcessor):
    def __init__(self, operator):
        super().__init__('local-file-update', operator)
        env = os.environ['env']
        # client = Client()
        self.lindorm_url = config.get("lindorm_url")
        self.lindorm_key = config.get("lindorm_key")
        self.lindorm_value = config.get("lindorm_value")

        self.kb_info_client = KnowledgeBaseABDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            index="kbab_local_file_" + env
        )

    def handle_error(self, error: Exception, request_id: str) -> str:
        """处理过程中发生的错误"""
        error_msg = f'request_id: {request_id}. Failed to call server: {self.server_name}. error: {str(error)}'
        logger.error(f"{error_msg}\n{traceback.format_exc()}\n")
        response = {
            "code": 1,
            "message": error_msg,
            "data":{}
        }
        return response

    def process_request(self, raw_request):
        try:
            request_id = raw_request.get('request_id', 'None')
            request_data = self.prepare_request_data(raw_request)
            chinese_tokenizer_coarse, chinese_tokenizer_fine = get_chinese_tokenizers()
            response_data = self.operator.process(request_data,                                     
                                                  chinese_tokenizer_coarse = chinese_tokenizer_coarse,
                                                  chinese_tokenizer_fine = chinese_tokenizer_fine)
            return Response(json.dumps(response_data, ensure_ascii=False), mimetype='application/json')

        except Exception as e:
            
            return self.handle_error(e, request_id)

local_file_update = LocalFileUpdateConstructor(
    LocalFileUpdateWrapper().get_instance())
@api_blueprint.route(local_file_update.server_name, methods=['GET', 'POST'])
def get_local_file_update_result():
    return local_file_update.process_request(g.request_data)

class CancelParseConstructor(RequestProcessor):
    def __init__(self, operator):
        super().__init__('cancel-parse', operator)

    def process_request(self, raw_request):
        try:
            request_id = raw_request.get('request_id', 'None') 
            request_data = self.prepare_request_data(raw_request)
            response_data = self.operator.process(request_data)
            return Response(json.dumps(response_data, ensure_ascii=False), mimetype='application/json')
        except Exception as e:
            return self.handle_error(e, request_id)

cancel_parse = CancelParseConstructor(
    CancelParseWrapper().get_instance())
@api_blueprint.route(cancel_parse.server_name, methods=['GET', 'POST'])
def get_cancel_parse_result():
    return cancel_parse.process_request(g.request_data)



class GetHealthStatusConstructor(RequestProcessor):
    def __init__(self, operator):
        super().__init__('get-health-status', operator)


    def process_request(self, raw_request):
        try:
            request_id = raw_request.get('request_id', 'None')
            response_data = self.operator.process()
            # 主要拿这个response data就行
            return Response(json.dumps(response_data, ensure_ascii=False), mimetype='application/json')
        except Exception as e:
            print('查询状态接口报错')
            return self.handle_error(e, request_id)

get_health_status = GetHealthStatusConstructor(
    GetHealthStatusWrapper().get_instance())
@api_blueprint.route(get_health_status.server_name, methods=['GET', 'POST'])
def get_get_health_status_result():
    return get_health_status.process_request(g.request_data)



# 先写个简单版本，之后加轮训


class GraphUpdateConstructor(RequestProcessor):
    def __init__(self, operator):
        super().__init__('graph-update', operator)
        env = os.environ['env']
        self.lindorm_url = config.get("lindorm_url")
        self.lindorm_key = config.get("lindorm_key")
        self.lindorm_value = config.get("lindorm_value")
        self.graph_info_client = KnowledgeBaseABDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            index="kbab_" + env + "_graph"
        )


    def process_request(self, raw_request):
        try:
            request_id = raw_request.get('request_id', 'None') 
            request_data = self.prepare_request_data(raw_request)

            future = executor.submit(self.operator.process, 
                                    request_data)

            future.add_done_callback(lambda f: self._handle_result(f, request_id))

            response_data = {
                                "code": 0,
                                "message": "",
                                "data": {
                                    "documentCount": -1,
                                    "status": "GRAPH UPDATEING",
                                    "details": []
                                }
                            }
            return Response(json.dumps(response_data, ensure_ascii=False), mimetype='application/json')
        except Exception as e:
            return self.handle_error(e, request_id)

    def _handle_result(self, future, request_id):
        """异步任务执行完成后自动调用"""
        try:
            processed_data, iscancel = future.result()

            # 异步拿到结果的时候处理
            if iscancel:
                print(f"[{request_id}] 用户取消任务")
                # cancel_event.clear()
            else:
            # # cancel 之后后续任务无需处理

            #     # 打印或处理结果
                self.log_processing_result(processed_data)
            self.graph_info_client.write(str(999997), str(999997), kb_construct_type={'cancel': str(0)})
            self.graph_info_client.write(str(999996), str(999996), kb_construct_type={'finish': str(0)})

            # TODO: 这里可以存数据库、写文件、发回调通知等

        except Exception as e:
            self.graph_info_client.write(str(999997), str(999997), kb_construct_type={'cancel': str(0)})
            self.graph_info_client.write(str(999996), str(999996), kb_construct_type={'finish': str(0)})
            print(f"[{request_id}] graph 更新的异步任务失败 ❌: {e}")




graph_update = GraphUpdateConstructor(
    GraphUpdateWrapper().get_instance())
@api_blueprint.route(graph_update.server_name, methods=['GET', 'POST'])
def get_graph_update_result():
    return graph_update.process_request(g.request_data)



class FaqApiSyncConstructor(RequestProcessor):
    def __init__(self, operator):
        super().__init__('faq-api-sync', operator)

    def process_request(self, raw_request):
        try:
            request_id = raw_request.get('request_id', 'None') 
            request_data = self.prepare_request_data(raw_request)
            response_data = self.operator.process(request_data)

            return Response(json.dumps(response_data, ensure_ascii=False), mimetype='application/json')
        except Exception as e:

            return self.handle_error(e, request_id)


    def handle_error(self, error: Exception, request_id: str) -> str:
        """处理过程中发生的错误"""
        error_msg = f'request_id: {request_id}. Failed to call server: {self.server_name}. error: {str(error)}'
        logger.error(f"{error_msg}\n{traceback.format_exc()}\n")
        response = {"code": 1, "message": error_msg, "data": {}}
        return response

faq_api_sync = FaqApiSyncConstructor(
    FaqApiSyncWrapper().get_instance())
@api_blueprint.route(faq_api_sync.server_name, methods=['GET', 'POST'])
def get_faq_api_sync_result():
    return faq_api_sync.process_request(g.request_data)
