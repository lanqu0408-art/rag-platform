import json
import logging
import traceback
from flask import Response
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RequestProcessor:
    """请求处理器基类，封装通用的请求处理流程"""
    
    def __init__(self, server_name: str, operator: Any) -> None:
        """初始化请求处理器"""
        self.server_name = server_name
        self.operator = operator

    def log_processing_result(self, processed_data: Dict[str, Any]) -> None:
        """记录处理结果日志"""
        log_entry = {
            'server_name': self.server_name,
            'info': processed_data
        }
        logger.info(json.dumps(log_entry, ensure_ascii=False))

    def prepare_request_data(self, raw_request: Dict[str, Any]) -> Dict[str, Any]:
        """准备输入数据"""
        return raw_request

    def prepare_response_data(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """准备输出数据"""
        output_data = processed_data.get('outputs', None)
        if output_data is None:
            output_data = processed_data

        processed_data = {
            "code": 0,
            "message": "success",
            "data":output_data

        }
        return processed_data

    def handle_error(self, error: Exception, request_id: str) -> str:
        """处理过程中发生的错误"""
        error_msg = f'request_id: {request_id}. Failed to call server: {self.server_name}. error: {str(error)}'
        logger.error(f"{error_msg}\n{traceback.format_exc()}\n")
        response = {
            "code": 1,
            "detail": error_msg,
            "data":{}
        }
        return response

    def process_request(self, raw_request: Dict[str, Any]) -> str:
        """处理请求的主流程"""
        try:

            request_id = raw_request.get('request_id', 'None')
            
            # 准备请求数据
            request_data = self.prepare_request_data(raw_request)
            
            # 执行处理
            processed_data = self.operator.process(request_data)
            
            # 记录日志
            self.log_processing_result(processed_data)
            
            # 准备返回数据
            response_data = self.prepare_response_data(processed_data)

            return Response(json.dumps(response_data, ensure_ascii=False), mimetype='application/json')
        except Exception as e:
            return self.handle_error(e, request_id)

