import json
import requests
from typing import Dict, Any, Optional, List
from requests.exceptions import ReadTimeout, RequestException


class BaseAPI:
    """API调用的基类，提供统一的请求处理和错误管理"""
    def __init__(self, config):
        """初始化API客户端"""
        self.config = config
        self.name = self.config['NAME']
        self.base_url = f"http://{self.config['HOST']}:{self.config['PORT']}"
        self.sub_url = self.config['SUB_URL']
        self.timeout = self.config['TIMEOUT']

    def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """发送API请求并返回响应数据
        
        Args:
            endpoint: API端点路径
            payload: 请求体数据
            
        Returns:
            解析后的响应数据
            
        Raises:
            RequestException: 网络请求错误
            ValueError: 响应数据格式错误或API返回失败状态
        """
        try:
            response = requests.post(
                f"{self.base_url}/{endpoint}",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            return data    
        except ReadTimeout:
            raise RequestException(f"{self.name} Request timed out after {self.timeout} seconds")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON response from {self.name} API")
        except Exception as e:
            raise RequestException(f"{self.name} API request failed: {str(e)}")


class ExampleAPI(BaseAPI):
    """特定API的实现类"""
    def get_something(self, info_carrier: Dict[str, Any]) -> List[Dict[str, str]]:
        """使用样例"""
        message_id = info_carrier.get("message_id", 'default')
        payload = {"config_id": message_id}
        data = self._make_request(
            endpoint=self.sub_url['SUB_URL'],
            payload=payload
        )
        return data