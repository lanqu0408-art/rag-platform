import json
import os
from typing import Optional, Union, Dict, List
from openai import AzureOpenAI
from core.config import config as config_global
from utils.util import str2json
import logging
logger = logging.getLogger(__name__)


class AzureOpenAIConfig:
    """Azure OpenAI API配置类"""
    def __init__(
        self,
        endpoint: str = "https://fintopia-none-cf.openai.azure.com/",
        model_name: str = "gpt-4o",
        deployment: str = "gpt-4o",
        api_version: str = "2024-12-01-preview"
    ):
        self.endpoint = endpoint  # API终结点
        self.model_name = model_name  # 模型名称
        self.deployment = deployment  # 部署名称
        self.api_version = api_version  # API版本

# 使用配置初始化客户端
config = AzureOpenAIConfig()
client = AzureOpenAI(
    api_version=config.api_version,
    azure_endpoint=config.endpoint,
    # api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
    api_key=config_global.get('ai-nlp-lab.ai.gpt_key', '')
)

def gpt_chat(
    user_content: Optional[str] = None,
    messages: Optional[List[Dict]] = None,
    system_content: str = "You are a helpful assistant.",
    to_json: bool = False,
    model_type: str = None,
    **kwargs
) -> Union[str, Dict]:
    """与Azure OpenAI模型对话
    
    参数:
        user_content: 用户消息内容
        messages: 对话历史消息列表
        system_content: 系统消息内容
        to_json: 是否将响应解析为JSON
        **kwargs: 其他API参数
        
    返回:
        模型响应字符串或解析后的JSON
        
    异常:
        ValueError: 如果没有提供user_content或messages
    """
    if not (user_content or messages):
        raise ValueError("Must provide either user_content or messages")        
    # 准备消息
    if messages:
        chat_messages = messages
    else:
        chat_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
    
    # 默认参数
    params = {
        "messages": chat_messages,
        "response_format": {"type": "json_object"},
        "temperature": 0,  # 温度参数控制随机性
        "max_tokens": kwargs.get("max_tokens", 16384),  # 最大token数
        "top_p": 0.1,  # 核心采样参数
        "frequency_penalty": 0.0,  # 频率惩罚
        "presence_penalty": 0.0,  # 存在惩罚
        "model": config.deployment,  # 使用的模型
    }
    
    # 更新额外参数
    params.update(kwargs)
    
    try:
        response = client.chat.completions.create(**params)
        print(f"OpenAI-API请求成功: {response}")

        content = response.choices[0].message.content
        
        if to_json:
            return str2json(content) or content
        return content
    except Exception as e:
        raise Exception(f"OpenAI-API请求失败: {str(e)}")