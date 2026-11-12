import json
import os
from typing import Optional, Union, Dict, List
from volcenginesdkarkruntime import Ark

from core.config import config as config_global
from utils.util import str2json


client = Ark(api_key=config_global.get('ai-nlp-lab.ai.ark_key', ''))


def ark_chat(
    user_content: Optional[str] = None,
    messages: Optional[List[Dict]] = None,
    system_content: str = "You are a helpful assistant.",
    to_json: bool = False,
    # model_type: str = "doubao-1-5-vision-pro-32k-250115",
    model_type: str = "doubao-1-5-thinking-vision-pro-250428",
    **kwargs
) -> Union[str, Dict]:
    
    """模型对话
    
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
    
    try:
        response = client.chat.completions.create(
            model=model_type,
            messages=chat_messages,
        )
        content = response.choices[0].message.content
        
        if to_json:
            return str2json(content) or content
        return content
    
    except Exception as e:
        raise Exception(f"Ark-API请求失败: {str(e)}")


