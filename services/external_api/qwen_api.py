import json
import os
from typing import Optional, Union, Dict, List
from openai import OpenAI
import logging
from core.config import config as config_global
from utils.util import str2json

logger = logging.getLogger(__name__)

# qwen_key = config_global.get('ai-nlp-lab.ai.qwen_key', '')
# logger.info(f'QwenAI-API密钥: {qwen_key}')

client = OpenAI(
    # api_key=os.environ.get("QWEN_API_KEY", ""),  # 从环境变量获取API密钥,
    api_key=config_global.get('ai-nlp-lab.ai.qwen_key', ''),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # base_url="https://api-inference.modelscope.cn/v1/",
    default_headers={'X-DashScope-DataInspection': '{"input":"disable", "output":"disable"}'},
)


def qwen_chat(
    user_content: Optional[str] = None,
    messages: Optional[List[Dict]] = None,
    system_content: str = "You are a helpful assistant.",
    to_json: bool = False,
    model_type: str = "Qwen/Qwen3-8B",
    **kwargs
) -> Union[str, Dict]:
    """与Qwen模型对话
    
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
            temperature=0,
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": False}
        )
        print(f"QwenAI-API请求成功: \n{response}\n")
        content = response.choices[0].message.content
        
        if to_json:
            return str2json(content) or content
        return content
    except Exception as e:
        raise Exception(f"QwenAI-API请求失败: {str(e)}")