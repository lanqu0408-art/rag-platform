import os
import json
import concurrent.futures
from typing import Optional, Union, Dict, List
from openai import OpenAI
import logging
# from core.config import config as config_global
from utils.util import str2json

logger = logging.getLogger(__name__)


client = OpenAI(
    api_key="EMPTY",
    # base_url="https://vllm.yangqianguan.com/172-21-125-220/8008/v1",
    base_url="https://xinference.fintopia.tech/v1/",
)
# qwen2.5-instruct
# Qwen3-Embedding-0.6B

# def aizt_chat_stream(
        
# )



def aizt_embedding(
    input_text: Union[str, List[str]],
    model_type: str = "qwen-30.6b-embedding",
    lora_name: str = None,
    to_json: bool = False,
    **kwargs
) -> Union[List[float], Dict]:
    """调用 Qwen embedding 模型

    参数:
        input_text: 输入文本或文本列表
        model_type: 模型名称 (默认为 qwen-30.6b-embedding)
        lora_name: LoRA 名称（如需覆盖模型）
        to_json: 是否返回完整 JSON
        **kwargs: 其他 API 参数

    返回:
        向量 embedding (list[float]) 或完整 JSON
    """
    try:
        # 如果指定了 lora_name，则覆盖模型
        if lora_name is not None:
            model_type = lora_name

        response = client.embeddings.create(
            model=model_type,
            input=input_text,
            **kwargs
        )

        if to_json:
            return response.dict()

        # 如果输入是单条文本
        if isinstance(input_text, str):
            return response.data[0].embedding

        # 如果输入是多条文本
        return [item.embedding for item in response.data]

    except Exception as e:
        raise Exception(f"QwenAI-Embedding API 请求失败: {str(e)}")


def aizt_chat(
    user_content: Optional[str] = None,
    messages: Optional[List[Dict]] = None,
    system_content: str = "You are a helpful assistant.",
    to_json: bool = False,
    model_type: str = "qwen2.5-instruct",
    lora_name: str = None,
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
   
    try:
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
        
        extra_body = {
            "chat_template_kwargs": {
                "enable_thinking": False
            },
        }

        if lora_name is not None:
            extra_body['model'] = lora_name

        response = client.chat.completions.create(
            model=model_type,
            messages=chat_messages,
            temperature=0,
            response_format={"type": "json_object"},
            stream=False, 
            extra_body=extra_body
        )
        content = response.choices[0].message.content
        print(content)


        # stream = client.chat.completions.create(
        #     model=model_type,
        #     messages=chat_messages,
        #     temperature=0,
        #     response_format={"type": "json_object"},
        #     stream=True,
        #     extra_body=extra_body
        # )

        # full_content = ""

        # for event in stream:
        #     if event.choices:
        #         delta = event.choices[0].delta.content  # 直接属性访问
        #         if delta:
        #             print(delta, end="", flush=True)  # 实时打印
        #             full_content += delta

        # print("\n---\nFinal collected response:")
        # print(full_content)


        # full_content = ""

        # # 遍历流式响应
        # for event in stream:
        #     # event 里每次只有增量
        #     if event.choices:
        #         delta = event.choices[0].delta.get("content", "")
        #         if delta:
        #             print(delta, end="", flush=True)  # 实时打印
        #             full_content += delta
                
        if to_json:
            return str2json(content) or content
        return content
            


    except Exception as e:
        raise Exception(f"QwenAI-API请求失败: {str(e)}")


def batch_call_aizt_chat(
    prompts: Optional[List[str]] = None,
    messages_list: Optional[List[List[Dict]]] = None,
    system_content: str = "You are a helpful assistant.",
    to_json: bool = False,
    model_type: str = "Qwen/Qwen3-8B",
    lora_name: str = None,
    max_workers: int = 5,
    **kwargs
) -> List[Dict]:
    """批量调用 aizt_chat，支持传入 prompts 或 messages_list

    参数:
        prompts: 用户消息内容列表（与 messages_list 二选一）
        messages_list: 对话历史消息列表（与 prompts 二选一）
        system_content: 系统消息内容
        to_json: 是否将响应解析为JSON
        model_type: 模型类型
        max_workers: 线程池最大工作线程数
        **kwargs: 其他传递给 aizt_chat 的参数

    返回:
        模型响应列表（每个元素为字符串或解析后的JSON）

    异常:
        部分失败时返回部分结果，并记录错误日志
    """
    if (prompts is None and messages_list is None) \
        or (prompts is not None and messages_list is not None):
        raise ValueError("必须提供 prompts 或 messages_list 中的一个，且不能同时提供")

    results = []
    failed_indices = []

    def _call_aizt_chat(input_data: Union[str, List[Dict]], index: int):
        try:
            if isinstance(input_data, str):
                # 使用 prompts
                result = aizt_chat(
                    user_content=input_data,
                    system_content=system_content,
                    to_json=to_json,
                    model_type=model_type,
                    lora_name=lora_name,
                    **kwargs
                )
            else:
                # 使用 messages_list
                result = aizt_chat(
                    messages=input_data,
                    system_content=system_content,
                    to_json=to_json,
                    model_type=model_type,
                    lora_name=lora_name,
                    **kwargs
                )
            return result
        except Exception as e:
            logger.error(f"批量调用 aizt_chat 失败 (输入索引 {index}): {str(e)}")
            failed_indices.append(index)
            return None

    input_data_list = prompts if prompts is not None else messages_list

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_call_aizt_chat, input_data, index): index
            for index, input_data in enumerate(input_data_list)
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.error(f"批量调用 aizt_chat 异常 (输入索引 {index}): {str(e)}")
                failed_indices.append(index)

    if failed_indices:
        logger.warning(f"批量调用 aizt_chat 部分失败，失败索引: {failed_indices}")

    return results


aizt_chat("你好，请跟我对话，对话长度至少200字")
import pdb
pdb.set_trace()

aizt_embedding("你好，请跟我对话，对话长度至少200字")