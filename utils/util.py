# encoding=utf-8
import os
import sys
import json
import time
from functools import wraps
from typing import Optional, Union, Dict, List


def now(time_type=None):
    if time_type == 's':
        return int(time.time())
    else:
        return int(time.time() * 1000)



def ensure_directory_exists(file_path):
    """
    确保文件路径中的目录存在，如果不存在则创建
    """
    # 获取目录路径
    dir_path = os.path.dirname(file_path)
    
    # 检查是否有目录需要创建（非空路径且不是根目录）
    if dir_path and not os.path.exists(dir_path):
        try:
            # 创建目录（包括所有中间目录）
            os.makedirs(dir_path)
            print(f"创建目录: {dir_path}")
        except OSError as e:
            print(f"创建目录失败: {e}")
            sys.exit(1)
    return True


def singleton(cls):
    """单例装饰器"""
    instances = {}
    
    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance


def str2json(inputs: str) -> Optional[Dict]:
    """将LLM生成的字符串转换为JSON格式
    
    参数:
        inputs: 需要转换的字符串，可能包含markdown代码块
        
    返回:
        成功则返回解析后的JSON字典，失败返回None
        
    异常:
        ValueError: 如果输入不是有效的字符串
    """
    if not isinstance(inputs, str):
        raise ValueError("输入必须是字符串")
        
    try:
        # 清理常见格式问题
        text = inputs.replace("json", "").replace("```", "").strip()
        text = text.replace("None", "null").replace("True", "true").replace("False", "false")
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {str(e)}")
        return None
    except Exception as e:
        print(f"未知错误: {str(e)}")
        return None
