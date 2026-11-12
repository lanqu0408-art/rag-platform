# encoding=utf-8

"""
初始化日志配置
"""

import os
# 在导入其他库之前设置环境变量
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"  # 自家加的
import logging
from yqg_common.config_local import local_config
from yqg_common.log_config import yqg_init_log, YQG_LOG_CONFIG

logger = logging.getLogger(__name__)

local_config.initialize(cmd_flag=True, env_flag=False, local_yaml=False)

cmd_envs = local_config.get('e')
log_dir = './logs'
for cmd_env in cmd_envs:
    if 'log-dir' in cmd_env:
        log_dir = cmd_env.replace('log-dir=', '')
        break

# 初始化日志
# handler = 'yqg_common.log_handlers.YqgExtendMQueueTimedRotatingFileHandler'
# YQG_LOG_CONFIG['handlers']['app_info_file']['class'] = handler
# YQG_LOG_CONFIG['handlers']['app_error_file']['class'] = handler
# yqg_init_log(log_dir=log_dir, project_name="NLP-Lab")
# logger.info(f'init log by dir: {log_dir}')


"""
gunicorn 配置项
"""
# 并行工作线程数(按实际需要更改)
workers = 1
# 工作模式
worker_class = 'gthread'
threads = 10
# 监听内网端口8851【按需要更改】
# bind = '0.0.0.0:8851'
# 设置守护进程【关闭连接时，程序仍在运行】
daemon = False
# 设置超时时间，默认为30s。按自己的需求进行设置
timeout = 600
# 设置访问日志和错误信息日志路径
accesslog = './logs/gunicorn_access.log'
errorlog = './logs/gunicorn_error.log'
# 进程共享lock
preload_app = False
pidfile = "./logs/pid.log"
keepalive = 600
