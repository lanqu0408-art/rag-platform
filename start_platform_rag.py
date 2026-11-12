# encoding=utf-8

import os
import logging

from yqg_common import config_center
from yqg_common.config_local import local_config
from core.config import config  # config初始化 # 循环饮用

# 兼容旧的架构使用redis和influxdb
local_config.initialize(local_yaml=False)
config_center.config = config

# from yqg_common.redis_client import yqg_redis_client
from yqg_common.influxdb_client import yqg_influxdb_cli

logger = logging.getLogger(__name__)

# 配置自定义redis
# yqg_redis_client.init_config_names(
#     redis_uri='smart_llm.datasource.redis.uri',
#     cluster_uri='smart_llm.datasource.redisCluster.uri',
#     prefix_key='smart_llm.datasource.redisCluster.prefixKey',
#     socket_timeout='smart_llm.datasource.redisCluster.socketTimeout',
#     socket_connect_timeout='smart_llm.datasource.redisCluster.connectTimeout',
#     refresh_interval='smart_llm.datasource.redisCluster.refreshInterval'
# )

# 配置自定义influx
if 'prod' in os.environ['env']:
    yqg_influxdb_cli.init_config_names(
        url='smart.datasource.influxdb.url',
        user='smart.datasource.influxdb.user',
        password='smart.datasource.influxdb.password',
        dbname='smart.datasource.influxdb.dbname'
    )
else:
    yqg_influxdb_cli.init_config_names(
        url='ai-nlp-lab.datasource.influxdb.url',
        user='ai-nlp-lab.datasource.influxdb.user',
        password='ai-nlp-lab.datasource.influxdb.password',
        dbname='ai-nlp-lab.datasource.influxdb.dbname'
    )
yqg_influxdb_cli.refresh_interval = 60 * 60  # 指定连接一小时更新一次

# 创建应用
from admin.app import create_app
app = create_app(os.environ['env'])
