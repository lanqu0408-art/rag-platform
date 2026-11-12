# encoding=utf-8


import logging
from yqg_common.influxdb_batch_client import batch_influx_client
from core.config import config
from core.utils import now
from core.enums import InfluxDBMeasurementEnum

logger = logging.getLogger(__name__)


def retention_policy():
    return config.get("smart_llm.influxDB.retentionPolicy", None)


def write_time_use(time_use, object_name, object_type, **kwargs):
    """
    通用耗时打点
    :param time_use: 耗时
    :param object_name: 名称（请求路径、方法名称、代码段标识）
    :param object_type: 类型（请求、方法、代码段）
    :return:
    """
    json_dict = {
        "measurement": InfluxDBMeasurementEnum.TIME_USE.value,
        "retention_policy": retention_policy(),
        "tags": {},
        "fields": {}
    }
    json_dict['fields']['time_use'] = time_use
    json_dict['tags']['object_name'] = object_name
    json_dict['tags']['object_type'] = object_type
    if kwargs:
        for k, v in kwargs.items():
            json_dict['tags'][k] = v

    batch_influx_client.write(**json_dict)


def write_time(object_name, object_type, flag=0, **kwargs):
    """
    通用耗时打点
    :param object_name: 名称（请求路径、方法名称、代码段标识）
    :param object_type: 类型（请求、方法、代码段）
    :param flag: 自定义标识
    :return:
    """
    json_dict = {
        "measurement": InfluxDBMeasurementEnum.TIME.value,
        "retention_policy": retention_policy(),
        "tags": {},
        "fields": {}
    }
    json_dict['fields']['flag'] = flag
    json_dict['tags']['object_name'] = object_name
    json_dict['tags']['object_type'] = object_type
    if kwargs:
        for k, v in kwargs.items():
            json_dict['tags'][k] = v

    batch_influx_client.write(**json_dict)


class InfluxExample(object):

    @classmethod
    def fun_influx_time_use(cls, fun, object_type, object_name=None, args=(), **kwargs):
        if not object_name:
            object_name = fun.__name__
        start_time = now()
        try:
            result = fun(*args, **kwargs)
            flag = 1
        except Exception as e:
            logger.exception(f"方法 {object_name} 调用失败! MSG: {e}")
            flag = 0
            result = None
        time_use = now() - start_time
        write_time_use(time_use=time_use, object_name=object_name, object_type=object_type, flag=flag)
        logger.info('method: %s, use time: %s' % (object_name, time_use))

        return result
