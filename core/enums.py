# encoding=utf-8


import logging
from enum import Enum, unique

logger = logging.getLogger(__name__)


@unique
class InfluxDBMeasurementEnum(Enum):
    """ measurement """
    TIME_USE = 'py_smart_llm_time_use'  # 耗时统计打点
    TIME = 'py_smart_llm_time'  # 触发打点
