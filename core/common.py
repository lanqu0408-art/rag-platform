# encoding=utf-8

import time
import logging
import socket


logger = logging.getLogger(__name__)


def now():
    return int(time.time() * 1000)


def get_local_host_name():
    """ 获取本地机器名称 """
    return socket.gethostname()
