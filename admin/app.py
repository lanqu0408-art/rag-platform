# -*- coding: UTF-8 -*-

"""
项目启动入口
"""

import re
import uuid
import time
import logging
import json
import traceback
from flask import Flask, g, request, Response
from werkzeug.utils import import_string
from yqg_common.log_config import trace
from core import influx_util

logger = logging.getLogger(__name__)


def now():
    return int(time.time() * 1000)


def set_request_data():
    """ 把request的相关信息设置到g里 """

    g.request_start_time = now()
    g.cur_path = re.sub('/+', '/', request.path + '/')

    if request.method == "POST" or request.method == 'PUT':
        if request.is_json:
            request_data = request.get_json(silent=True)
        else:
            request_data = request.form
    elif request.method == "GET":
        request_data = request.args
    else:
        request_data = request.values
    g.request_data = request_data if request_data else dict()
    trace_id = g.request_data.get('traceId')
    # 设置请求标识 request_id
    g.request_id = trace_id if trace_id else uuid.uuid1()
    trace.set(g.request_id)
    logger.info('START REQUEST PATH: %s, METHOD: %s, DATA: %s' % (
    g.cur_path, request.method, json.dumps(g.request_data, ensure_ascii=False)))

    influx_util.write_time(object_type='req_before', object_name=g.cur_path)


def log_request_time_and_response(response):
    """ 日志记录请求时间和返回结果 """
    request_used_time = now() - g.request_start_time

    influx_util.write_time_use(time_use=request_used_time, object_type='request', object_name=g.cur_path)
    logger.info('FINISH REQUEST: %s IN MILLISECOND: %s RESULT: %s' % (g.cur_path, request_used_time, response.json))

    return response


def handle_exception(e: Exception):
    """ 处理未定义 Exception  """
    logger.error('未定义异常： %s' % traceback.format_exc())
    return Response({'code': 1000, 'error': e})


def create_app(env):
    """程序工厂函数"""
    app = Flask('smart_llm')

    app.up_time = time.time()

    # 加载蓝图模块
    blueprints = [
        'admin.apis:api_blueprint',
        'admin.apis:health'
    ]
    for bp_name in blueprints:
        app.register_blueprint(import_string(bp_name))

    # 加载钩子函数  before_request：先加先用，after_request：先加后用
    app.before_request(set_request_data)
    app.after_request(log_request_time_and_response)

    app.register_error_handler(Exception, handle_exception)

    logger.info("application started on %s, found %s apis", env, len(list(app.url_map.iter_rules())) - 1)

    return app
