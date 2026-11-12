# encoding=utf-8

import time
import logging
import requests
import traceback
from json.decoder import JSONDecodeError
from requests.exceptions import Timeout, ConnectionError
from tenacity import (retry, wait_exponential, stop_after_attempt, retry_if_exception_type, after_log,RetryCallState)


logger = logging.getLogger(__name__)


def now():
    return int(time.time() * 1000)

def return_last_value(retry_state: RetryCallState):
    if 'url' in retry_state.kwargs:
        url = retry_state.kwargs.get('url')
    else:
        url = retry_state.args[0]

    exception = retry_state.outcome.exception()
    logger.error('url: %s request failed for error: %s' % (url, exception))

    return None

@retry(
    retry=retry_if_exception_type((Timeout, ConnectionError, JSONDecodeError, ValueError)),
    wait=wait_exponential(multiplier=1, max=4),  # 重试时间间隔满足：2^n * multiplier,
    stop=stop_after_attempt(3),  # 重试3次
    retry_error_callback=return_last_value,
    reraise=True,
    after=after_log(logger, logging.INFO)
)
def request_api(url, method, data, res_type='json'):
    headers = {"Content-Type": "application/json",'Connection':'close'}
    logger.info('url: %s start request!' % url)
    if method == 'get':
        response = requests.get(url, headers=headers, timeout=5)
    else:
        response = requests.post(url, data=data, headers=headers, timeout=5)

    if response.status_code == 200:
        logger.info('[smart_audit] url: %s request success!' % url)
        if res_type == 'json':
            return response.json()
        else:
            return response.text
    else:
        raise ConnectionError(f"request_api --- {response.status_code} --- {response.text}")


