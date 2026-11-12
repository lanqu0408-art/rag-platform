# encoding=utf-8
"""
新的 config sdk 包装兼容之前的
"""

import os
from client.Client import Client

os.environ['__CLUSTER_ENV'] = os.environ['env']


class Config:
    def __init__(self):
        self.config_client = Client()

    def get(self, key, default=None):
        try:
            value = self[key]
            return value
        except KeyError:
            return default

    def __getitem__(self, item):
        # value = self.config_client.get_value(item)
        configs = self.config_client.get_configs()
        value = configs[item]
        if value in ['true', 'True', 'TRUE', 'T']:
            return True
        elif value in ['false', 'False', 'FALSE', 'F']:
            return False
        return value

    @property
    def client(self):
        return self.config_client


config = Config()
 