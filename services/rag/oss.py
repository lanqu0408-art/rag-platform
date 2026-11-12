import oss2
from itertools import islice
import logging
import time
import random
# from client.Client import Client
from core.config import config
import alibabacloud_oss_v2 as oss
from datetime import timedelta

class Oss_Process:
    """处理飞书的组件"""
    def __init__(self, bucket_name='ai-rag-evaluation-dataset'):
        # client = Client()
        self.endpoint = config.get("oss_endpoint")
        self.access_key_id = config.get("access_key_id")
        self.access_key_secret = config.get("access_key_secret")
        # bucket_name = 'ai-rag-evaluation-dataset'
        self.region = 'cn-beijing'
        auth = oss2.AuthV4(self.access_key_id, self.access_key_secret)
        # 使用Auth实例初始化bucket。注意，使用V4签名时，region参数必须指定，否则会报错。
        self.bucket = oss2.Bucket(auth, self.endpoint, bucket_name, region=self.region)
        self.bucket_name=bucket_name
    def upload_file(self, object_name, data):
        try:
            result = self.bucket.put_object(object_name, data)
            logging.info(f"File uploaded successfully, status code: {result.status}")
        except oss2.exceptions.OssError as e:
            logging.error(f"Failed to upload file: {e}")

    def download_file(self, object_name):
        try:
            file_obj = self.bucket.get_object(object_name)
            content = file_obj.read().decode('utf-8')
            logging.info("File content:")
            logging.info(content)
            return content
        except oss2.exceptions.OssError as e:
            logging.error(f"Failed to download file: {e}")

    def list_objects(self):
        try:
            objects = list(islice(oss2.ObjectIterator(self.bucket), 100))
            for obj in objects:
                logging.info(obj.key)
        except oss2.exceptions.OssError as e:
            logging.error(f"Failed to list objects: {e}")

    def delete_objects_by_prefix(self, prefix):
        try:
            # 遍历带前缀的对象
            for obj in oss2.ObjectIterator(self.bucket, prefix=prefix):
                self.bucket.delete_object(obj.key)
                logging.info(f"Deleted object: {obj.key}")
        except oss2.exceptions.OssError as e:
            logging.error(f"Failed to delete objects: {e}")


    def cp_objects_by_prefix(self, src_prefix, dst_prefix):
        """
        把符合 src_prefix 的对象复制成新的 dst_prefix。
        例如：src_prefix="a/"，dst_prefix="b/"，
        a/file1.txt -> b/file1.txt
        """
        try:
            for obj in oss2.ObjectIterator(self.bucket, prefix=src_prefix):
                # 构造新的 key，把 src_prefix 替换成 dst_prefix
                new_key = obj.key.replace(src_prefix, dst_prefix, 1)
                # 调用 copy_object 复制
                self.bucket.copy_object(self.bucket.bucket_name, obj.key, new_key)
                logging.info(f"Copied {obj.key} -> {new_key}")
        except oss2.exceptions.OssError as e:
            logging.error(f"Failed to copy objects: {e}")

    def download_pdf_content(self, object_name):
        """
        从 OSS 下载 PDF 文件内容（不保存到本地）
        """
        try:
            file_obj = self.bucket.get_object(object_name)
            pdf_bytes = file_obj.read()
            logging.info(f"PDF content downloaded successfully: {object_name}")
            return pdf_bytes
        except oss2.exceptions.OssError as e:
            logging.error(f"Failed to download PDF content: {e}")
        except Exception as e:
            logging.error(f"Unexpected error while downloading PDF content: {e}")
            return None

    def get_url(self, object_name):
        """
          生成 DocMind 可解析的 V4 内网预签名 URL
        """
        provider=oss.credentials.StaticCredentialsProvider(
            access_key_id=self.access_key_id,
            access_key_secret=self.access_key_secret,
        )

        # 创建 V4 客户端
        cfg=oss.config.Config(
            region=self.region,
            credentials_provider=provider,
        )

        client=oss.Client(cfg)

        # 构造 V4 预签名 GET 请求
        presign=client.presign(
            oss.GetObjectRequest(
                bucket=self.bucket_name,
                key=object_name,
            ),
            expires=timedelta(hours=2)
        )

        # 把公网域名替换成内网域名（DocMind 必须走内网）
        presign.url=presign.url.replace(
            f".oss-{self.region}",
            f".oss-{self.region}-internal"
        )

        logging.info(f"[GenURL] 内部可访问 URL 生成成功：{presign.url}")
        return presign.url



