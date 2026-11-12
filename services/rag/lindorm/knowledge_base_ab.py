from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from opensearchpy import OpenSearch
from haystack import Document

class KnowledgeBaseABDocumentStore:

    def __init__(
            self,
            hosts: List[Dict[str, Any]],
            http_auth: Optional[Tuple[str, str]] = None,
            *,
            index: str,
            field_kb_id: str = "kb_id",
            field_kb_name: str = "kb_name",
            field_kb_construct_type: str = "kb_construct_type",
            field_kb_doc_info: str = "kb_doc_info",
            field_update_res: str = "update_res",
            num_shards: int = 2,
            create_index: bool = True,
            use_ssl: bool = False,
            http_compress: bool = False,
            timeout: int = 30,
            pool_maxsize: int = 128,
    ) -> None:
        self.index = index
        self.num_shards = num_shards
        self.field_kb_id = field_kb_id
        self.field_kb_name = field_kb_name
        self.field_kb_construct_type = field_kb_construct_type
        self.field_kb_doc_info = field_kb_doc_info
        self.field_update_res = field_update_res
        self.client = OpenSearch(
            hosts=hosts,
            http_auth=http_auth,
            http_compress=http_compress,
            use_ssl=use_ssl,
            pool_maxsize=pool_maxsize,
            timeout=timeout,
        )

        if create_index:
            self._create_index_if_needed()

    # --------------------------- index management --------------------------- #
    def _create_index_if_needed(self) -> None:
        if self.client.indices.exists(index=self.index):
            return

        index_body: Dict[str, Any] = {
            "settings": {
                "index": {
                    "number_of_shards": self.num_shards,
                }
            },
            "mappings": {
                "properties": {
                    self.field_kb_id: {
                        "type": "keyword",
                    },
                    self.field_kb_name: {
                        "type": "keyword",
                    },
                    self.field_kb_construct_type: {
                        "type": "object"
                    },  # 新增
                    self.field_kb_doc_info: {
                        "type": "object"
                    },  # 新增

                },
            },
        }

        self.client.indices.create(index=self.index, body=index_body)

    @staticmethod
    def _json_dumps(obj: Any) -> str:
        import json

        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    # ----------------------------- write / delete --------------------------- #
    def write(self, kb_id: str, kb_name: str, kb_construct_type: Optional[Dict[str, Any]] = None, kb_doc_info: Optional[List[Dict[str, Any]]] = None, update_res: Optional[Dict[str, Any]] = None) -> int:
        ops: List[str] = []
        body = {
            self.field_kb_id: kb_id,
            self.field_kb_name: kb_name,
        }

        if kb_construct_type:
            body[self.field_kb_construct_type] = kb_construct_type  # 存入 dict
        if kb_doc_info:
            body[self.field_kb_doc_info] = kb_doc_info  # 存入 dict
        if update_res:
            body[self.field_update_res] = update_res  # 存入 dict



        # Use doc.id if provided; otherwise let Lindorm generate one
        # if doc.id:
        
        index_meta = {"index": {"_index": self.index}}
        index_meta["index"]["_id"] = str(kb_id)
        ops.append(self._json_dumps(index_meta))
        ops.append("\n")
        ops.append(self._json_dumps(body))
        ops.append("\n")
        if ops:
            self.client.bulk(body="".join(ops), refresh=True)
        return 1

    def write_or_append(
        self,
        kb_id: str,
        kb_name: str,
        kb_construct_type: Optional[Dict[str, Any]] = None,
        kb_doc_info: Optional[List[Dict[str, Any]]] = None,
        update_res: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        如果文档不存在则新建，否则在原有文档基础上合并更新。
        支持列表追加（去重）与字典字段合并。
        """

        # 尝试读取已存在文档
        try:
            existing = self.client.get(index=self.index, id=kb_id, ignore=[404])
            source = existing.get('_source', {}) if existing and '_source' in existing else {}
            exists = bool(source)
        except Exception as e:
            print(f"[write_or_append] Failed to get existing doc for kb_id={kb_id}: {e}")
            source, exists = {}, False

        # 初始化 body
        body = {
            self.field_kb_id: kb_id,
            self.field_kb_name: kb_name,
        }

        # 合并 kb_construct_type
        if kb_construct_type:
            old_type = source.get(self.field_kb_construct_type, {})
            body[self.field_kb_construct_type] = {**old_type, **kb_construct_type}
        elif exists:
            body[self.field_kb_construct_type] = source.get(self.field_kb_construct_type, {})

        # 合并 kb_doc_info（追加 + 去重）
        old_doc_info = source.get(self.field_kb_doc_info, [])
        if kb_doc_info:
            existing_file_ids = {d.get("file_id") for d in old_doc_info if "file_id" in d}
            new_docs = [
                d for d in kb_doc_info if d.get("file_id") not in existing_file_ids
            ]
            merged_doc_info = old_doc_info + new_docs
            body[self.field_kb_doc_info] = merged_doc_info
        elif exists:
            body[self.field_kb_doc_info] = old_doc_info

        # 合并 update_res
        if update_res:
            old_update = source.get(self.field_update_res, {})
            body[self.field_update_res] = {**old_update, **update_res}
        elif exists:
            body[self.field_update_res] = source.get(self.field_update_res, {})

        # 构造 bulk 请求
        index_meta = {"index": {"_index": self.index, "_id": str(kb_id)}}
        ops = [
            self._json_dumps(index_meta), "\n",
            self._json_dumps(body), "\n"
        ]

        # 执行写入
        try:
            self.client.bulk(body="".join(ops), refresh=True)
            if exists:
                print(f"[write_or_append] Updated existing kb_id={kb_id} ✅")
            else:
                print(f"[write_or_append] Created new kb_id={kb_id} ✅")
            return 1
        except Exception as e:
            print(f"[write_or_append] Failed to write_or_append kb_id={kb_id}: {e}")
            return 0




    def delete_by_id(self, kb_id: str) -> bool:
        """
        按 kb_id 删除文档（你的写入逻辑里 _id == kb_id）。
        True=删除成功，False=不存在或失败
        """
        try:
            self.client.delete(index=self.index, id=str(kb_id), refresh=True)
            return True
        except Exception as e:
            print(f"[delete_by_kb_id] Failed kb_id={kb_id}: {e}")
            return False


    def count(self) -> int:
        return int(self.client.count(index=self.index)["count"])

    # ------------------------------- filtering ------------------------------ #
    def get(self, kb_id: str) -> str:
        q = {
            "query": {
                "term": {
                    "kb_id": kb_id,
                }
            }
        }
        res = self.client.search(index=self.index, body=q)
        hit = res.get("hits", {}).get("hits", [])
        if len(hit) == 0:
            return -1
        return hit[0]["_source"]["kb_name"]


    def get_update_res(self, kb_id: str) -> str:
        q = {
            "query": {
                "term": {
                    "kb_id": kb_id,
                }
            }
        }
        res = self.client.search(index=self.index, body=q)
        hit = res.get("hits", {}).get("hits", [])
        if len(hit) == 0:
            return -1
        return hit[0]["_source"].get("update_res", {})


    def get_construct_type(self, kb_id: str) -> str:
        q = {
            "query": {
                "term": {
                    "kb_id": kb_id,
                }
            }
        }
        res = self.client.search(index=self.index, body=q)
        hit = res.get("hits", {}).get("hits", [])
        if len(hit) == 0:
            return -1
        return hit[0]["_source"].get("kb_construct_type", {})
    
    def get_doc_info(self, kb_id: str) -> str:
        q = {
            "query": {
                "term": {
                    "kb_id": kb_id,
                }
            }
        }
        res = self.client.search(index=self.index, body=q)
        hit = res.get("hits", {}).get("hits", [])
        if len(hit) == 0:
            return -1
        doc_list =hit[0]["_source"].get("kb_doc_info", [])
        # 这里要打断一下
        # 这里是一个存量问题的处理，之前可能是一个字典
        # 现在优化成了一个列表
        if isinstance(doc_list, list):
            kb_doc_info = {
                item["id"]: {"version": item["version"], "path": item["path"]}
                for item in doc_list}
        else:
            kb_doc_info = doc_list
        return kb_doc_info

    def get_local_file_status(self, kb_id: str, file_id: str) -> str:
        q = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"kb_id": file_id}}
                    ]
                }
            }
        }
        res = self.client.search(index=self.index, body=q)
        hit = res.get("hits", {}).get("hits", [])
        if len(hit) == 0:
            return -1, -1

        kb_construct_type = hit[0]["_source"]['kb_construct_type']
        update_status = kb_construct_type.get("update_status", -1)
        cancel_reason = kb_construct_type.get("cancel_reason", -1)

        return update_status, cancel_reason
        # return hit[0]["_source"]['kb_construct_type'].get('update_status', -1), hit[0]["_source"]['kb_construct_type'].get('cancel_reason', -1)
    
