from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from opensearchpy import OpenSearch
from haystack.document_stores.types import DuplicatePolicy
from haystack import Document, component
from copy import deepcopy

class LindormDocumentStore:
    """A minimal DocumentStore for Alibaba Cloud Lindorm Vector Engine.

    This store uses the OpenSearch/Elasticsearch-compatible HTTP API provided by
    Lindorm's *Search Engine* to index, delete, and retrieve documents, and its
    *Vector Engine* capabilities to perform KNN vector search.

    Parameters
    ----------
    hosts : List[Dict[str, Any]]
        opensearch-py hosts, e.g. [{"host": "<ld-xxx.lindorm.aliyuncs.com>", "port": 30070}].
    http_auth : Tuple[str, str] | None
        (username, password) if authentication is enabled.
    index : str
        Index name to use.
    embedding_dim : int
        Dimension of your dense embeddings.
    vector_field : str
        Field name that stores the vector/embedding. Default: "embedding".
    content_field : str
        Field name that stores the document text. Default: "content".
    meta_field : str
        Field name that stores document metadata dict. Default: "meta".
    algorithm : str
        Vector index algorithm: "hnsw" | "ivfpq" | "flat". Default: "hnsw".
    space_type : str
        Distance metric: "l2" | "cosinesimil" | "innerproduct". Default: "cosinesimil".
    num_shards : int
        Number of shards. Default: 2.
    create_index : bool
        If True, create index with vector mappings if it doesn't exist.
    use_ssl, http_compress, timeout, pool_maxsize :
        Passed to OpenSearch client.
    """

    def __init__(
            self,
            hosts: List[Dict[str, Any]],
            http_auth: Optional[Tuple[str, str]] = None,
            *,
            index: str = "temp",
            embedding_dim: int = 1024,
            vector_field: str = "embedding",
            content_field: str = "content",
            kb_id_field: str = "kb_id",
            meta_field: str = "meta",
            algorithm: str = "hnsw",
            space_type: str = "cosinesimil",
            num_shards: int = 2,
            create_index: bool = True,
            use_ssl: bool = False,
            http_compress: bool = False,
            timeout: int = 30,
            pool_maxsize: int = 128,
    ) -> None:
        self.index = index
        self.embedding_dim = int(embedding_dim)
        self.vector_field = vector_field
        self.content_field = content_field
        self.kb_id_field = kb_id_field
        self.meta_field = meta_field
        self.algorithm = algorithm
        self.space_type = space_type
        self.num_shards = num_shards

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



    def remove_source_doc_and_cleanup(
            self,
            *,
            kb_id: Union[str, int],
            doc_url: str,
            source_doc_field: str = "source_doc",
            mark_field: str = "__delete_me",
            refresh: bool = True,
            conflicts: str = "proceed",
            wait_for_completion: bool = True,
            request_timeout: Optional[int] = None,
        ) -> Dict[str, int]:
            """
            稳健版：
            - 只用 kb_id 过滤（避免 source_doc term 匹配不到）
            - 脚本内部判断是否包含 doc_url；包含才删，不包含则 noop
            - 删完为空数组则打标，之后按标记 delete_by_query
            """

            # kb_id 既可能是 numeric 也可能是 keyword，这里用 terms 同时喂两种
            kb_terms = [kb_id]
            try:
                kb_terms.append(int(kb_id))  # 如果 kb_id 是 "260129" 这种字符串
            except Exception:
                pass
            kb_terms.append(str(kb_id))

            update_body: Dict[str, Any] = {
                "query": {
                    "bool": {
                        "filter": [
                            {"terms": {self.kb_id_field: list(dict.fromkeys(kb_terms))}}
                        ]
                    }
                },
                "script": {
                    "lang": "painless",
                    "source": """
                        def f = params.f;
                        def url = params.url;
                        def mark = params.mark;

                        if (!ctx._source.containsKey(f) || ctx._source[f] == null) {
                            ctx.op = 'noop';
                            return;
                        }

                        // source_doc 期望是数组
                        def arr = ctx._source[f];

                        // 如果不包含目标 url，则不更新
                        if (!arr.contains(url)) {
                            ctx.op = 'noop';
                            return;
                        }

                        // 删除所有等于 url 的元素（防止重复）
                        arr.removeIf(x -> x == url);

                        // 删除后为空则打标，否则去掉标记
                        if (arr.size() == 0) {
                            ctx._source[mark] = true;
                        } else {
                            ctx._source.remove(mark);
                        }
                    """,
                    "params": {"f": source_doc_field, "url": doc_url, "mark": mark_field},
                },
            }

            update_resp = self.client.update_by_query(
                index=self.index,
                body=update_body,
                refresh=refresh,
                conflicts=conflicts,
                wait_for_completion=wait_for_completion,
                request_timeout=request_timeout,
            )

            updated = int(update_resp.get("updated", 0))

            delete_body: Dict[str, Any] = {
                "query": {
                    "bool": {
                        "filter": [
                            {"terms": {self.kb_id_field: list(dict.fromkeys(kb_terms))}},
                            {"term": {mark_field: True}},
                        ]
                    }
                }
            }

            delete_resp = self.client.delete_by_query(
                index=self.index,
                body=delete_body,
                refresh=refresh,
                conflicts=conflicts,
                wait_for_completion=wait_for_completion,
                request_timeout=request_timeout,
            )

            deleted = int(delete_resp.get("deleted", 0))
            return {"updated": updated, "deleted": deleted}





    # --------------------------- index management --------------------------- #
    def _create_index_if_needed(self) -> None:
        if self.client.indices.exists(index=self.index):
            return

        if self.algorithm not in {"hnsw", "ivfpq", "flat"}:
            raise ValueError("algorithm must be one of: 'hnsw', 'ivfpq', 'flat'")

        index_body: Dict[str, Any] = {
            "settings": {
                "index": {
                    "number_of_shards": self.num_shards,
                    "knn": True,
                }
            },
            "mappings": {
                "_source": {"excludes": [self.vector_field]},  # optional: don't return embeddings by default
                "properties": {
                    self.vector_field: {
                        "type": "knn_vector",
                        "dimension": self.embedding_dim,
                        # Lindorm requires specifying the index method
                        "method": {
                            "engine": "lvector",
                            "name": self.algorithm,
                            "space_type": self.space_type,
                            # Reasonable defaults for HNSW; IVFPQ requires different params (see docs)
                            "parameters": {"m": 24, "ef_construction": 200}
                            if self.algorithm == "hnsw"
                            else {},
                        },
                    },
                    self.content_field: {"type": "text", "analyzer": "ik_max_word"},
                    self.meta_field: {"type": "object", "enabled": True},
                    self.kb_id_field: {"type": "keyword"}
                },
            },
        }

        # IVFPQ-specific mandatory flags (see Lindorm docs)
        if self.algorithm == "ivfpq":
            index_body["settings"]["index"]["knn.offline.construction"] = True
            # IVFPQ requires parameters.m == dimension; supply a basic default
            index_body["mappings"]["properties"][self.vector_field]["method"][
                "parameters"
            ] = {"m": self.embedding_dim, "nlist": 10000}

        self.client.indices.create(index=self.index, body=index_body)


    def set_index(self, index: str) -> None:
        """Dynamically set or switch the index for the document store."""
        self.index = index
        self._create_index_if_needed()  # Recreate index if necessary

    # ----------------------------- write / delete --------------------------- #
    def write_documents(self, documents: Iterable[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> int:
        """Bulk-index Documents.

        Each Document is serialized as:
            {
                content_field: doc.content,
                vector_field: doc.embedding,
                meta_field: doc.meta,
            }
        """
        ops: List[str] = []
        count = 0
        for doc in documents:
            body = {
                self.content_field: doc.content,
                self.meta_field: doc.meta or {},
                self.kb_id_field: doc.kb_id,
            }
            if doc.embedding is not None:
                body[self.vector_field] = doc.embedding
            # Use doc.id if provided; otherwise let Lindorm generate one
            index_meta = {"index": {"_index": self.index}}
            if doc.id:
                index_meta["index"]["_id"] = str(doc.id)
            ops.append(self._json_dumps(index_meta))
            ops.append("\n")
            ops.append(self._json_dumps(body))
            ops.append("\n")
            count += 1

        if ops:
            self.client.bulk(body="".join(ops), refresh=True)
        return count

    def delete_documents(
            self,
            document_ids: Optional[List[str]] = None,
            *,
            delete_all: bool = False,
            filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        if delete_all:
            self.client.indices.delete(index=self.index, ignore=[404])
            self._create_index_if_needed()
            return 0

        if document_ids:
            ops: List[str] = []
            for _id in document_ids:
                ops.append(self._json_dumps({"delete": {"_index": self.index, "_id": str(_id)}}))
                ops.append("\n")
            if ops:
                self.client.bulk("".join(ops),refresh=True)
            return len(document_ids)

        if filters:
            query = self._build_filter_query(filters)
            self.client.delete_by_query(index=self.index, body={"query": query})
            # The response contains deleted count, but keep return type simple
            return 0

        return 0

    def count_documents(self) -> int:
        return int(self.client.count(index=self.index)["count"])

    # ------------------------------- filtering ------------------------------ #
    # def filter_documents(self, filters: Optional[Dict[str, Any]] = None, size: int = 10) -> List[Document]:
    #     if not filters:
    #         q = {"query": {"match_all": {}}, "size": size}
    #     else:
    #         q = {"query": self._build_filter_query(filters), "size": size}
    #     res = self.client.search(index=self.index, body=q)
    #     return [self._to_document(hit) for hit in res.get("hits", {}).get("hits", [])]

    def _build_filter_query(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Very simple term/range filter translation.
        Supports equality and list membership. To use range queries, pass a dict
        like: {"meta.price": {"gte": 10, "lte": 20}}.
        """
        must: List[Dict[str, Any]] = []
        for field, value in (filters or {}).items():
            if isinstance(value, dict):
                must.append({"range": {field: value}})
            elif isinstance(value, list):
                must.append({"terms": {field: value}})
            else:
                must.append({"term": {field: value}})
        return {"bool": {"must": must}} if must else {"match_all": {}}
    


    def filter_graph_info(
        self,
        filters: Optional[Dict[str, Any]] = None,
        size: int = 10,
        all: bool = True,
        scroll: str = "2m"
    ) -> List[Document]:
        """
        过滤并返回文档。
        - filters: 过滤条件
        - size: 单批大小
        - all: 是否获取所有文档（使用 scroll）
        - scroll: scroll 游标有效期
        """
        if not filters:
            q = {"query": {"match_all": {}}, "size": size}
        else:
            q = {"query": self._build_filter_query(filters), "size": size}

        # if not all:
        #     # 原始逻辑，只取一页
        #     res = self.client.search(index=self.index, body=q)
        #     return [self._to_document(hit) for hit in res.get("hits", {}).get("hits", [])]

        # --- 获取所有文档，使用 scroll ---
        res = self.client.search(index=self.index, body=q, scroll=scroll)
        sid = res["_scroll_id"]
        hits = res["hits"]["hits"]

        all_res = deepcopy(hits)

        # all_docs = [self._to_document(hit) for hit in hits]

        while hits:
            res = self.client.scroll(scroll_id=sid, scroll=scroll)
            sid = res["_scroll_id"]
            hits = res["hits"]["hits"]
            all_res.extend(hits)

        return all_res




    def filter_documents(
        self,
        filters: Optional[Dict[str, Any]] = None,
        size: int = 10,
        all: bool = True,
        scroll: str = "2m"
    ) -> List[Document]:
        """
        过滤并返回文档。
        - filters: 过滤条件
        - size: 单批大小
        - all: 是否获取所有文档（使用 scroll）
        - scroll: scroll 游标有效期
        """
        if not filters:
            q = {"query": {"match_all": {}}, "size": size}
        else:
            q = {"query": self._build_filter_query(filters), "size": size}

        if not all:
            # 原始逻辑，只取一页
            res = self.client.search(index=self.index, body=q)
            return [self._to_document(hit) for hit in res.get("hits", {}).get("hits", [])]

        # --- 获取所有文档，使用 scroll ---
        res = self.client.search(index=self.index, body=q, scroll=scroll)
        sid = res["_scroll_id"]
        hits = res["hits"]["hits"]

        all_docs = [self._to_document(hit) for hit in hits]

        while hits:
            res = self.client.scroll(scroll_id=sid, scroll=scroll)
            sid = res["_scroll_id"]
            hits = res["hits"]["hits"]
            all_docs.extend([self._to_document(hit) for hit in hits])

        return all_docs


    # def filter_documents(
    #     self,
    #     filters: Optional[Dict[str, Any]] = None,
    #     size: int = 10,
    #     all: bool = True,
    #     scroll: str = "2m",
    #     mode = "doc"
    # ) -> List[Document]:
    #     """
    #     过滤并返回文档。
    #     - filters: 过滤条件
    #     - size: 单批大小
    #     - all: 是否获取所有文档（使用 scroll）
    #     - scroll: scroll 游标有效期
    #     """
    #     if not filters:
    #         q = {"query": {"match_all": {}}, "size": size}
    #     else:
    #         q = {"query": self._build_filter_query(filters), "size": size}

    #     if not all:
    #         # 原始逻辑，只取一页
    #         res = self.client.search(index=self.index, body=q)

    #         if mode == "entity":
    #             return [self._to_entity(hit) for hit in res.get("hits", {}).get("hits", [])]
    #         elif mode == "relation":
    #             return [self._to_relation(hit) for hit in res.get("hits", {}).get("hits", [])]
    #         else:
    #             return [self._to_document(hit) for hit in res.get("hits", {}).get("hits", [])]

    #     # --- 获取所有文档，使用 scroll ---
    #     res = self.client.search(index=self.index, body=q, scroll=scroll)
    #     sid = res["_scroll_id"]
    #     hits = res["hits"]["hits"]


    #     if mode == "entity":
    #         all_docs = [self._to_entity(hit) for hit in hits]
    #     elif mode == "relation":
    #         all_docs = [self._to_relation(hit) for hit in hits]
    #     else:
    #         all_docs = [self._to_document(hit) for hit in hits]

    #     # all_docs = [self._to_document(hit) for hit in hits]

    #     while hits:
    #         res = self.client.scroll(scroll_id=sid, scroll=scroll)
    #         sid = res["_scroll_id"]
    #         hits = res["hits"]["hits"]

    #         if mode == "entity":
    #             all_docs.extend([self._to_entity(hit) for hit in hits])
    #         elif mode == "relation":
    #             all_docs.extend([self._to_relation(hit) for hit in hits])
    #         else:
    #             all_docs.extend([self._to_document(hit) for hit in hits])


    #     return all_docs


    # ------------------------------ vector search --------------------------- #
    def knn_search(
            self,
            query_embedding: List[float],
            *,
            top_k: int = 5,
            num_candidates: Optional[int] = None,
            filters: Optional[Dict[str, Any]] = None,
            source_fields: Optional[List[str]] = None,
    ) -> List[Document]:
        """KNN vector search using Lindorm/OpenSearch `knn` query.

        Parameters
        ----------
        query_embedding : List[float]
            Dense embedding of the query.
        top_k : int
            Number of results to return.
        num_candidates : int | None
            Optional candidate list size for approximate search.
        filters : dict | None
            Additional scalar filters expressed as OpenSearch field names
            (e.g. {"meta.lang": "zh"}).
        source_fields : List[str] | None
            Fields to include in _source. Defaults to content+meta.
        """
        knn_clause: Dict[str, Any]
        # OpenSearch knn query supports two shapes; use field->vector object
        if num_candidates is None:
            knn_clause = {
                "knn": {self.vector_field: {"vector": query_embedding, "k": int(top_k)}}
            }
        else:
            knn_clause = {
                "knn": {
                    self.vector_field: {
                        "vector": query_embedding,
                        "k": int(top_k),
                        "num_candidates": int(num_candidates),
                    }
                }
            }

        base_query: Dict[str, Any] = {"size": int(top_k), "query": knn_clause}

        if filters:
            # Combine knn with scalar filters using a bool query + rescore (simple approach)
            base_query = {
                "size": int(top_k),
                "query": {
                    "bool": {
                        "must": [knn_clause],
                        "filter": self._build_filter_query(filters),
                    }
                },
                # "ext": {"lvector": {"min_score": "0.1"}}
            }
        # print(json.dumps(base_query, indent=2))
        if source_fields is None:
            source_fields = [self.content_field, self.meta_field]
        base_query["_source"] = {"includes": source_fields}

        res = self.client.search(index=self.index, body=base_query)
        return [self._to_document(hit) for hit in res.get("hits", {}).get("hits", [])]





    # ------------------------------- utilities ----------------------------- #
    def _to_document(self, hit: Dict[str, Any]) -> Document:
        src = hit.get("_source", {})
        return Document(
            id=str(hit.get("_id")),
            content=src.get(self.content_field, ""),
            meta=src.get(self.meta_field, {}) or {},
            # Embeddings are excluded from _source by default; set to None
            embedding=None,
            score=float(hit.get("_score", 0.0)),
        )
    
    # # 这里被裁掉了，记得在这里进行补全，先看看我们需要什么

    @staticmethod
    def _json_dumps(obj: Any) -> str:
        import json

        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

# ------------------------------ Retriever --------------------------------- #
@component
class LindormRetriever:
    """Retriever that recalls documents from a LindormDocumentStore via vector KNN.

    Inputs
    ------
    query_embedding : List[float]
        Dense embedding for the query (produce with your embedder component).
    filters : Dict[str, Any] | None
        Optional scalar filters (e.g. {"meta.category": "tech"}).
    top_k : int | None
        Optional override of the retriever's default top_k.

    Outputs
    -------
    documents : List[Document]
        Retrieved Documents with scores.
    """

    def __init__(
            self,
            kb_id,
            document_store: LindormDocumentStore,
            *,
            top_k: int = 20,
            num_candidates: Optional[int] = None,
            source_fields: Optional[List[str]] = None,
            default_filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.store = document_store
        self.top_k = int(top_k)
        self.num_candidates = num_candidates
        self.kb_id = kb_id
        self.source_fields = ["content", "meta", "kb_id"]
        self.default_filters = default_filters or {}

    @component.output_types(documents=List[Document])
    def run(
            self,
            query_embedding: List[float],
            filters: Optional[Dict[str, Any]] = None,
            top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Merge default filters with per-call filters
        merged_filters = {**(self.default_filters or {}), **(filters or {})}
        merged_filters["kb_id"] = self.kb_id
        k = int(top_k or self.top_k)
        docs = self.store.knn_search(
            query_embedding=query_embedding,
            top_k=k,
            num_candidates=self.num_candidates,
            filters=merged_filters or None,
            source_fields=self.source_fields,
        )
        return {"documents": docs}


class LindormBM25Retriever:
    # \"\"\"Keyword (BM25) retriever for LindormDocumentStore.\"\"\"
    def __init__(
        self,
        document_store,
        kb_id,
        top_k: int = 10,
        fuzziness: Optional[str] = None,
        all_terms_must_match: bool = False
    ):
        self.kb_id = kb_id
        self.document_store = document_store
        self.top_k = top_k
        self.fuzziness = fuzziness
        self.all_terms_must_match = all_terms_must_match


    def _build_filter_query(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 filters 转为 ES/Lindorm 的 filter 子句（bool.filter 可接收这个返回值）。
        filters 示例: {"kb_id": "kb_123", "meta.tag": ["a","b"], "title": "foo*"}
        规则：
        - 值为 list -> terms
        - 值为字符串包含 '*' -> wildcard
        - 其余 -> term
        """
        clauses = []
        for field, value in filters.items():
            if isinstance(value, list):
                clauses.append({"terms": {field: value}})
            elif isinstance(value, str) and ("*" in value or "?" in value):
                clauses.append({"wildcard": {field: value}})
            else:
                clauses.append({"term": {field: value}})
        # filter 接受一个数组或单个对象，返回一个 bool 下的 filter 数组
        return clauses

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        k = top_k or self.top_k
        merged_filters = {**(filters or {})}
        merged_filters["kb_id"] = self.kb_id  # 强制限定在该知识库
        # 基本 match 查询
        body: Dict[str, Any] = {
            "query": {
                "match": {
                    self.document_store.content_field: {
                        "query": query
                    }
                }
            },
            "size": k
        }
        if self.fuzziness is not None:
            body["query"]["match"][self.document_store.content_field]["fuzziness"] = self.fuzziness
        if self.all_terms_must_match:
            body["query"]["match"][self.document_store.content_field]["operator"] = "and"

        # 如果有 filters，使用 bool + filter
        if merged_filters:
            filter_clause = self.document_store._build_filter_query(merged_filters)
            body = {
                "query": {
                    "bool": {
                        "must": body["query"],
                        "filter": filter_clause
                    }
                },
                
                "ext": {
                    "lvector": {
                        "filter_type": "pre_filter"
                    }
                },

                "size": k
            }

        res = self.document_store.client.search(index=self.document_store.index, body=body)
        max_score = res.get("hits", {}).get("max_score", 1)
        docs: List[Document] = []
        for hit in res.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})
            docs.append(
                Document(
                    id=str(hit.get("_id")),
                    content=src.get(self.document_store.content_field, ""),
                    meta=src.get(self.document_store.meta_field, {}) or {},
                    score=float(hit.get("_score", 0.0))/max_score
                )
            )
        return {"documents": docs}

