from __future__ import annotations

from typing import List, Dict, Any, Optional, Set

# 假设 LindormDocumentStore 和 OpenSearch 已经在别处定义好
from .lindorm import LindormDocumentStore


class LindormGraphStorageMinimal:
    """
    用 Lindorm Search/OpenSearch 模拟的“图存储”组件：

    - 实体（node）存到一个索引（entities_store.index），结构类似：
        {
            "id": "entity:Graph_Update",
            "name": "Graph_Update",
            "type": "Function",
            "description": "...",
            "chunk_ids": ["chunk_1", "chunk_5"],
            "meta": {...}
        }

    - 关系（edge）存到另一个索引（relations_store.index），结构类似：
        {
            "id": "rel:Graph_Update->Lindorm:create_index",
            "src_id": "entity:Graph_Update",
            "tgt_id": "entity:LindormDocumentStore",
            "relation_type": "CALLS",
            "keywords": ["create_index", "graph"],
            "description": "...",
            "chunk_ids": ["chunk_2", "chunk_7"],
            "meta": {...}
        }

    主要能力：
    - upsert_entity(...)         插入/更新实体
    - get_entity(...)            获取实体
    - upsert_relation(...)       插入/更新关系
    - get_relations_of_entity()  获取某个实体参与的所有关系
    - get_neighbors()            获取某个实体的一阶邻居实体 + 经过的关系
    """

    def __init__(
        self,
        entities_store: "LindormDocumentStore",
        relations_store: "LindormDocumentStore",
        *,
        auto_refresh: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        entities_store : LindormDocumentStore
            用于存实体的 DocumentStore，对应一个实体索引，例如 "kg_entities_xxx"
        relations_store : LindormDocumentStore
            用于存关系的 DocumentStore，对应一个关系索引，例如 "kg_relations_xxx"
        auto_refresh : bool
            写入后是否立即 refresh 索引。数据量不大、方便调试时可以 True，
            大规模写入时建议 False，改为外部批量 refresh。
        """
        self.entities_store = entities_store
        self.relations_store = relations_store
        self.auto_refresh = auto_refresh

    # ============================================================
    # 实体相关
    # ============================================================

    def upsert_entity(
        self,
        kb_id: str,
        entity_id: str,
        name: str,
        etype: str,
        description: str = "",
        chunk_ids: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        插入或更新一个实体。

        kb_id : 数据库id 
        entity_id：实体唯一 ID（你可以用 entity_name，或者带前缀的 'entity:xxx'）
        name：实体名称（展示用）
        etype：实体类型（Person / Paper / Function / Concept ...）
        description：实体描述
        chunk_ids：该实体出现过的证据 chunk 列表
        meta：额外元数据（kb_id、source 等）
        """
        body = {
            "kb_id": kb_id,
            "id": entity_id,
            "name": name,
            "type": etype,
            "description": description,
            "chunk_ids": list(chunk_ids or []),
            "meta": meta or {},
        }

        self.entities_store.client.index(
            index=self.entities_store.index,
            id=entity_id,
            body=body,
            refresh="wait_for" if self.auto_refresh else False,
        )



    def bulk_upsert_entities(
        self,
        kb_id: str,
        entities: List[Dict[str, Any]],
    ) -> None:
        """
        批量插入或更新实体。

        """
        if not entities:
            return

        actions = []
        refresh = "wait_for" if self.auto_refresh else False

        for e in entities:
            entity_id = e["entity_id"]
            name = e.get("name", entity_id)
            # etype = e.get("etype", "Other")
            description = e.get("description", "")
            chunk_ids = list(e.get("chunk_ids") or [])
            meta = e.get("meta") or {}


            if kb_id is None:
                raise ValueError("kb_id is None in bulk_upsert_entities")


            # 存入的东西是这样子的吗
            body = {
                "entity_id": entity_id,
                "kb_id": str(kb_id),
                "name": name,
                "description": description,
                "meta": meta,
                "source_doc": chunk_ids
            }

            # OpenSearch bulk 格式：一个 action 行 + 一个 source 行
            actions.append(
                {
                    "index": {
                        "_index": self.entities_store.index,
                        "_id": entity_id,
                    }
                }
            )
            actions.append(body)
    
        # 真正调用 bulk 写入
        self.entities_store.client.bulk(
            body=actions,
            refresh=refresh,
        )


    # def get_entity(self, entity_id: str, kb_id: str) -> Optional[Dict[str, Any]]:
    #     """
    #     根据 entity_id 获取实体文档，找不到返回 None。
    #     """
    #     try:
    #         res = self.entities_store.client.get(
    #             index=self.entities_store.index,
    #             id=entity_id,
    #         )
    #         return res.get("_source")
    #     except Exception:
    #         # 不区分 404/其他错误，简单返回 None，
    #         # 真要细分可以根据异常类型来处理
    #         return None


    def get_entity(self, nid: str, kb_id: str) -> Optional[Dict[str, Any]]:
        """
        在指定 kb_id 中，通过 filter 精确筛选 name == nid 的实体。
        命中返回 entity(dict)，未命中返回 None。

        说明：
        - 优先用 term + name.keyword 做精确匹配（推荐）。
        - 若索引没有 name.keyword，则 fallback 到 term(name) 或 match(name)。
        """
        if not nid:
            raise ValueError("nid (name) must be non-empty")
        if not kb_id:
            raise ValueError("kb_id must be non-empty")

        client = self.entities_store.client
        index = self.entities_store.index

        # 1) 优先：精确匹配（keyword）
        query = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"kb_id": kb_id}},
                        {"term": {"name.keyword": nid}},
                    ]
                }
            },
            "size": 2,   # 取 2 条用于检测同名重复
        }
        try:
            resp = client.search(index=index, body=query)
        except Exception:
            # 2) fallback：有些 mapping 没有 name.keyword（或字段不是 keyword）
            # 先尝试 term(name)，仍不行再改 match(name)
            query["query"]["bool"]["filter"][1] = {"term": {"name": nid}}
            resp = client.search(index=index, body=query)

        hits = (resp.get("hits") or {}).get("hits") or []
        if not hits:
            return None

        # 如果同名多条，你需要定义策略；这里默认取第一条，同时可选择报错更安全
        if len(hits) > 1:
            # 更稳妥：直接报错，避免返回错误实体
            # raise ValueError(f"Multiple entities found for kb_id={kb_id}, name={nid}")
            pass

        hit = hits[0]
        src = hit.get("_source") or {}
        src["_id"] = hit.get("_id")
        return src

    # ============================================================
    # 关系相关
    # ============================================================

    def upsert_relation(
        self,
        kb_id: str,
        relation_id: str,
        src_id: str,
        tgt_id: str,
        relation_type: str,
        description: str = "",
        keywords: Optional[List[str]] = None,
        chunk_ids: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        插入或更新一条关系（边）。

        relation_id：关系唯一 ID（例如 'src->tgt:type'）
        src_id / tgt_id：两端的实体 ID（注意要和实体索引里保持一致）
        relation_type：关系类型（CALLS / WORKS_ON / PART_OF / RELATED_TO ...）
        description：关系描述
        keywords：一些关键词，便于搜索
        chunk_ids：作为这条边证据的 chunk 列表
        meta：额外元数据（kb_id、source 等）
        """
        if kb_id is None:
            raise ValueError("kb_id is None in bulk_upsert_entities")

        body = {
            "kb_id": str(kb_id),
            "id": relation_id,
            "src_id": src_id, 
            "tgt_id": tgt_id,
            "relation_type": relation_type,
            "description": description,
            "keywords": list(keywords or []),
            "chunk_ids": list(chunk_ids or []),
            "meta": meta or {},
        }

        self.relations_store.client.index(
            index=self.relations_store.index,
            id=relation_id,
            body=body,
            refresh="wait_for" if self.auto_refresh else False,
        )

    def bulk_upsert_relations(
        self,
        kb_id: str,
        relations: List[Dict[str, Any]],
    ) -> None:
        """
        批量插入或更新关系（边）。

        relations: 每个元素形如：
        {
            "relation_id": str,
            "src_id": str,
            "tgt_id": str,
            "relation_type": str,
            "description": str,
            "keywords": List[str],
            "chunk_ids": List[str],
            "meta": Dict[str, Any],
        }
        """
        if not relations:
            return

        actions = []
        refresh = "wait_for" if self.auto_refresh else False

        for r in relations:
            relation_id = r["relation_id"]
            src_id = r["src_id"]
            tgt_id = r["tgt_id"]
            relation_type = r.get("relation_type", "related_to")
            description = r.get("description", "")
            keywords = list(r.get("keywords") or [])
            chunk_ids = list(r.get("chunk_ids") or [])
            meta = r.get("meta") or {}

            body = {
                "kb_id": kb_id,
                "relation_id": relation_id,
                "src_id": src_id,
                "tgt_id": tgt_id,
                "relation_type": relation_type,
                "description": description,
                "keywords": keywords,
                "meta": meta,
                "source_doc": chunk_ids
            }

            actions.append(
                {
                    "index": {
                        "_index": self.relations_store.index,
                        "_id": relation_id,
                    }
                }
            )
            actions.append(body)

        self.relations_store.client.bulk(
            body=actions,
            refresh=refresh,
        )

    def get_relations_of_entity(
        self,
        entity_id: str,
        kb_id: str,
        size: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        获取某个实体参与的所有关系（包括出边 + 入边）。

        返回的每个元素是关系索引中的 _source：
        {
            "id": ...,
            "src_id": ...,
            "tgt_id": ...,
            "relation_type": ...,
            "description": ...,
            "keywords": [...],
            "chunk_ids": [...],
            "meta": {...}
        }
        """
        query = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"kb_id": kb_id}},   # 若是 keyword/数字字段，直接这样
                    ],
                    "should": [
                        {"term": {"src_id.keyword": entity_id}},
                        {"term": {"tgt_id.keyword": entity_id}}

                    ],
                    "minimum_should_match": 1,
                }
            },
            "size": size,
        }

        res = self.relations_store.client.search(
            index=self.relations_store.index,
            body=query,
        )
        hits = res.get("hits", {}).get("hits", [])
        return [h.get("_source", {}) for h in hits]

    # ============================================================
    # 邻居查询
    # ============================================================

    def get_neighbors(
        self,
        entity_id: str,
        kb_id: str,
        size: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        获取某个实体的一阶邻居实体。

        返回结构：
        [
            {
                "entity": <邻居实体文档>,
                "via_relations": [<关系文档>, ...]  # 通过哪些边连到它
            },
            ...
        ]
        """

        # 通过实体id和数据库id去寻找关系
        relations = self.get_relations_of_entity(entity_id, size=size, kb_id = kb_id)

        # 收集所有邻居实体 ID
        neighbor_ids: Set[str] = set()
        for rel in relations:
            src = rel.get("src_id")
            tgt = rel.get("tgt_id")
            # 找到了有关系的其他实体
            if src and src != entity_id:
                neighbor_ids.add(src)
            if tgt and tgt != entity_id:
                neighbor_ids.add(tgt)

        neighbors: List[Dict[str, Any]] = []


        # 通过id去找到相应的实体吗
        for nid in neighbor_ids:
            ent = self.get_entity(nid, kb_id)
            if not ent:
                continue

            # 找出所有刚好连接 entity_id 和 nid 的边
            via = [
                r
                for r in relations
                if {r.get("src_id"), r.get("tgt_id")} == {entity_id, nid}
            ]

            neighbors.append(
                {
                    "entity": ent,
                    "via_relations": via,
                }
            )

        return neighbors
