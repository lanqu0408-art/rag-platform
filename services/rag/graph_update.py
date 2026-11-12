
import os
from client.Client import Client
from .lindorm.knowledge_base_ab import KnowledgeBaseABDocumentStore
from utils.wrapper import Wrapper
from .lindorm.lindorm import LindormDocumentStore
from hashlib import md5
import re
from .prompt import PROMPTS
from ..external_api.aizt_api import aizt_chat
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional
from .lindorm.lindorm_graph import LindormGraphStorageMinimal
import traceback

class Graph_Update:
    def __init__(self):
        env = os.environ['env']
        self.language = "Chinese" 
        self.entity_types = ['Person', 'Creature', 'Organization', 'Location', 'Event', 'Concept', 'Method', 'Content', 'Data', 'Artifact', 'NaturalObject', 'System']
        client = Client()
        self.lindorm_url = client.get_value("lindorm_url")
        self.lindorm_key = client.get_value("lindorm_key")
        self.lindorm_value = client.get_value("lindorm_value")

        self.kb_info_client = KnowledgeBaseABDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            index="kbab_" + env + "_new_3"
        )

        # 存储graph信息的
        self.graph_info_client = KnowledgeBaseABDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            index="kbab_" + env + "_graph"
        )

        self.document_store= LindormDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            embedding_dim=1024,
            ) # 读的库 # 后续制定index
        
        self.entities_store = LindormDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            )

        self.relations_store = LindormDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value)
            )

    # 存入唯一的chunk_id的事情需要解决
    def compute_mdhash_id(self, content: str, prefix: str = "") -> str:
        """
        Compute a unique ID for a given content string.

        The ID is a combination of the given prefix and the MD5 hash of the content string.
        """
        return prefix + self.compute_args_hash(content)
    
    def compute_args_hash(self, *args):
        """Compute a hash for the given arguments with safe Unicode handling.

        Args:
            *args: Arguments to hash
        Returns:
            str: Hash string
        """
        # Convert all arguments to strings and join them
        args_str = "".join([str(arg) for arg in args])

        # Use 'replace' error handling to safely encode problematic Unicode characters
        # This replaces invalid characters with Unicode replacement character (U+FFFD)
        try:
            return md5(args_str.encode("utf-8")).hexdigest()
        except UnicodeEncodeError:
            # Handle surrogate characters and other encoding issues
            safe_bytes = args_str.encode("utf-8", errors="replace")
            return md5(safe_bytes).hexdigest()


    # 这个函数是将新的 entity 和relation merge 到已有的库中 # 不太对，这个函数缺功能，要在增量的时候实现一下
    # 看看如何实现的


    # 这里要优化一下
    def merge_nodes_and_edges_basic_sync(
        self,
        chunk_results: List[Tuple[dict, dict]],
        graph_storage,
        kb_id = None
    ):
        """
        把所有 chunk 的 (maybe_nodes, maybe_edges) 合并后，
        直接写入 LindormGraphStorageMinimal 对应的实体索引 / 关系索引，
        """

        # 合并所有 chunk 的 nodes / edges
        all_nodes: Dict[str, list] = defaultdict(list)
        all_edges: Dict[tuple, list] = defaultdict(list)

        # 拿到这次的 chunk_result 

        # chunk_result装的是所有chunk的entity 和 edge
        for maybe_nodes, maybe_edges in chunk_results:
            # maybe_nodes: {entity_name: [entity_dict, ...]}
            for entity_name, entities in maybe_nodes.items():
                all_nodes[entity_name].extend(entities)

            # maybe_edges: {(src_name, tgt_name): [edge_dict, ...]}
            for edge_key, edges in maybe_edges.items():
                sorted_edge_key = tuple(sorted(edge_key))
                all_edges[sorted_edge_key].extend(edges)
        
        # 收集好所有的entity 和 relation
        # 以dict形式
        # key 是 entity名字
        # value 是 在不同chunk处出现的值

        if self.graph_info_client.get_construct_type(str(999997)).get("cancel", "0") == "1":
            self.post_process_kb_by_cancel(kb_id, num = 1)
            return True
        
        # 1. 先在内存里收集所有实体，稍后一次性 bulk 写入
        entity_docs: list[dict] = []
        processed_entity_ids: list[str] = []

        for entity_name, entities in all_nodes.items():
            if not entities:
                continue

            first = entities[0] # 这个按照0给的话是没问题的
            etype = first.get("entity_type", "Other")
            desc = first.get("entity_description", "")
            entity_id = first.get("entity_id") or entity_name

            # 汇总所有出现过的 chunk_id
            # 实例的话是不是只需要同名的用一个就行
            # 毕竟descripe都是模型给的
            chunk_ids = set()
            for e in entities:
                cid = e.get("doc_url")
                if cid:
                    if isinstance(cid, list):
                        for sub_cid in cid:
                            chunk_ids.add(sub_cid)
                    else:
                        chunk_ids.add(cid)

            meta = {
                "doc_ids": list({e.get("doc_id") for e in entities if e.get("doc_id")}),
                "occurrence_count": len(entities),
            }

            cur_id = self.compute_mdhash_id(desc)

            entity_docs.append(
                {
                    "entity_id": cur_id,
                    "name": entity_name,
                    "etype": etype,
                    "description": desc,
                    "chunk_ids": list(chunk_ids),
                    "meta": meta,
                }
            )

            processed_entity_ids.append(entity_id)

        # 调用bulk
        if entity_docs:
            graph_storage.bulk_upsert_entities(
                kb_id=str(kb_id),
                entities=entity_docs,
            )

        # 重复关系如何处理
        # 增量也是从这里进行改动
        # 2. 收集所有关系，同样一次性 bulk 写入
        relation_docs: list[dict] = []
        processed_rel_pairs: set[tuple[str, str]] = set()

        for edge_key, edges in all_edges.items():
            if not edges:
                continue

            src_name, tgt_name = edge_key
            # 这里做合并
            first = edges[0]  # 这个也拿第0个有点不合理吧 # 如果有不同的关系怎么半呢

            rel_keywords = first.get("relationship_keywords", "")
            rel_type = rel_keywords.split(",")[0].strip() if rel_keywords else "related_to"
            desc = first.get("relationship_description", "")
            src_id = first.get("src_id") or src_name
            tgt_id = first.get("tgt_id") or tgt_name

            # 汇总证据 chunk
            chunk_ids = set()
            for r in edges:
                cid = r.get("doc_url")
                if isinstance(cid, list):
                    for sub_cid in cid:
                        chunk_ids.add(sub_cid)
                else:
                # cid = r.get("doc_url")
                # if cid:
                    chunk_ids.add(cid)

            relation_id = first.get("relation_id") or f"{src_id}->{tgt_id}:{rel_type}"

            meta = {
                "doc_ids": list({r.get("doc_id") for r in edges if r.get("doc_id")}),
                "occurrence_count": len(edges),
            }

            if len(edges) > 1:
                # 多个关系，需要模型处理
                content = ""
                content += "第一个实体是 "
                content += edges[0]["src_id"]
                content += "，第二个实体是"
                content += edges[0]["tgt_id"]

                for cur_idx in range(len(edges)):
                    cur_edge = edges[cur_idx]
                    cur_desc = cur_edge['description']
                    content +=  f", 第{cur_idx + 1}个关系描述是 "
                    content += cur_desc
                prompt = "根据下面内容, 整合关于两个实体的关系，只返回整合后的关系描述，不要返回其它任何东西。当前多个关系描述如下："
                prompt += content
                desc = aizt_chat(prompt, model_type="qwen3")

            cur_id = self.compute_mdhash_id(desc)

            relation_docs.append(
                {
                    "relation_id": cur_id,
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "relation_type": rel_type,
                    "description": desc,
                    "keywords": [k.strip() for k in rel_keywords.split(",") if k.strip()],
                    "chunk_ids": list(chunk_ids),
                    "meta": meta,
                }
            )

            processed_rel_pairs.add(tuple(sorted([src_id, tgt_id])))

        if self.graph_info_client.get_construct_type(str(999997)).get("cancel", "0") == "1":
            self.post_process_kb_by_cancel(kb_id, num = 2)
            return True

        if relation_docs:
            graph_storage.bulk_upsert_relations(
                kb_id=str(kb_id),
                relations=relation_docs,
            )

        return False

    # 参考lightrag，处理模型的输出
    def process_extraction_result(
        self,
        result: str,
        chunk_key: str = "",
        file_path: str = "unknown_source",
        tuple_delimiter: str = "<|#|>",
        completion_delimiter: str = "<|COMPLETE|>",
        doc_url: str = None
    ) -> tuple[dict, dict]:
        """
        # 参考lightrag，处理模型的输出
        简化版：根据 LLM 输出的 entity/relation 行，解析成节点和边。

        假设 LLM 输出严格为：
        entity<|#|>entity_name<|#|>entity_type<|#|>entity_description
        relation<|#|>source_entity<|#|>target_entity<|#|>relationship_keywords<|#|>relationship_description
        ...
        <|COMPLETE|>

        返回:
            maybe_nodes: { entity_name: [entity_dict, ...] }
            maybe_edges: { (src_id, tgt_id): [relation_dict, ...] }
        """

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)

        if completion_delimiter not in result:
            print(f"{chunk_key}: COMPLETE delimiter not found in extraction result")

        lines = result.splitlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line == completion_delimiter:
                break

            parts = line.split(tuple_delimiter)
            if not parts:
                continue

            kind = parts[0].strip().lower()

            # ---- 实体 ----
            if kind == "entity":
                if len(parts) < 4:
                    print(f"{chunk_key}: invalid entity record: {line}")
                    continue

                entity_name = parts[1].strip()
                entity_type = parts[2].strip()
                entity_desc = parts[3].strip()

                entity_data = {
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "description": entity_desc,
                    "entity_description": entity_desc,
                    "chunk_id": chunk_key,
                    "file_path": file_path,
                    "doc_url": doc_url
                }
                maybe_nodes[entity_name].append(entity_data)

            # ---- 关系 ----
            elif kind in ("relation", "relationship"):
                if len(parts) < 5:
                    print(f"{chunk_key}: invalid relation record: {line}")
                    continue

                src = parts[1].strip()
                tgt = parts[2].strip()
                keywords = parts[3].strip()
                desc = parts[4].strip()

                relation_data = {
                    "src_id": src,
                    "tgt_id": tgt,
                    "description": desc,
                    "relationship_keywords": keywords,
                    "weight": 1,  # 每条原始关系给一个基础权重 # 这个是默认的，之后得看看怎么给这个权重c
                    "relationship_description": desc,
                    "keywords": keywords,  # ⭐ 必须是字符串
                    "chunk_id": chunk_key,
                    "file_path": file_path,
                    "doc_url":  doc_url
                }
                maybe_edges[(src, tgt)].append(relation_data)

            else:
                print(f"{chunk_key}: unknown record type: {line}")
                continue

        return dict(maybe_nodes), dict(maybe_edges)

    # 从chunk的content中提取实体
    def extract_graph_info(self, chunks, kb_id):

        """
        从 chunks 提取 entity

        返回值：
            list[tuple[dict, dict]]，每个元素是 (maybe_nodes, maybe_edges)
            后面 merge_nodes_and_edges 仍然可以直接使用。

            bool ，用于判断任务是不是被暂停
        """
        # 提取当前chunk的信息
        total_chunks = len(chunks)
        
        # 获取examples写入pe
        examples = "\n".join(PROMPTS["entity_extraction_examples"])
        example_context_base = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            entity_types=", ".join(self.entity_types),
            language=self.language,
        )
        examples = examples.format(**example_context_base)
        

        context_base = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            entity_types=",".join(self.entity_types),
            examples=examples,
            language=self.language,
        )

        processed_chunks = 0
        chunk_results = [] # 处理这个就好，增量的话，先把相关的，之前存储过的，放进来这里面，然后重新merge 并进行构图

        # 这里存入的时候需要用到一个唯一id，用于后续做去重
        # # 顺序处理每个 chunk，不再做并发ll
        for i in range(len(chunks)):
            content = chunks[i].content
            # 对于每一个content 生成 一个 唯一的chunk_id
            
            doc_url = chunks[i].meta['doc_url']
            entity_extraction_system_prompt = PROMPTS[
                "entity_extraction_system_prompt"
            ].format(**{**context_base, "input_text": str(content)})  # 构建提取的prompt
            entity_extraction_user_prompt = PROMPTS[
                "entity_extraction_user_prompt"
            ].format(**{**context_base, "input_text": str(content)})
            # 构建两个prompt，使用的是lightrag中的pe

            graph_content = aizt_chat(entity_extraction_user_prompt, 
                                      system_content = entity_extraction_system_prompt, 
                                      model_type="qwen3",
                                      temperature=0,
                                      presence_penalty=0,
                                      frequency_penalty=0)

            if self.graph_info_client.get_construct_type(str(999997)).get("cancel", "0") == "1":
                self.post_process_kb_by_cancel(kb_id, num = 3)
                return chunk_results, True

            # 从模型输出中找到node和edges
            # 这个怎么跨chunk

            # 一个chunk的实例和关系
            doc_url = chunks[i].meta['doc_url']
            maybe_nodes, maybe_edges = self.process_extraction_result(
                graph_content,
                tuple_delimiter=context_base["tuple_delimiter"],
                completion_delimiter=context_base["completion_delimiter"],
                doc_url = doc_url
            )


            # 我想要的不是这个chunkid ，而是存进去的时候，bulk的时候给的唯一id
            if self.graph_info_client.get_construct_type(str(999997)).get("cancel", "0") == "1":
                self.post_process_kb_by_cancel(kb_id, num = 4)
                return chunk_results, True

            processed_chunks += 1
            entities_count = len(maybe_nodes)
            relations_count = len(maybe_edges)
            print(f"Chunk {processed_chunks} of {total_chunks} ")
            print(f"extracted {entities_count} Ent + {relations_count} Rel")
            # 转成个三元组吧
            # 只保留文档内部的关系
            # 文档之间的关系重新进merge
            # 感觉ok
            # 这个merge 是不是不太对
            chunk_results.append((maybe_nodes, maybe_edges)) 
        return chunk_results, False
        
    def kg_query_lindorm_basic(
        self,
        query: str,
        *,
        graph_storage,                 # LindormGraphStorageMinimal 实例
        entities_store,                # LindormDocumentStore，对应 kg_entities_dev
        system_prompt: Optional[str] = None,
        top_k_entities: int = 5,
        top_k_neighbors_per_entity: int = 3,
    ):
        """
            使用 Lindorm Search + 向量库 + 伪图存储 的极简 Graph-RAG 查询函数。

            步骤：
            1. 在 kg_entities_dev 里按文本检索相关实体
            2. 对这些实体，调用 graph_storage.get_neighbors(...) 拿一阶邻居 + 关系描述
            3. 用 chunks_vdb 做一次向量检索，拿若干相关 chunk
            4. 拼接成 context，构建 system prompt，调用 llm_func
            5. 返回 QueryResult（只处理两种结果：字符串 / Streaming iterator）
            """

        # query 一定存在
        # ----------------------------------------------------
        # 1) 从 Lindorm 实体索引中按关键词搜索相关实体
        # ----------------------------------------------------
        # 这里用最简单的 multi_match，你可以根据实际 mapping 调整字段名和 boost

        ent_search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["name^3", "description"],  # name 权重高一点
                }
            },
            "size": top_k_entities,
        }
        ent_res = entities_store.client.search(
            index=entities_store.index,
            body=ent_search_body,
        )
        ent_hits = ent_res.get("hits", {}).get("hits", [])
        entities = [h.get("_source", {}) for h in ent_hits]





        # 没命中实体也没关系，后面还有 chunk 级检索
        # ----------------------------------------------------
        # 2) 对每个实体，用伪图存储拿一阶邻居
        # ----------------------------------------------------
        graph_snippets: List[str] = []

        for ent in entities:
            entity_id = ent.get("id")
            name = ent.get("name") or entity_id
            etype = ent.get("type", "")
            desc = ent.get("description", "")

            # 当前实体描述
            snippet = f"[Entity] {name} ({etype})\n{desc}"
            graph_snippets.append(snippet)

            if not entity_id:
                continue

            neighbors = graph_storage.get_neighbors(
                entity_id=entity_id,
                size=top_k_neighbors_per_entity,
            )

            for item in neighbors:
                nb = item.get("entity") or {}
                nb_id = nb.get("id")
                nb_name = nb.get("name") or nb_id
                nb_type = nb.get("type", "")
                nb_desc = nb.get("description", "")

                # 经过哪些关系连到它
                rels = item.get("via_relations") or []
                for rel in rels:
                    r_type = rel.get("relation_type", "related_to")
                    r_desc = rel.get("description", "")
                    line = (
                        f"[Relation] {name} --{r_type}--> {nb_name}\n"
                        f"  rel_desc: {r_desc}\n"
                        f"  neighbor_desc: {nb_desc}"
                    )
                    graph_snippets.append(line)

        # ----------------------------------------------------
        # 3) 用向量库检索 chunk（普通 RAG）
        # ----------------------------------------------------
        chunk_snippets: List[str] = ["mock版本，这个通过自己的rag实现"]

        # if chunks_vdb is not None:
        #     # 这里假设你的 BaseVectorStorage 有一个 async 搜索方法，形如：
        #     #   await chunks_vdb.search(query_text, top_k=...)
        #     # 具体名字你可以改成你真实实现的那个。
        #     lindorm_retri=LindormRetriever(document_store=chunks_vdb, kb_id="9013")
        #     embed_query=aizt_embedding(query)  # 这是在文档入库
        #     chunk_hits=lindorm_retri.run(embed_query)['documents']

        #     # 期望返回结构类似：
        #     #   [{"id": "...", "content": "...", "score": ..., "meta": {...}}, ...]
        #     for ch in chunk_hits:
        #         content = ch.get("content") or ch.get("text") or ""
        #         # 一定会有content
        #         chunk_snippets.append(f"[Chunk]\n{content}")


        # ----------------------------------------------------
        # 4) 组装 context，控制一下 token 长度
        # ----------------------------------------------------
        # 简单拼接：实体/关系信息在前，chunk 在后
        all_snippets = graph_snippets + chunk_snippets
        context_text = "\n\n".join(all_snippets)

        # 粗暴用 tokenizer 截断过长的 context
        # 这个不这么阶段，而是通过最后的leng截断
        # if tokenizer is not None and context_text:
        #     tokens = tokenizer.encode(context_text)
        #     if len(tokens) > max_context_tokens:
        #         tokens = tokens[:max_context_tokens]
        #         context_text = tokenizer.decode(tokens)

        # if not context_text:
        #     # 没有任何上下文，就直接告诉用户查不到
        #     return QueryResult(content=PROMPTS["fail_response"])


        # ----------------------------------------------------
        # 5) 构造 system_prompt + 调 LLM
        # ----------------------------------------------------
        # 你原来 PROMPTS["rag_response"] 是类似：
        #   "...{response_type}...{user_prompt}...{context_data}..."
        base_sys_prompt = system_prompt or PROMPTS["rag_response"]
        sys_prompt = base_sys_prompt.format(
            response_type="Multiple Paragraphs",
            user_prompt="",      # 这里不额外加 user_prompt，有需要可以再扩展
            context_data=context_text,
        )

        # 直接调 llm_func，不做 cache、不搞多种 mode

        response = aizt_chat(query, 
                            system_content = sys_prompt, 
                            model_type="qwen3")

    def process(self, info_carrier):
        if info_carrier['indexConstructConfig'].get("enableGraph", True):
        # 这里需要判断吧是不是删除逻辑
            try:
                env = os.environ['env']
                kb_id = info_carrier['id']
                kb_used = self.kb_info_client.get(str(kb_id)) #
                self.graph_info_client.write(str(999997), str(999997), kb_construct_type={'cancel': str(0)})
                self.graph_info_client.write(str(999996), str(999996), kb_construct_type={'finish': str(1)}) # 写在kb_used中
                self.graph_info_client.write(str(999999), str(999999), kb_construct_type={'updating_kb': str(kb_id)}) # 写在kb_used中
        
                if kb_used == -1:
                    kb_used = "b"
                self.document_store.set_index("kb_" + str(kb_used) + "_" + env)
                

                # 获取graph的状态， graph 和 chunk 并不是完全同步的
                graph_used = self.graph_info_client.get(str(kb_id)) # 判断库状态  # 这个拿的是可使用的，，要拿到之后存入另一个库

                if graph_used == -1 or graph_used == "b":
                    # 证明 这个库是新建的，向a中写
                    graph_write = "a"  # 写如a
                    graph_used = "b"
                    graph_delete_a = False # 整个update结束，删除b
                elif graph_used == "a":
                    graph_write = "b" 
                    graph_delete_a = True # 整个update 结束 删除 a


                # 先设定知识库为used，filter之后再改回去 write
                self.entities_store.set_index("kg_entities_" + str(graph_used) + "_" + env)
                self.relations_store.set_index("kg_relations_" + str(graph_used) + "_" + env)


                # filter 的时候记得删除，sourceid  我加一下
                # 

                if info_carrier.get("uploadType", "FEI_SHU") == "FEI_SHU":
                    # 如果类型是飞书，则是这么处理
            
                    # 获取版本
                    native_kb_info = self.kb_info_client.get_doc_info(str(kb_id))
                    graph_kb_info = self.graph_info_client.get_doc_info(str(kb_id))
                    

                    if graph_kb_info == -1:
                        graph_kb_info = {}
                    # 然后要进行对比，找到需要更新的文档

                    new_doc_info = native_kb_info

                    # 要更新的
                    docs_update_graph = [   # 如果两个版本不一样，证明是需要更新的，这个时候里面存储的还是一个wiki token
                        k
                        for k, va in native_kb_info.items()
                        if (k not in graph_kb_info) or (va.get("version") != graph_kb_info[k].get("version"))
                    ]

                    # 直接获取的
                    docs_updated = [
                        k for k, va in native_kb_info.items()
                        if k not in docs_update_graph
                    ] # 这个也是存储的 wiki token，但是可以从知识库中进行获取的
                


                    # 需要删除的
                    docs_need_delete = [
                        k for k, va in graph_kb_info.items()
                        if k not in native_kb_info.keys()
                    ] 
                    


                    # 这里注意调整本地上传
                    for i in range(len(docs_update_graph)):
                        docs_update_graph[i] = os.path.join("https://fintopia.feishu.cn/wiki/", docs_update_graph[i]) 
                    for i in range(len(docs_updated)):
                        docs_updated[i] = os.path.join("https://fintopia.feishu.cn/wiki/", docs_updated[i]) 
                    for i in range(len(docs_need_delete)):
                        docs_need_delete[i] = os.path.join("https://fintopia.feishu.cn/wiki/", docs_need_delete[i]) 

                    # 需要后处理写入graph_kb_info的也要在这里获取

                    filters = {"kb_id": str(kb_id),
                            "meta.doc_url.keyword": docs_update_graph}
                    filter_chunks = self.document_store.filter_documents(filters = filters)
                    # 没有需要新处理的chunks

                else:
                    # 本地文件上传版本
                    new_doc_info = {}
                    docs_updated = [] # 纯增量
                    docs_need_delete = []
                    docs_update_graph = []
                    local_file = info_carrier.get("localFileSourceVOList", [])
                    for i in range(len(local_file)):
                        docs_update_graph.append(local_file[i]["fileId"])
                    filters = {"kb_id": str(kb_id),
                            "meta.doc_url.keyword": docs_update_graph}
                    filter_chunks = self.document_store.filter_documents(filters = filters)
                # 现在其实看到这里
                # 可以不用处理


                if not filter_chunks:
                    if not docs_need_delete:
                    # 没有新的chunk需要入库，也没有chunk需要被删除
                        update_res =  {
                            "code": 0,
                            "message": "没有新的chunk需要处理，图谱无需更新",
                            "data": {
                                "documentCount": -1,
                                "operationType" : "UPDATE",
                                "status": "FINISH",
                                "details": []
                            }
                        }
                        self.post_process_kb_by_cancel(str(kb_id), update_res, num = 9)
                        return {}, False
                    else:
                        # 根据id去删除就好了
                        for sub_doc_url in docs_need_delete:
                            res = self.entities_store.remove_source_doc_and_cleanup(
                                kb_id=str(kb_id),
                                doc_url=sub_doc_url
                            )
                            res = self.relations_store.remove_source_doc_and_cleanup(
                                kb_id=str(kb_id),
                                doc_url=sub_doc_url
                            )

                        update_res =  {
                            "code": 0,
                            "message": "已删除相关graph，图谱更新成功",
                            "data": {
                                "documentCount": -1,
                                "operationType" : "UPDATE",
                                "status": "FINISH",
                                "details": []
                            }
                        }
                        self.post_process_kb_by_cancel(str(kb_id), update_res, num = 5, doc_info= native_kb_info)
                        return {}, False
                        # self.graph_info_client.write(str(kb_id),graph_used, construct_info, native_kb_info, update_res) 
                        # 处理完了之后需要更新graph_client
                # 需要拼凑一下结构
                # 提取实例
                # 这里是不是就应该处理增量了


                # 这里为什么filter不到
                filters = {"kb_id": str(kb_id),
                        "source_doc.keyword": docs_updated}


                filter_graph_entity_info = self.entities_store.filter_graph_info(filters = filters) # 拿到的实例
                # filter_graph_entity_info_2 = self.entities_store.filter_documents(filters = filters)
                filter_graph_relation_info = self.relations_store.filter_graph_info(filters=filters)
                # 这里要删除一下贡献文件


                filter_graph_entity_info = self.process_entity_info(filter_graph_entity_info, docs_updated)
                filter_graph_relation_info = self.process_relation_info(filter_graph_relation_info, docs_updated)

                if not filter_graph_entity_info and not filter_graph_relation_info:
                    operationType = "CREATE"
                else:
                    operationType  = "UPDATE"

                self.operationType = operationType

                # 这里切换了一下ab库
                self.entities_store.set_index("kg_entities_" + str(graph_write) + "_" + env)
                self.relations_store.set_index("kg_relations_" + str(graph_write) + "_" + env)


                # 格式上的转换，转换完放到chunk_result就算完成了其实
                # 优化一下 lindorm存入信息，里面有好多无用信息
                chunk_results, is_cancled = self.extract_graph_info(filter_chunks, kb_id)
                chunk_results.append((filter_graph_entity_info, filter_graph_relation_info))

                if is_cancled:
                    response = {"outputs":
                                    {
                                    "status": "FINISH"
                                    }
                                }
                    return response, False
                



                # 下面是一个前处理，某些时候有问题的话，还是要二次确认删一下
                filter_entities = self.entities_store.filter_documents(filters={"kb_id": str(kb_id)})
                filter_relations = self.relations_store.filter_documents(filters={"kb_id": str(kb_id)})

                if filter_entities:  # 避免空列表导致的无效操作
                    self.entities_store.delete_documents(document_ids=filter_entities)
                if filter_relations:  # 避免空列表导致的无效操作
                    self.relations_store.delete_documents(document_ids=filter_relations)   

                # LindormGraphStorageMinimal 里面有一些同时处理entity和relation的函数，
                graph_storage = LindormGraphStorageMinimal(
                    entities_store=self.entities_store,
                    relations_store=self.relations_store,
                    auto_refresh=True,
                )
                if self.graph_info_client.get_construct_type(str(999997)).get("cancel", "0") == "1":
                    self.post_process_kb_by_cancel(kb_id, num = 6)
                    return {}, True

                is_cancled = self.merge_nodes_and_edges_basic_sync(chunk_results, graph_storage, kb_id = kb_id) # 此时已经写到库中了
                if is_cancled:
                    self.post_process_kb_by_cancel(kb_id, num = 7)
                    return {}, True    
                
                # 这里少传了一个东西，update_res
                # 这里面要获取时operationtype


                response = {"outputs":
                            {
                                "status": "FINISH",
                                "operationType":operationType
                            }
                        }

                update_res = self.get_update_info(response)

                self.post_process_kb(kb_id, graph_delete_a, doc_info = new_doc_info, update_res = update_res)

                return response, False
            except Exception as e:
                print(f"\n{traceback.format_exc()}\n")  # 测试用
                info_carrier['error'] = f"{self.__class__.__name__} error: {str(e)}"
                raise Exception(f"{self.__class__.__name__} error: {str(e)} ")
        else:
            env = os.environ['env']
            kb_id = info_carrier['id']
            self.graph_info_client.write(str(999997), str(999997), kb_construct_type={'cancel': str(0)})
            self.graph_info_client.write(str(999996), str(999996), kb_construct_type={'finish': str(1)}) # 写在kb_used中
            self.graph_info_client.write(str(999999), str(999999), kb_construct_type={'updating_kb': str(kb_id)}) # 写在kb_used中
            self.operationType = "DELETE"


            graph_used = self.graph_info_client.get(str(kb_id)) # 判断库状态  # 这个拿的是可使用的，，要拿到之后存入另一个库
            if graph_used == -1 or graph_used == "b":
                # 证明 这个库是新建的，向a中写
                graph_used = "b"
                graph_delete_a = False # 整个update结束，删除b
            else:
                graph_used = "a"
                graph_delete_a = True # 整个update 结束 删除 a
                
            
            response = {"outputs":
                        {
                            "status": "FINISH",
                            "operationType":self.operationType
                        }
                    }

            update_res = self.get_update_info(response)
            self.post_process_kb(kb_id,graph_delete_a, delete_by_kb_id = True, update_res = update_res)
            return response, False
        
    # def delete_graphkb_by_id(self, kb_id):


        


    #     # 先设定知识库为used，filter之后再改回去 write
    #     self.entities_store.set_index("kg_entities_" + str(graph_used) + "_" + env)
    #     self.relations_store.set_index("kg_relations_" + str(graph_used) + "_" + env)


    def get_update_info(self, processed_data=None):
        """准备输出数据"""
        output_data = processed_data.get('outputs', None)
        if output_data is None:
            output_data = processed_data
        output_data['details'] = []
        output_data['documentCount'] = -1
        processed_data = {
            "code": 0,
            "message": "",
            "data":output_data

        }
        return processed_data


    def process_entity_info(self, graph_entity_info, docs_updated):
        # [{'entity_name': '蚂蚁集团', 'entity_type': 'Organization', 'description': '蚂蚁集团前身为支付宝，最初是阿里巴巴集团内部的支付解决方案，于2004年正式推出，2014年从阿里巴巴集团剥离并更名为蚂蚁金服，2020年更名为蚂蚁集团，总部位于中国杭州，核心产品包括支付宝、余额宝、花呗、借呗等。', 'entity_description': '蚂蚁集团前身为支付宝，最初是阿里巴巴集团内部的支付解决方案，于2004年正式推出，2014年从阿里巴巴集团剥离并更名为蚂蚁金服，2020年更名为蚂蚁集团，总部位于中国杭州，核心产品包括支付宝、余额宝、花呗、借呗等。', 'chunk_id': '', 'file_path': 'unknown_source', 'doc_url': 'https://fintopia.feishu.cn/wiki/NEkSwHBZLiXSGokmdQFcT3pYnyf'}]
        entity_res = {}
        # docs_updated 用于删掉被更改过的文献来源
        for i in range(len(graph_entity_info)):
            cur_entity = graph_entity_info[i]
            entity_name = cur_entity['_id']
            description = cur_entity['_source'].get("description", "")
            doc_url = cur_entity['_source'].get("source_doc", [])
            docs_updated = set(docs_updated)
            doc_url[:] = [x for x in doc_url if x in docs_updated]   # 原地修改 a
            entity_res[entity_name] = [{"entity_name":entity_name,
                                          "description": description,
                                          "doc_url":doc_url}]
        return entity_res

            # doc_url_filtered = {k: v for k, v in doc_url.items() if k in docs_updated}



    def process_relation_info(self, graph_relation_info, docs_updated):
        relation_res = {}
        for i in range(len(graph_relation_info)):
            cur_relation = graph_relation_info[i]
            src_id = cur_relation['_source'].get("src_id", "")
            tgt_id = cur_relation['_source'].get("tgt_id", "")
            relation_name = (src_id, tgt_id)
            keywords = cur_relation['_source'].get("keywords", "")
            relationship_keywords = ','.join(keywords)
            
            description = cur_relation['_source'].get("description", "")


            doc_url = cur_relation['_source'].get("source_doc", [])
            docs_updated = set(docs_updated)
            doc_url[:] = [x for x in doc_url if x in docs_updated]   # 原地修改 a
            relation_res[relation_name] = [{"src_id":src_id,
                                          "tgt_id": tgt_id,
                                          "relationship_keywords":relationship_keywords,
                                          "keywords": keywords,
                                          "description":description,
                                          "doc_url":doc_url}]
        return relation_res

    def post_process_kb(self, kb_id, delete_a, construct_info={}, doc_info={}, update_res={}, delete_by_kb_id = False):
        # 如果delete_by_kb_id 为true，证明是按照库进行删除，所以无需维护 信息的那个idnex
        self.graph_info_client.write(str(999995), str(999995), kb_construct_type=update_res)
        cur_env = os.environ.get("env")
        doc_info_list = [
            {"id": k, **v} for k, v in doc_info.items()
        ]
        # self.kb_info_client.write(str(kb_name), kb_used, construct_info, doc_info_list, update_res)
        if delete_a:  
            if delete_by_kb_id:
                try:
                    self.graph_info_client.delete_by_id(kb_id=str(kb_id))
                except Exception as e:
                    print(f"[delete_by_kb_id] Failed kb_id={kb_id}: {e}")
                    return False

            else:
                self.graph_info_client.write(str(kb_id), "b", construct_info, doc_info_list, update_res)

            # 要删掉的库有两个
            self.entities_store.set_index("kg_entities_" + "a" + "_" + cur_env)
            self.relations_store.set_index("kg_relations_" + "a" + "_" + cur_env)
        else:
            if delete_by_kb_id:
                try:
                    self.graph_info_client.delete_by_id(kb_id=str(kb_id))
                except Exception as e:
                    print(f"[delete_by_kb_id] Failed kb_id={kb_id}: {e}")
                    return False

            else:
                self.graph_info_client.write(str(kb_id), "a", construct_info, doc_info_list, update_res) 
            self.entities_store.set_index("kg_entities_" + "b" + "_" + cur_env)
            self.relations_store.set_index("kg_relations_" + "b" + "_" + cur_env)
          
        filter_entities = self.entities_store.filter_documents(filters={"kb_id": str(kb_id)})
        filter_relations = self.relations_store.filter_documents(filters={"kb_id": str(kb_id)})

        entity_ids = [d.id for d in filter_entities]        # ✅ 取 _id
        relation_ids = [d.id for d in filter_relations]
        if filter_entities:  # 避免空列表导致的无效操作
            self.entities_store.delete_documents(document_ids=entity_ids)
        if filter_relations:  # 避免空列表导致的无效操作
            self.relations_store.delete_documents(document_ids=relation_ids) 
        self.graph_info_client.write(str(999999), str(999999), kb_construct_type={'updating_kb': str(-1)}) # 写在kb_used中  


    def post_process_kb_by_cancel(self, kb_name, update_res=None, num=0, doc_info = None):
        print("graph这个函数是否被调用了呢")
        print(f"第{num}处调用post函数")
        construct_info = self.graph_info_client.get_construct_type(str(kb_name))
        # 方便从外边传入
        if  doc_info == None:
            doc_info = self.graph_info_client.get_doc_info(str(kb_name))
        if construct_info == -1:
            construct_info = {}
        if doc_info == -1:
            doc_info = {}
        graph_kb_used = self.graph_info_client.get(str(kb_name)) # 判断库状态  # 这个拿的是可使用的，，要拿到之后存入另一个库
        if graph_kb_used == -1:
            graph_kb_used = "a"

        if not update_res:
            cancel_reason = self.graph_info_client.get_construct_type(str(999998))["cancel_reason"]
            update_res = {
                        "code": 1,
                        "message": cancel_reason,
                        "data": {
                            "documentCount": -1,
                            "status": "FINISH",
                            "operationType": self.operationType,
                            "details": []
                        }
                    }
        self.graph_info_client.write(str(999995), str(999995), kb_construct_type=update_res)
        # 这里会写信息
        doc_info_list = [
            {"id": k, **v} for k, v in doc_info.items()
        ]
        self.graph_info_client.write(str(kb_name), graph_kb_used, construct_info, doc_info_list, update_res)
        self.graph_info_client.write(str(999999), str(999999), kb_construct_type={'updating_kb': str(-1)}) # 写在kb_used中


class GraphUpdateWrapper(Wrapper):
    def __init__(self, config=None):
        self.instance = Graph_Update()

