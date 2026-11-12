import os
import json
import copy
import re
import base64
import traceback
from .lindorm.lindorm import LindormDocumentStore, LindormRetriever, LindormBM25Retriever
from .lindorm.lindorm_graph import LindormGraphStorageMinimal
from typing import Optional, Union, Dict, List
import threading
from haystack import Pipeline
from utils.wrapper import Wrapper
from haystack.components.joiners import DocumentJoiner
from services.external_api import ark_chat
from pprint import pprint
from haystack import Document, Pipeline
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.document_stores.types import DuplicatePolicy
from elasticsearch import Elasticsearch
from haystack.components.embedders import SentenceTransformersTextEmbedder
from .lindorm.knowledge_base_ab import KnowledgeBaseABDocumentStore
from ..external_api.aizt_api import aizt_chat, aizt_embedding, aizt_extract_keyword, aizt_rerank, aizt_rewrite
from .feishu_process import Feishu_Process
import os
import json
import concurrent.futures
from typing import Optional, Union, Dict, List
import time
from openai import OpenAI
import string
import random
import logging
from .oss import Oss_Process
# from client.Client import Client
from collections import Counter
from core.config import config
from collections import defaultdict
logger=logging.getLogger(__name__)


def extract_last_user_query(inputs):
    """从对话历史中提取上一条用户问题"""
    for msg in reversed(inputs[:-1]):  # 排除当前问题
        if msg.get('role')=='user':
            return msg.get('content', '')
    return "无法找到上一条提问"


class RAG:
    """rag Process"""

    def __init__(self):
        self.feishu_processor=Feishu_Process()
        self.oss_processor=Oss_Process()
        env=os.environ['env']
        # client=Client()
        self.lindorm_url=config.get("lindorm_url")
        self.lindorm_key=config.get("lindorm_key")
        self.lindorm_value=config.get("lindorm_value")
        self.kb_info_client=KnowledgeBaseABDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            index="kbab_"+env+"_new_3"
        )
        self.chat_url = config.get("chat_model_url")


        self.graph_info_client = KnowledgeBaseABDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            index="kbab_" + env + "_graph"
        )

        self.entities_store = LindormDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            embedding_dim=1024,  # 这里可以随便填，因为实体索引不一定用到向量
            )

        self.relations_store = LindormDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            embedding_dim=1024,  # 这里可以随便填，因为实体索引不一定用到向量
            )
        


    def process(self, info_carrier, model_chat=None, chat_tokenizer=None, text_embedder=None):
        try:
            time_start=time.time()
            self.model=model_chat
            self.tokenizer=chat_tokenizer
            self.text_embedder=text_embedder
            if info_carrier['ragConfig']['llmReplyConfig'].get('enableMultiTurn', False) and len(
                    info_carrier['inputs'])>1:  # 多轮一定会改写
                query, _, _=self.query_rewrite(info_carrier['inputs'])
            else:
                query=info_carrier['inputs'][0]['content']

            self.rerank_info = info_carrier['ragConfig'].get("rerankConfig", {})
            self.enablererank = self.rerank_info.get("enableRerank", False)
            self.rerank_model = self.rerank_info.get("rerankerModel", "Qwen/Qwen3-Reranker-8B")
            self.rerank_sim_threshold = self.rerank_info.get("similarityThreshold", 0)




            kb_info = info_carrier['ragConfig']['knowledgeBaseConfigs']
            global_direct_relations  = []
            global_expand_relations = []
            local_direct_entities  = []
            local_expand_entities = []
    
            for sub_kb_info in kb_info:
                direct_entities, expand_entities, local_realtions, direct_relations, expand_relations, global_entity= self.get_graph_info(sub_kb_info, query)
                local_direct_entities += direct_entities
                local_expand_entities += expand_entities
                local_direct_entities += global_entity

                global_direct_relations += direct_relations
                global_expand_relations += expand_relations
                global_direct_relations += local_realtions



            direct_relations_processed, expand_relations_processed, _ = self.dedup_and_add_frequency(global_direct_relations, 
                                                                                                     global_expand_relations)
            direct_entities_processed,  expand_entities_processed,  _ = self.dedup_and_add_frequency(local_direct_entities, 
                                                                                                     local_expand_entities, 
                                                                                                     key = "entity_id")


            relations_reference = self.group_by_kb_id(direct_relations_processed + expand_relations_processed)
            entities_reference = self.group_by_kb_id(direct_entities_processed + expand_entities_processed)
            kb_id_graph_used =  set(relations_reference.keys()) | set(entities_reference.keys())

            kb_name_graph_used = {}
            for cur_kb in info_carrier['ragConfig']['knowledgeBaseConfigs']:
                if str(cur_kb['id']) in kb_id_graph_used:
                    kb_name_graph_used[str(cur_kb['id'])]  = cur_kb.get('name', None)

            time_rewrite=time.time()
            logger.info('time：改写用时: %s'%(time_rewrite-time_start))

            retriever_chunks=self.retrieve_fromkb(info_carrier['ragConfig'], query)
            time_retriever=time.time()
            logger.info('time：召回用时: %s'%(time_retriever-time_rewrite))
            logger.info('retriever_chunks的数量: %s'%(len(retriever_chunks['documents'])))
            clean_chunks, images_dict=self.post_process(retriever_chunks, remove_ocr = True)
            time_post=time.time()
            logger.info('time：后处理用时: %s'%(time_post-time_retriever))
            logger.info('clean_chunks的数量: %s'%(len(clean_chunks['documents'])))

            # response_chunks=[]
            # for i in range(len(clean_chunks['documents'])):
            #     content=clean_chunks['documents'][i].content
            #     if clean_chunks['documents'][i].meta.get('constructType')=='QUESTION':
            #         content=clean_chunks['documents'][i].meta['query_name']+":"+content+","+ \
            #                 clean_chunks['documents'][i].meta['answer_name']+":"+clean_chunks['documents'][i].meta[
            #                     'related_answer']+"."
            #         # logger.info('第: %s, 个content的内容是: %s' % (
            #         #         i, content))
            #     doc_url=clean_chunks['documents'][i].meta.get('doc_url',None)
            #     title=clean_chunks['documents'][i].meta.get('title', None)
            #     response_chunks.append({"content": content, "title": title, "url": doc_url})



            # q 入库这样写ok
            # 如果是qa 入库
            response_chunks = []
            for i in range(len(clean_chunks['documents'])):
                 content_question = clean_chunks['documents'][i].meta.get('question')
                 content_answer = clean_chunks['documents'][i].meta.get('related_answer')
                 content = clean_chunks['documents'][i].content
                 if clean_chunks['documents'][i].meta.get('constructType') == 'QUESTION':
                    content = clean_chunks['documents'][i].meta['query_name'] + ":" + content + "," + clean_chunks['documents'][i].meta['answer_name'] + ":" + clean_chunks['documents'][i].meta['related_answer'] + "."
                    # logger.info('第: %s, 个content的内容是: %s' % (
                    #         i, content))
                 doc_url=clean_chunks['documents'][i].meta.get('doc_url',None)
                 title=clean_chunks['documents'][i].meta.get('title', None)
                 response_chunks.append({"content": content, "content_answer": content_answer,"content_question": content_question, "title": title, "url" : doc_url})

            if self.enablererank:

                # rerank：按当前 query 对召回 chunk 打分并重排（不改原 response_chunks）
                reranked_chunks = copy.deepcopy(response_chunks)
                try:
                    rerank_docs = [item.get("content", "") for item in reranked_chunks]
                    rerank_scores = aizt_rerank(query, rerank_docs, model_type = self.rerank_model)
                    if len(rerank_scores) == len(response_chunks):
                        scored_chunks = []
                        for i, item in enumerate(response_chunks):
                            item_copy = dict(item)
                            item_copy["rerank_score"] = rerank_scores[i]
                            scored_chunks.append(item_copy)
                        reranked_chunks = [
                            item for item in scored_chunks
                            if float(item.get("rerank_score", float("-inf"))) > float(self.rerank_sim_threshold)
                        ]
                        reranked_chunks.sort(key=lambda x: x.get("rerank_score", float("-inf")), reverse=True)
                    else:
                        logger.warning(
                            "rerank 返回数量(%s)与 chunks 数量(%s)不一致，跳过重排",
                            len(rerank_scores),
                            len(response_chunks),
                        )
                except Exception as e:
                    logger.warning("rerank 失败，保持原召回顺序: %s", str(e))

                response_chunks = reranked_chunks

            graphs = []
            cur_env=os.environ.get("env")
            if cur_env == "test":
                kb_path = "https://ai-test.yangqianguan.com/hub/rag/knowledge/detail/"
            elif cur_env == "dev":
                kb_path = ""
            else:
                kb_path = "https://ai.fintopia.tech/hub/rag/knowledge/detail/"
                # 生产环境
      
            for sub_kb_id in kb_id_graph_used:
                sub_graph_indo = {}
                sub_graph_indo['title'] = f"{kb_name_graph_used[str(sub_kb_id)]}-知识图谱"
                sub_graph_indo['url']  = os.path.join(kb_path, sub_kb_id)
                entities = []
                relationships = []
                if sub_kb_id in entities_reference.keys():
                    sub_kb_id = str(sub_kb_id)
                    for i in range(len(entities_reference[sub_kb_id])):

                        cur_entity = "name: " + entities_reference[sub_kb_id][i]['name'] + ", description: " +  entities_reference[sub_kb_id][i]['description'] 
                        entities.append(cur_entity)
                if sub_kb_id in relations_reference:
                    sub_kb_id = str(sub_kb_id)
                    for i in range(len(relations_reference[sub_kb_id])):
                        tgt_id = relations_reference[sub_kb_id][i].get("tgt_id", " ")
                        src_id = relations_reference[sub_kb_id][i].get("src_id", " ")
                        relation_type = relations_reference[sub_kb_id][i].get("relation_type", " ")
                        description = relations_reference[sub_kb_id][i].get("description", " ")
                        cur_relations = f"{src_id} —[{relation_type}]→ {tgt_id}"
                        if description:
                            cur_relations += f"说明：{description}"
                        relationships.append(cur_relations)
                sub_graph_indo['entities'] = entities
                sub_graph_indo['relationships'] = relationships

                graphs.append(sub_graph_indo)


            related_images={}
            # 根据images_dict 重新拿取

            for idx, value in images_dict.items():
                # 通过索引获取元组中的两个值
                doc_id=value[0]
                img_id=value[1]
                kb_name=value[2]
                cur_env=os.environ.get("env")
                kb_used=self.kb_info_client.get(str(kb_name))
                obj_name=cur_env+str(kb_name)+"_kb_"+kb_used+doc_id+img_id
                # b64_str = self.oss_processor.download_file(obj_name)
                related_images["img"+str(idx)]=obj_name
            # 配合image
            time_image=time.time()
            logger.info('time：图片处理用时: %s'%(time_image-time_post))

            response_info={
                "outputs": {
                    "chunks": response_chunks,
                    "images": related_images,
                    "graphs": graphs
                }
            }
            return response_info
        except Exception as e:
            print(f"\n{traceback.format_exc()}\n")  # 测试用
            info_carrier['error']=f"{self.__class__.__name__} error: {str(e)}"
            raise Exception(f"{self.__class__.__name__} error: {str(e)} ")

    def process_unstream(self, info_carrier, model_chat=None):
        try:
            time_start=time.time()
            self.model=model_chat
            self.rerank_info = info_carrier['ragConfig'].get("rerankConfig", {})
            self.enablererank = self.rerank_info.get("enableRerank", False)
            self.rerank_model = self.rerank_info.get("rerankerModel", "Qwen/Qwen3-Reranker-8B")
            self.rerank_sim_threshold = self.rerank_info.get("similarityThreshold", 0)
            if info_carrier['ragConfig']['llmReplyConfig']['enableMultiTurn'] and len(info_carrier['inputs'])>1:
                query, _, _=self.query_rewrite(info_carrier['inputs'])
                # 检查是否是“我刚才问什么”这类问题
                if query.startswith("你上一轮问的是："):
                    return {
                        "content": query,
                        "images": {},
                        "references": [],
                        "finish": True
                    }
            else:
                query=info_carrier['inputs'][-1]['content']



            # 现在一定是会去找下图，至于后续要不要加开关可以再看看
            # 这里要判断一下是否需要使用graph

            # 在这里处理库的信息好一点
            kb_info = info_carrier['ragConfig']['knowledgeBaseConfigs']
            global_direct_relations  = []
            global_expand_relations = []
            local_direct_entities  = []
            local_expand_entities = []
            # local_realtions_sum = []
            # global_entity_sum = []


            for sub_kb_info in kb_info:
                direct_entities, expand_entities, local_realtions, direct_relations, expand_relations, global_entity= self.get_graph_info(sub_kb_info, query)
                local_direct_entities += direct_entities
                local_expand_entities += expand_entities
                local_direct_entities += global_entity

                global_direct_relations += direct_relations
                global_expand_relations += expand_relations
                global_direct_relations += local_realtions



            direct_relations_processed, expand_relations_processed, _ = self.dedup_and_add_frequency(global_direct_relations, 
                                                                                                     global_expand_relations)
            direct_entities_processed,  expand_entities_processed,  _ = self.dedup_and_add_frequency(local_direct_entities, 
                                                                                                     local_expand_entities, 
                                                                                                     key = "entity_id")

            graph_kb_reference = self.collect_unique_kb_ids([direct_relations_processed,
                                                            expand_relations_processed,
                                                            direct_entities_processed,
                                                            expand_entities_processed])
            
            kb_name_graph_used = {}
            for cur_kb in info_carrier['ragConfig']['knowledgeBaseConfigs']:
                if str(cur_kb['id']) in graph_kb_reference:
                    kb_name_graph_used[str(cur_kb['id'])]  = cur_kb.get('name', None)


            # graph_kb_reference = self.collect_unique_kb_ids(direct_relations_processed + expand_relations_processed + direct_entities_processed + expand_entities_processed)
            time_rewrite=time.time()
            logger.info('time：改写用时: %s'%(time_rewrite-time_start))


            retriever_chunks=self.retrieve_fromkb(info_carrier['ragConfig'], query)
            time_retriever=time.time()
            logger.info('time：召回用时: %s'%(time_retriever-time_rewrite))
            clean_chunks, images_dict=self.post_process(retriever_chunks)

            if self.enablererank:
                # rerank：clean_chunks 结构为 {"documents": [...]}
                reranked_chunks = copy.deepcopy(clean_chunks)
                try:
                    rerank_source_docs = reranked_chunks.get("documents", [])
                    rerank_docs = []
                    for doc in rerank_source_docs:
                        content = getattr(doc, "content", "")
                        meta = getattr(doc, "meta", {}) or {}
                        if meta.get("constructType") == "QUESTION":
                            content = (
                                meta.get("query_name", "")
                                + ":"
                                + content
                                + ","
                                + meta.get("answer_name", "")
                                + ":"
                                + meta.get("related_answer", "")
                                + "."
                            )
                        rerank_docs.append(content)
                    rerank_scores = aizt_rerank(query, rerank_docs, model_type = self.rerank_model)
                    if len(rerank_scores) == len(rerank_source_docs):
                        scored_docs = []
                        threshold = float(self.rerank_sim_threshold)
                        for i, doc in enumerate(rerank_source_docs):
                            rerank_score = float(rerank_scores[i])
                            # if score > threshold:
                            if rerank_score > threshold:
                                # 显式补充 rerank_score，便于下游直接读取
                                doc_meta = getattr(doc, "meta", None)
                                if isinstance(doc_meta, dict):
                                    doc_meta["rerank_score"] = rerank_score
                                setattr(doc, "rerank_score", rerank_score)
                                scored_docs.append((doc, rerank_score))
                        scored_docs.sort(key=lambda x: x[1], reverse=True)
                        reranked_chunks["documents"] = [doc for doc, _ in scored_docs]
                    else:
                        logger.warning(
                            "rerank 返回数量(%s)与 chunks 数量(%s)不一致，跳过重排",
                            len(rerank_scores),
                            len(rerank_source_docs),
                        )
                except Exception as e:
                    logger.warning("rerank 失败，保持原召回顺序: %s", str(e))
                clean_chunks = reranked_chunks


            time_post=time.time()
            logger.info('time：后处理用时: %s'%(time_post-time_retriever))
            if info_carrier['ragConfig']['llmReplyConfig']['enableMultiTurn'] and len(info_carrier['inputs'])>1:
                history_content=info_carrier['inputs'][-11:] # 取最后五轮qa
                history_chat=""
                for sub_content in history_content[:-1]:
                    history_chat+=sub_content['role']+':'
                    history_chat+=sub_content['content']+'. '
                    history_chat+="\n"
            else:
                history_chat=""
            
            model_inputs=self.get_model_input(info_carrier['inputs'][-1]['content'], 
                                              clean_chunks, 
                                              history_chat, 
                                              direct_relations_processed, 
                                              expand_relations_processed,
                                              direct_entities_processed,
                                              expand_entities_processed)
            system_content="You are a helpful assistant. Your task is to answer users' questions based on provided reference materials."
            # model_inputs = "请输出500个字的小作文"
            chat_messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": model_inputs}
            ]
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False
                },
            }
            client=OpenAI(
                api_key="EMPTY",
                base_url=self.chat_url
            )
            response_text=""
            stream_resp=client.chat.completions.create(
                model="Qwen/Qwen3-14B",
                messages=chat_messages,
                temperature=0,
                presence_penalty=0.5,  # 控制新主题的出现（-2.0 到 2.0）
                frequency_penalty=1,  # 控制重复内容的出现（-2.0 到 2.0）
                extra_body=extra_body,
                response_format={"type": "json_object"})
            response_text=stream_resp.choices[0].message.content
            response_dict=json.loads(response_text)
            pure_answer=response_dict['answer']
            logger.info("=== 模型生成的 answer 原始内容 ===")
            logger.info(repr(pure_answer))

            if not pure_answer.endswith("\n"):
                pure_answer+="\n"

            time_response=time.time()
            logger.info('time：生成用时: %s'%(time_response-time_post))

            references_list=[]
            if "暂时无法回答您的问题" not in pure_answer:
                    
                try:
                    doc_list=[]
                    for chunk_id in response_dict['reference_idx']:
                        if chunk_id==0 or chunk_id>len(clean_chunks['documents']):  # 不合法
                            continue
                        if clean_chunks['documents'][chunk_id-1].meta['doc_url'] in doc_list:  # 已保存
                            continue
                        doc_list.append(clean_chunks['documents'][chunk_id-1].meta['doc_url'])
                        references_list.append({"title": clean_chunks['documents'][chunk_id-1].meta['title'],
                                                "url": clean_chunks['documents'][chunk_id-1].meta['doc_url']})

                    cur_env=os.environ.get("env")
                    if cur_env == "test":
                        kb_path = "https://ai-test.yangqianguan.com/hub/rag/knowledge/detail/"
                    elif cur_env == "dev":
                        kb_path = ""
                    else:
                        kb_path = "https://ai.fintopia.tech/hub/rag/knowledge/detail/"
                        # 生产环境
                    if graph_kb_reference:
                        for sub_graph_kb_id in range(len(graph_kb_reference)):
                            sub_kb_id = graph_kb_reference[sub_graph_kb_id]
                            # kb_name_graph_used
                            references_list.append({"title": f"{kb_name_graph_used[str(sub_kb_id)]}-知识图谱",
                                                    "url": os.path.join(kb_path, sub_kb_id)})

                except:
                    pass
    
            images_return={}
            cur_env=os.environ.get("env")
            for img_small_key in images_dict.keys():
                doc_id, img_id, kb_name=images_dict.get(img_small_key)
                kb_used=self.kb_info_client.get(str(kb_name))
                obj_name=cur_env+str(kb_name)+"_kb_"+kb_used+doc_id+img_id
                images_return["img"+str(img_small_key)]=obj_name
            response_info={
                "content": pure_answer,
                "images": images_return,
                "references": references_list,
                "finish": True
            }
            return response_info
        except Exception as e:
            print(f"\n{traceback.format_exc()}\n")  # 测试用
            info_carrier['error'] = f"{self.__class__.__name__} error: {str(e)}"
            raise Exception(f"{self.__class__.__name__} error: {str(e)} ")

    def process_stream(self, info_carrier, model_chat=None, chat_tokenizer=None, text_embedder=None):
        try:
            time_start=time.time()
            self.model=model_chat
            self.tokenizer=chat_tokenizer
            self.text_embedder=text_embedder
            self.rerank_info = info_carrier['ragConfig'].get("rerankConfig", {})
            self.enablererank = self.rerank_info.get("enableRerank", False)
            self.rerank_model = self.rerank_info.get("rerankerModel", "Qwen/Qwen3-Reranker-8B")
            self.rerank_sim_threshold = self.rerank_info.get("similarityThreshold", 0)
            try:
                if info_carrier['ragConfig']['llmReplyConfig']['enableMultiTurn'] and len(info_carrier['inputs'])>1:
                    query, rewrite_pe, rewrite_sys_pe=self.query_rewrite(info_carrier['inputs'])
                else:
                    query=info_carrier['inputs'][-1]['content']
                    rewrite_pe = ""
                    rewrite_sys_pe = ""



                kb_info = info_carrier['ragConfig']['knowledgeBaseConfigs']
                global_direct_relations  = []
                global_expand_relations = []
                local_direct_entities  = []
                local_expand_entities = []
                # local_realtions_sum = []
                # global_entity_sum = []


                for sub_kb_info in kb_info:
                    direct_entities, expand_entities, local_realtions, direct_relations, expand_relations, global_entity= self.get_graph_info(sub_kb_info, query)
                    local_direct_entities += direct_entities
                    local_expand_entities += expand_entities
                    local_direct_entities += global_entity

                    global_direct_relations += direct_relations
                    global_expand_relations += expand_relations
                    global_direct_relations += local_realtions

                direct_relations_processed, expand_relations_processed, _ = self.dedup_and_add_frequency(global_direct_relations, 
                                                                                                        global_expand_relations)
                direct_entities_processed,  expand_entities_processed,  _ = self.dedup_and_add_frequency(local_direct_entities, 
                                                                                                     local_expand_entities, 
                                                                                                     key = "entity_id")
                # 在这里过滤每一个变量的id

                graph_kb_reference = self.collect_unique_kb_ids([direct_relations_processed,
                                                                expand_relations_processed,
                                                                direct_entities_processed,
                                                                expand_entities_processed])

                kb_name_graph_used = {}
                for cur_kb in info_carrier['ragConfig']['knowledgeBaseConfigs']:
                    if str(cur_kb['id']) in graph_kb_reference:
                        kb_name_graph_used[str(cur_kb['id'])]  = cur_kb.get('name', None)

                time_rewrite=time.time()
                logger.info('time：改写用时: %s'%(time_rewrite-time_start))
                time_quert_rewrite_cost = time_rewrite-time_start
                time_quert_rewrite_cost = round(time_quert_rewrite_cost, 3)
                if info_carrier.get('debug', False): # 开启了debug模式，需要返回一些日志信息
                    if not info_carrier.get('ragConfig', False).get('llmReplyConfig', False).get('enableMultiTurn', False):
                        startResult = json.dumps({"rewrite_query": False, "reason":"未开启多轮对话"}, indent=4, ensure_ascii=False)
                    elif len(info_carrier['inputs']) == 1:
                        startResult = json.dumps({"rewrite_query": False, "reason":"当前为首轮，没有历史信息"}, indent=4, ensure_ascii=False)
                    else:
                        # used_history = info_carrier['inputs'][-11:]
                        startResult = {}
                        startResult['rewrite_query'] = True
                        startResult['system_prompt'] = rewrite_sys_pe
                        startResult['rewrite_prompt'] = rewrite_pe
                        startResult = json.dumps(startResult, indent=4, ensure_ascii=False)
                    endResult = json.dumps({"rewrite_query":query}, indent=4, ensure_ascii=False)
                    query_rewrite_body = self.yield_think_info("query改写", time_quert_rewrite_cost, "success",startResult, endResult)
                    yield query_rewrite_body
   
            except Exception as e:
                # 获取异常的详细信息
                error_message = traceback.format_exc()
                if info_carrier.get('debug', False): # 开启了debug模式，需要返回一些日志信息
                    startResult = json.dumps({}, indent=4, ensure_ascii=False)
                    endResult = json.dumps({"error": error_message}, indent=4, ensure_ascii=False)
                    time_quert_rewrite_cost = 0
                    # 这是出错了
                    query_rewrite_body = self.yield_think_info("query改写", time_quert_rewrite_cost, "failure",startResult, endResult)
                    yield query_rewrite_body
                raise e
            try:
                time_retriever=time.time()
                retriever_chunks=self.retrieve_fromkb(info_carrier['ragConfig'], query)
                
                logger.info('time：召回用时: %s'%(time_retriever-time_rewrite))
                clean_chunks, images_dict=self.post_process(retriever_chunks)








                for tempchunkid in range(len(clean_chunks['documents'])):
                    content=clean_chunks['documents'][tempchunkid].content
                    if clean_chunks['documents'][tempchunkid].meta.get('constructType')=='QUESTION':
                        content=clean_chunks['documents'][tempchunkid].meta['query_name']+":"+content+","+ \
                                clean_chunks['documents'][tempchunkid].meta['answer_name']+":"+clean_chunks['documents'][tempchunkid].meta[
                                    'related_answer']+"."
                        print('第: %s, 个content的内容是: %s' % (
                                tempchunkid, content))
                time_post=time.time()
                time_retrive_cost =time_post - time_retriever
                logger.info('time：后处理用时: %s'%(time_post-time_retriever))
                time_retrive_cost = round(time_retrive_cost, 3)

                if info_carrier.get('debug', False): # 开启了debug模式，需要返回一些日志信息
                    kb_info = info_carrier['ragConfig']['knowledgeBaseConfigs']
                    used_kb_id = []
                    for sub_kb in kb_info:
                        # sub_kb_id = kb_info[sub_kb]['id']
                        used_kb_id.append(sub_kb['id'])
                    startResult = json.dumps({"query":query, "retriever_kb":used_kb_id}, indent=4, ensure_ascii=False)

                    think_retriever_chunk = {}
                    for tempchunkid in range(len(clean_chunks['documents'])):
                        content=clean_chunks['documents'][tempchunkid].content
                        if clean_chunks['documents'][tempchunkid].meta.get('constructType')=='QUESTION':
                            content=clean_chunks['documents'][tempchunkid].meta['query_name']+":"+content+","+ \
                                    clean_chunks['documents'][tempchunkid].meta['answer_name']+":"+clean_chunks['documents'][tempchunkid].meta[
                                        'related_answer']+"."
                            print('第: %s, 个content的内容是: %s' % (
                                    tempchunkid, content))
                        tempchunk_info = {
                            "content":content,
                            "title":clean_chunks['documents'][tempchunkid].meta.get('title', None),
                            "url":clean_chunks['documents'][tempchunkid].meta.get('doc_url', None),
                            "kb_id":clean_chunks['documents'][tempchunkid].meta.get('source_kb', None),
                            "score":clean_chunks['documents'][tempchunkid].score

                        }

                        content_question = clean_chunks['documents'][tempchunkid].meta.get("question",None)
                        if content_question:
                            tempchunk_info['content_question'] = content_question
                        content_answer = clean_chunks['documents'][tempchunkid].meta.get("related_answer",None)
                        if content_answer:
                            tempchunk_info['content_answer'] = content_answer

                        new_key = f"chunk_{tempchunkid}"
                        think_retriever_chunk[new_key] = tempchunk_info
                    

                    entity_direct = self.entities_to_dict_min(direct_entities_processed)
                    relation_direct = self.relations_to_dict_min(direct_relations_processed)
                    entity_expand = self.entities_to_dict_min(expand_entities_processed)
                    relation_expand = self.relations_to_dict_min(expand_relations_processed)


                    if entity_direct:
                        think_retriever_chunk['entity_direct'] = entity_direct
                    if relation_direct:
                        think_retriever_chunk['relation_direct'] = relation_direct
                    if entity_expand:
                        think_retriever_chunk['entity_expand'] = entity_expand
                    if relation_expand:
                        think_retriever_chunk['relation_expand'] = relation_expand 
                    endResult = json.dumps({"retriever_res":think_retriever_chunk}, indent=4, ensure_ascii=False)
                    retriever_body = self.yield_think_info("召回模块", time_retrive_cost, "success",startResult, endResult)

                    yield retriever_body
                # rerank_docs = ['iPhone 重置密码步骤：连接电脑，进入恢复模式，选择恢复，完成后重新激活。']
                # rerank_scores = aizt_rerank(query, rerank_docs, model_type = self.rerank_model)
                if self.enablererank:
                    # rerank：clean_chunks 结构为 {"documents": [...]}
                    reranked_chunks = copy.deepcopy(clean_chunks)
                    try:
                        rerank_source_docs = reranked_chunks.get("documents", [])
                        rerank_docs = []
                        for doc in rerank_source_docs:
                            content = getattr(doc, "content", "")
                            meta = getattr(doc, "meta", {}) or {}
                            if meta.get("constructType") == "QUESTION":
                                content = (
                                    meta.get("query_name", "")
                                    + ":"
                                    + content
                                    + ","
                                    + meta.get("answer_name", "")
                                    + ":"
                                    + meta.get("related_answer", "")
                                    + "."
                                )
                            rerank_docs.append(content)
                        rerank_scores = aizt_rerank(query, rerank_docs, model_type = self.rerank_model)
                        print("下面两个的长度")
                        print(rerank_scores)
                        print(rerank_source_docs)
                        print(len(rerank_scores))
                        print(len(rerank_source_docs))
                        if len(rerank_scores) == len(rerank_source_docs):
                            scored_docs = []
                            threshold = float(self.rerank_sim_threshold)
                            for i, doc in enumerate(rerank_source_docs):
                                rerank_score = float(rerank_scores[i])
                                # if score > threshold:
                                if rerank_score > threshold:
                                    # 显式补充 rerank_score，便于下游直接读取
                                    doc_meta = getattr(doc, "meta", None)
                                    if isinstance(doc_meta, dict):
                                        doc_meta["rerank_score"] = rerank_score
                                    setattr(doc, "rerank_score", rerank_score)
                                    scored_docs.append((doc, rerank_score))
                            scored_docs.sort(key=lambda x: x[1], reverse=True)
                            reranked_chunks["documents"] = [doc for doc, _ in scored_docs]
                        else:
                            logger.warning(
                                "rerank 返回数量(%s)与 chunks 数量(%s)不一致，跳过重排",
                                len(rerank_scores),
                                len(rerank_source_docs),
                            )
                    except Exception as e:
                        logger.warning("rerank 失败，保持原召回顺序: %s", str(e))
                    clean_chunks = reranked_chunks

                    if info_carrier.get('debug', False): # 开启了debug模式，需要返回一些日志信息
                        kb_info = info_carrier['ragConfig']['knowledgeBaseConfigs']
                        used_kb_id = []
                        for sub_kb in kb_info:
                            used_kb_id.append(sub_kb['id'])
                        startResult = json.dumps({"query":query, "retriever_kb":used_kb_id}, indent=4, ensure_ascii=False)

                        think_retriever_chunk = {}
                        for tempchunkid in range(len(clean_chunks['documents'])):
                            content=clean_chunks['documents'][tempchunkid].content
                            if clean_chunks['documents'][tempchunkid].meta.get('constructType')=='QUESTION':
                                content=clean_chunks['documents'][tempchunkid].meta['query_name']+":"+content+","+ \
                                        clean_chunks['documents'][tempchunkid].meta['answer_name']+":"+clean_chunks['documents'][tempchunkid].meta[
                                            'related_answer']+"."

                                print('第: %s, 个content的内容是: %s' % (
                                        tempchunkid, content))
                            tempchunk_info = {
                                "content":content,
                                "title":clean_chunks['documents'][tempchunkid].meta.get('title', None),
                                "url":clean_chunks['documents'][tempchunkid].meta.get('doc_url', None),
                                "kb_id":clean_chunks['documents'][tempchunkid].meta.get('source_kb', None),
                                "score":clean_chunks['documents'][tempchunkid].get("score", 0),
                                "rerank_score":clean_chunks['documents'][tempchunkid].get(rerank_score, 0),

                            }

                            content_question = clean_chunks['documents'][tempchunkid].meta.get("question",None)
                            if content_question:
                                tempchunk_info['content_question'] = content_question
                            content_answer = clean_chunks['documents'][tempchunkid].meta.get("related_answer",None)
                            if content_answer:
                                tempchunk_info['content_answer'] = content_answer

                            new_key = f"chunk_{tempchunkid}"
                            think_retriever_chunk[new_key] = tempchunk_info
                        
                        endResult = json.dumps({"rerank_res":think_retriever_chunk}, indent=4, ensure_ascii=False)
                        retriever_body = self.yield_think_info("ReRank模块", time_retrive_cost, "success",startResult, endResult)

                        yield retriever_body


            except Exception as e:
                # 获取异常的详细信息
                error_message = traceback.format_exc()
                if info_carrier.get('debug', False): # 开启了debug模式，需要返回一些日志信息
                    startResult = json.dumps({}, indent=4, ensure_ascii=False)
                    endResult = json.dumps({"error": error_message}, indent=4, ensure_ascii=False)
                    time_quert_rewrite_cost = 0
                    # 这是出错了
                    retriever_body = self.yield_think_info("召回模块", time_quert_rewrite_cost, "failure",startResult, endResult)
                    yield retriever_body
                raise e



            try:

                if info_carrier['ragConfig']['llmReplyConfig']['enableMultiTurn'] and len(info_carrier['inputs'])>1:
                    history_content=info_carrier['inputs'][-11:]
                    history_chat=""
                    for sub_content in history_content[:-1]:
                        history_chat+=sub_content['role']+':'
                        history_chat+=sub_content['content']+'. '
                        history_chat+="\n"
                else:
                    history_chat=""




                time_llm_start = time.time()
                # language_used = self.get_output_language(info_carrier['inputs'][-1]['content'], clean_chunks)
                model_inputs=self.get_model_input(info_carrier['inputs'][-1]['content'], 
                                                  clean_chunks, 
                                                  history_chat, 
                                                  direct_relations_processed, 
                                                  expand_relations_processed,
                                                  direct_entities_processed,
                                                  expand_entities_processed)

                system_content="You are a helpful assistant. Your task is to answer users' questions based on provided reference materials."
                # model_inputs = "请输出500个字的小作文"
                chat_messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": model_inputs}
                ]
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    },
                }
                client=OpenAI(
                    api_key="EMPTY",
                    base_url=self.chat_url
                )
                response_text=""

                stream_resp=client.chat.completions.create(
                    model="Qwen/Qwen3-14B",
                    messages=chat_messages,
                    temperature=0,
                    stream=True,
                    presence_penalty=0.5,  # 控制新主题的出现（-2.0 到 2.0）
                    frequency_penalty=1,  # 控制重复内容的出现（-2.0 到 2.0）
                    extra_body=extra_body,
                    response_format={"type": "json_object"}
                )


                time_llm_start_stream = time.time()
                time_pre_process = time_llm_start_stream - time_llm_start
                time_pre_process = round(time_pre_process, 3)



                if info_carrier.get('debug', False): # 开启了debug模式，需要返回一些日志信息

                    startResult = json.dumps({"history_chat":history_chat, 
                                              "query":info_carrier['inputs'][-1]['content']}, indent=4, ensure_ascii=False)
                    endResult = json.dumps({"model_inputs":model_inputs, "system_pe":system_content}, indent=4, ensure_ascii=False)
                    response_body = self.yield_think_info("response前处理模块", time_pre_process, "success",startResult, endResult)
                    yield response_body
            except Exception as e:
                error_message = traceback.format_exc()
                if info_carrier.get('debug', False): # 开启了debug模式，需要返回一些日志信息
                    startResult = json.dumps({}, indent=4, ensure_ascii=False)
                    endResult = json.dumps({"error": error_message}, indent=4, ensure_ascii=False)
                    time_quert_rewrite_cost = 0
                    # 这是出错了
                    retriever_body = self.yield_think_info("response前处理模块", time_quert_rewrite_cost, "failure",startResult, endResult)
                    yield retriever_body
                raise e


            # 在这给一个yield
            response_text=""
            answer_so_far=""
            pre_str=""
            is_first=True
            for event in stream_resp:
                if not event.choices:
                    continue
                delta=getattr(event.choices[0].delta, "content", None)
                if not delta:
                    continue

                response_text+=delta

                match = re.search(
                    r'"answer"\s*:\s*"(.*)',   # 捕获从 answer 开始的全部内容
                    response_text,
                    re.S
                )

                if not match:
                    answer = None
                    continue

                after_answer = match.group(1)   # 包含正文 + 可能的 ",\n "reference_idx" 之类

                # 终止标记：JSON 里真正的字段是 "reference_idx"
                sentinel_str = '"reference_idx"'
                # 完整结构：",\s*"reference_idx"
                sentinel_re = re.compile(r'"\s*,\s*' + re.escape(sentinel_str))
            
                text = after_answer

                # 2. 先看有没有出现完整的 , "reference_idx" 结构
                m_full = sentinel_re.search(text)
                if m_full:
                    # 从这个结构开始都不要了（包括前面的那个引号和逗号）
                    stable = text[:m_full.start()]
                else:
                    # 3. 还没出现完整结构：需要“保护”尾部可能是 "reference_idx" 的前缀
                    max_k = 0
                    for k in range(1, len(sentinel_str)):
                        if text.endswith(sentinel_str[:k]):
                            max_k = k

                    if max_k > 0:
                        # 说明当前结尾有一段像 `"refere` / `"reference_` 这种前缀；
                        # 先扣住这段，不把它算进答复里，避免半截 reference 跑出去。
                        stable = text[:-max_k]
                    else:
                        # 尾部完全不像 sentinel 前缀，全当正常内容输出
                        stable = text

                raw_answer = stable.rstrip()
                if pre_str:  # 不是第一次
                    add_txt=raw_answer.split(pre_str)[-1]  # 单词增加的在这里
                else:
                    add_txt=raw_answer  # 如果没有prestr，也就是第一次，所以直接输出raw_answer


                try:
                    # 用 JSON 反序列化，自动处理 \\n -> \n
                    current_answer=json.loads(f'"{raw_answer}"')
                    add_txt=json.loads(f'"{add_txt}"')
                    pre_str=raw_answer

                except json.JSONDecodeError:
                    continue



                if len(current_answer)>len(answer_so_far):  # 这个用于判断是否输出无用信息
                    answer_so_far=current_answer
                    if is_first:
                        images_first={}
                        cur_env=os.environ.get("env")

                        for img_small_key in images_dict.keys():
                            doc_id, img_id, kb_name=images_dict.get(img_small_key)
                            kb_used=self.kb_info_client.get(str(kb_name))
                            obj_name=cur_env+str(kb_name)+"_kb_"+kb_used+doc_id+img_id
                            images_first["img"+str(img_small_key)]=obj_name
                        yield {"type": "content","content": add_txt, "finish": False, "images": images_first}
                    else:
                        yield {"type": "content", "content": add_txt, "finish": False}
                    is_first=False
            # print(response_text)
            response_dict=json.loads(response_text)
            pure_answer=response_dict['answer']


            if "暂时无法回答您的问题" in pure_answer:
                references_list=[]
            else:
                time_response=time.time()
                logger.info('time：生成用时: %s'%(time_response-time_post))
                references_list=[]
                try:
                    doc_list=[]
                    for chunk_id in response_dict['reference_idx']:
                        if chunk_id==0 or chunk_id>len(clean_chunks['documents']):  # 不合法
                            continue
                        if clean_chunks['documents'][chunk_id-1].meta['doc_url'] in doc_list:  # 已保存
                            continue
                        doc_list.append(clean_chunks['documents'][chunk_id-1].meta['doc_url'])
                        references_list.append({"title": clean_chunks['documents'][chunk_id-1].meta['title'],
                                                "url": clean_chunks['documents'][chunk_id-1].meta['doc_url']})
                        

                    cur_env=os.environ.get("env")
                    if cur_env == "test":
                        kb_path = "https://ai-test.yangqianguan.com/hub/rag/knowledge/detail/"
                    elif cur_env == "dev":
                        kb_path = ""
                    else:
                        kb_path = "https://ai.fintopia.tech/hub/rag/knowledge/detail/"
                        # 生产环境


                    if graph_kb_reference:
                        for sub_graph_kb_id in range(len(graph_kb_reference)):
                            sub_kb_id = graph_kb_reference[sub_graph_kb_id]
                            references_list.append({"title": f"{kb_name_graph_used[str(sub_kb_id)]}-知识图谱",
                                                    "url": os.path.join(kb_path, sub_kb_id)})
                except:
                    pass
                    # 防止参考链接出问题的手段，但其实应该不会出问题



            response_info={
                "type": "content",
                "content": "",
                "references": references_list,
                "finish": True
            }
            yield response_info

        except Exception as e:
            print(f"\n{traceback.format_exc()}\n")  # 测试用
            info_carrier['error']=f"{self.__class__.__name__} error: {str(e)}"
            raise Exception(f"{self.__class__.__name__} error: {str(e)} ")



    def group_by_kb_id(self, *lists):
        grouped = defaultdict(list)

        for lst in lists:
            for item in lst:
                if isinstance(item, dict) and "kb_id" in item:
                    grouped[item["kb_id"]].append(item)

        # 按你的要求：返回一个“列表”，每个元素是 {kb_id: [dicts]}
        return dict(grouped)


    def yield_think_info(self, step_name, timeCost, resultStatus, startResult, endResult):
        query_rewrite_body = {
            # "code": 0,
            # "message": "",
            # "data": {
                "type": "think",
                "thinkResult": { 
                    "stepName":step_name,
                    "timeCost":timeCost,
                    "resultStatus":resultStatus,
                    "startResult":startResult,
                    "endResult":endResult
                },
                "finish": False,
                "content":"",
                "references":[],
                "images":{}
            }
        # }

        return  query_rewrite_body




    def entities_to_dict_min(self, entities):
        seen = set()
        blocks = {}

        for e in entities:
            if not isinstance(e, dict):
                continue
            name = (e.get("name") or "").strip()
            desc = (e.get("description") or "").strip()
            if not name or not desc:
                continue
            if name in seen:
                continue
            seen.add(name)

            blocks[name] = desc
        return blocks



    def relations_to_dict_min(self, relations):
        blocks = {}

        for r in relations:
            if not isinstance(r, dict):
                continue

            src = (r.get("src_id") or "").strip()
            rel = (r.get("relation_type") or "").strip()
            tgt = (r.get("tgt_id") or "").strip()
            desc = (r.get("description") or "").strip()

            # 至少要有 src / rel / tgt 之一，描述可选
            if not (src and rel and tgt):
                continue

            block_lines = []
            block_lines.append(f"{src} —[{rel}]→ {tgt}")
            if desc:
                block_lines.append(f"说明：{desc}")

            if not desc:
                continue
            blocks[f"{src} —[{rel}]→ {tgt}"] = desc

        return blocks


    def entities_to_pe_min(self, entities) -> str:
        seen = set()
        blocks = []

        for e in entities:
            if not isinstance(e, dict):
                continue
            name = (e.get("name") or "").strip()
            desc = (e.get("description") or "").strip()
            if not name or not desc:
                continue
            if name in seen:
                continue
            seen.add(name)

            blocks.append(f"名称：{name}\n描述：{desc}")
        return "\n\n".join(blocks)

    def relations_to_pe_min(self, relations) -> str:
        blocks = []

        for r in relations:
            if not isinstance(r, dict):
                continue

            src = (r.get("src_id") or "").strip()
            rel = (r.get("relation_type") or "").strip()
            tgt = (r.get("tgt_id") or "").strip()
            desc = (r.get("description") or "").strip()

            # 至少要有 src / rel / tgt 之一，描述可选
            if not (src and rel and tgt):
                continue

            block_lines = []
            block_lines.append(f"{src} —[{rel}]→ {tgt}")
            if desc:
                block_lines.append(f"说明：{desc}")

            blocks.append("\n".join(block_lines))

        return "\n\n".join(blocks)


    def collect_unique_kb_ids(self, lists):
        kb_ids = set()
        for lst in lists:
            for item in lst:
                if isinstance(item, dict) and "kb_id" in item:
                    kb_ids.add(item["kb_id"])
        return list(kb_ids)


    def get_model_input(self, 
                        query, 
                        clean_chunks, 
                        history_chat, 
                        direct_relations_processed = [], 
                        expand_relations_processed = [], 
                        direct_entities_processed = [], 
                        expand_entities_processed = []):

        def classify_question(q):
            # 预清洗：全角转半角、去空格、小写化
            q=q.replace("？", "?").replace("，", ",").replace("：", ":").replace("。", ".")
            q=q.strip().lower()

            text_keywords=["怎么", "如何", "步骤", "操作", "流程", "说明", "接入", "同步", "开启",
                           "知识库", "飞书", "文档", "节奏", "更新", "注意事项", "指南"]
            struct_keywords=["表格", "代码", "字段", "配置项", "参数", "json", "yaml", "列名", "格式"]

            # 优先说明性问题
            # if any(re.search(k, q) for k in text_keywords):
            #     return "text"
            #
            # # 含飞书、知识库类字样也当 text
            # if any(k in q for k in ["飞书", "知识库", "同步", "接入", "步骤", "配置"]):
            #     return "text"

            # 仅明确命中结构关键词时才返回 struct
            if any(re.search(k, q) for k in struct_keywords):
                return "struct"
            return "text"
      
        content_adjust_language = []
        for i, chunk in enumerate(clean_chunks['documents']):
            content=chunk.content
            content_adjust_language.append(f"[{i+1}] {content[:600]}")  # 600 chars cap
            if len(content_adjust_language) >= 8:  # top 8 is usually enough for language
                break
        # content_adjust_language
        content_language = "\n".join(content_adjust_language) if content_adjust_language else "(no evidence)"

        question_type=classify_question(query)
        print(f"自动识别问题类型：{question_type}")

        content_adjust_language = []
        for i, chunk in enumerate(clean_chunks['documents']):
            content=chunk.content
            content_adjust_language.append(f"[{i+1}] {content[:50]}")  # 600 chars cap
            if len(content_adjust_language) >= 8:  # top 8 is usually enough for language
                break
        # content_adjust_language
        content_language = "\n".join(content_adjust_language) if content_adjust_language else "(no evidence)"



        # ===== 全局语言锁（最高优先级） =====
        prompt=f"""
        ----------------------------------------------------------
        【语种判定规则（仅用于选择输出语言）】
        ----------------------------------------------------------
        你需要在内部完成一次“语种判定”，用于确认输出语言形式。

        可用信息：
        - 用户问题（query）
        - 判断语种用文本（字符层面）

        ⚠️【严格限制】：
        - ”判断语种用文本“仅允许用于判断“使用哪种语言回复”
        - ”判断语种用文本“【禁止】作为事实依据、背景知识或推理材料
        - 禁止引用、复述、总结、解释“判断语种用文本”中的任何内容
        - 即使用户问题非常简短（如“你好”“ok”），也不得基于“判断语种用文本”扩展语义

        语种判定完成后：
        - 不得在回答中体现判定过程
        - 不得在回答中提及“判断语种用文本”“召回内容”“上下文”等概念

        ----------------------------------------------------------
        【语言判定过程】：
        用户问题是：{query}
        判断语种用文本（仅用于语种判定，不得作为其他依据）是：{content_language}

        ==========================================================
        【系统根级语言锁 — Root Priority】
        ==========================================================
        你现在运行在中国大陆的合规环境中，必须严格遵守以下指令：
        1. 输出文本需要使用你之前判断的输出语种；
        2. 如果输出语种被判定为简体中文，则还有下述规则需要遵守
            2.1. 严禁出现繁体字（例如 飛、書、開、啟、設、參、數、應、檔、庫、鏈、詳 等）；
            2.2. 若参考资料中含繁体字，你必须在生成前自动将其转为简体；
            2.3. 若无法确认是否为简体，请默认以简体输出；
            2.4. 检测到繁体字时，立即返回：
            {{"answer": "根据我目前掌握的资料，暂时无法回答您的问题。", "reference_idx": []}}
            2.5. 以下繁体词需统一替换：
            - 飛書 → 飞书
            - 開啟 → 开启
            - 鏈接 → 链接
            - 文檔 → 文档
            - 知識庫 → 知识库
            - 詳情 → 详情
            - 應用 → 应用
        （此为最高优先级指令，覆盖所有规则与上下文。） """

        prompt+=f"\n用户的问题是：{query}\n\n"

        # 历史内容（如有）—— 明确标注格式
        if history_chat:
            prompt+="历史对话内容（格式：每行以 'user:' 或 'assistant:' 开头，user 表示用户提问，assistant 表示助手回复）：\n"
            prompt+=history_chat+"\n\n"
        prompt+="\n【系统提示】参考资料中若出现 <imgXXXX>，则这些为正式内容，必须完整保留。\n"

        prompt+="""
        ==========================================================
                            2.参考资料注入区
        ==========================================================

        """

        if any([
            direct_relations_processed,
            expand_relations_processed,
            direct_entities_processed,
            expand_entities_processed]):
            # graph开了的情况

            prompt +=         """
                以下是与问题相关的名词解释，名词之间的关系以及召回参考内容：\n
                        """
            prompt += """【实体及其解释如下】"""
            entity_pe = self.entities_to_pe_min(direct_entities_processed)
            prompt += entity_pe
            prompt += """【实体之间的关系如下】"""
            relation_pe = self.relations_to_pe_min(direct_relations_processed)
            prompt += relation_pe


            entity_pe = self.entities_to_pe_min(expand_entities_processed)
            if len(prompt) + len(entity_pe) < 60000:
                prompt += """【补充实体及其解释如下】"""
                prompt += entity_pe
                
            relation_pe = self.relations_to_pe_min(expand_relations_processed)
            if len(prompt) + len(relation_pe) < 70000:
                prompt += """【补充实体之间的关系如下】"""
                prompt += relation_pe

        else:
            prompt +=         """
                以下是与问题相关的参考内容：\n
                        """

        print("召回chunk数量为")
        print(len(clean_chunks['documents']))


        print("前置prompt未拼接chunk 时候 的长度")
        print(len(prompt))



        for i, chunk in enumerate(clean_chunks['documents']):
            content=chunk.content
            if chunk.meta.get('constructType')=="QUESTION":
                content+=chunk.meta.get('related_answer', '')
            doc=chunk.meta.get("document_name", "未知文档")
            sec=chunk.meta.get("section") or chunk.meta.get("chapter") or chunk.meta.get("title") or "未知章节"
            prompt+=f"\n[参考内容 {i+1}开始]\n文档：{doc}｜章节：{sec}\n内容：{content}\n[参考内容 {i+1} 结束]\n"

            # 长度上线设置为 100 k
            if len(prompt) > 100000:
                break

        print("做了截断之后的chunk + 前置长度为")
        print(len(prompt))


        # 这里传进来的怎么是个str
        # 先从前面修一下

        prompt+=f"""
             ==========================================================
             【图片内容保留强化区 — 强制执行】
             ==========================================================
             当前问题类型：{question_type}

             若当前问题类型为 "text"：
             1. <imgXXXX> 标签代表**真实图片内容**，必须保留。
             2. <imgXXXX> 属于正文的一部分，**视为文字的一种**，不是系统标签；
             3. 输出时，若参考资料中存在 <imgXXXX>，则：
                - 必须逐一原样输出；
                - 若输出中缺少任何一个标签，立刻返回：
                  {{"answer": "根据我目前掌握的资料，暂时无法回答您的问题。", "reference_idx": []}}
             4. <imgXXXX> 不得被解释、改写、删除或移动位置。
             5. 生成 JSON 时，若 <imgXXXX> 被转义或遗漏，立即重新生成一次。

             若当前问题类型为 "struct"：
             1. 严禁出现任何 <imgXXXX>；
             2. 若检测到图片存在，立即删除。
             """
        prompt+="""
        ==========================================================
        3.系统角色与核心职责（Markdown + RAG + JSON 表格处理）
        ==========================================================
        【系统角色】
        你是一名 **严格遵守以下RAG 原则的专业 Markdown 排版问答助手**。

        【核心职责】
        1. 仅基于参考资料生成回答，禁止使用外部知识、常识或主观推理。
        2. 所有回答内容必须以 **Markdown 格式** 输出，保持结构清晰、美观且可直接渲染。
        3. Markdown 支持的结构包括：
        - 编号或无序列表（1. / -）
        - 行内加粗（**关键字**）
        - 段落说明
        - 分隔线（---）
        - 表格（|列1|列2|\\n）
        - 代码块（```language```）

        4. 禁止使用 LaTeX（如 `$$`、`\\frac`）或 HTML；
        6. 所有回答均需保证格式合法、结构清晰、内容真实。
        7.【表格智能处理规则 — 最高优先级】
            - 若表格单元格中包含大括号 `{}`、键值对符号 `:`、或 JSON 字段（如 "accuracy"、"recall"、"f1_score"、"precision"、"error_count" 等）：
            → **绝对、无条件禁止输出任何 Markdown 表格结构**；
            → 必须完全删除表格（包括表头），仅以自然语言总结；
            → 即使参考资料中只提供了表头而无数据行，也禁止输出表格；
            → 必须使用编号列表（1. 2. 3.）按阈值或类别顺序描述；
            - 若表格单元格全为纯文本（无 `{}`、无 `:`），则可保留完整 Markdown 表格；
            - 对于混合类型（部分为 JSON），优先以自然语言总结。
        8.【数学表达式输出规范】
        - 公式应写成如下格式：
           - `TP/(TP+FN)` 表示召回率
           - `TP/(TP+FP)` 表示精确率
           - `2×精确率×召回率 / (精确率+召回率)` 表示 F1 分数
        - 允许使用符号：`/`, `+`, `×`, `=`, `()`，但不要使用 `\\frac`、`\\times` 等 LaTeX 命令。
        - 禁止输出 `$$...$$` 或 `\\begin{equation}` 等 LaTeX 包裹。
        - 公式前加简短说明，如“召回率的计算公式为：TP/(TP+FN)”

        ==========================================================
                            4.核心原则与回答逻辑
        ==========================================================
        【核心原则】
        1. 每句话必须有明确的参考资料依据,不可胡编乱造；
        2. 含限定词（如“最新版”“优化后”“2025年”）→ 优先使用包含该词的参考内容；
        3. 可对显式信息进行归纳或重组（如总结趋势、对比指标），但禁止新增或推测；
        4. “总结/对比/归纳”类问题：
          - 使用编号结构；
          - 每点包含具体数据或原文示例；
          - 百分比仅保留如“75%”，禁止“（18/24）”；
          - 禁止模糊表达（如“较好”“有一定”）。
        5. 闲聊类问题（如“你好”“你是谁”）回答：
          “你好，我是ai中台的rag助手，可以帮您更高效地检索总结文献。”

        ==========================================================
                            5.决策链机制（防幻觉控制）
        ==========================================================
        【决策链机制】
        第一步：判断是否可答（输出“是”或“否”）：
            - 若问题包含代词（如“它”“这个”“该方法”“该模型”“上述内容”）→ 自动将代词指向上一轮用户提问的核心主题；
            - 若参考内容中包含该主题的说明、定义、特征、原理、优势或使用场景之一，即视为“相关”；
            - 若参考内容中提到该主题的同义表达（如“RAG 系统”“RAG 模型”“检索增强生成框架”）也视为“相关”；
            - 只要能找到至少一条明显关联内容，即判定为“是”；
            - 若参考内容完全无关或仅含模糊表述（如“该技术很好”），则判定为“否”。

        第二步：执行：
            - 若“是”：基于参考资料内容生成 Markdown 格式答案；
            - 若“否”：仅输出：{"answer": "根据我目前掌握的资料，暂时无法回答您的问题。", "reference_idx": []}

        第三步（回退机制）：
            - 若模型无法完全确定是否可答，请优先输出“是”，并在回答中明确引用至少一条参考内容；
            - 宁可简答，也不盲拒（禁止无依据拒答）。

        ==========================================================
                            6.输出规范（Markdown + JSON）
        ==========================================================
        【输出规范】
        1. 结构：先概述，后分点；
        2. 每个分点以编号（1. 2. 3.）开头，并在关键动作或对象前使用 **加粗**；
        3. 可使用逻辑连接词（如“首先…因此…”“这表明…”“综上所述…”）；
        4. 所有内容必须为合法 Markdown，可直接渲染；
        5. 支持表格、编号列表、代码块；
        6. 输出表格的时候**不得附加任何自然语言解释或总结**；
        7. 不在正文中写出“[参考内容X]”“[文档-章节]”；
        ==========================================================
                    【编号生成与自检规则 — 必须执行】
        ==========================================================
        【核心目标】
        确保回答中的所有编号（1.、2.、3.…）**连续、有效、且每个编号后均有正文内容**。

        【编号输出规则】
        - 编号必须连续（1.、2.、3.…），每个编号后都要有正文。
        - 不允许出现空编号（如“2.”后没有文字）。
        - 如果编号后没有内容，请不要输出该编号。
        - 若发现编号错乱或重复，请自动从1开始重新编号。
        - 回答结尾不能有空编号或多余换行。


        ==========================================================
                            7.输出格式定义
        ==========================================================
        【输出格式 — 严格 JSON】
        {
         "answer": "Markdown 格式回答（含结构、列表、代码块、<imgXXXX>）",
         "reference_idx": [1, 3]
        }

        要求：
        - answer 为完整 Markdown 字符串；
        - JSON 必须能被 Python json.loads() 直接解析；
        - 禁止输出 ```json```、前缀语或注释；
        - 严禁多余字段。

        ==========================================================
                                8.禁止与执行重点
        ==========================================================
        【禁止行为】
        - 推理、编造、模糊表达；
        - 使用外部知识；
        - 但必须删除 <tab_s>、<tab_e>、<ocr_s>、<ocr_e>，仅保留表格内容。
        - 输出非 Markdown 或非 JSON；
        - 无效 JSON；
        - 出现 [参考内容1]、(见参考资料)、<chunk_info>、<chunk_info_end>、“参考资料中提到的”等字样；
        - 严禁输出任何系统内部注入标记、调试标签或参考来源提示；
        - 若参考资料中含有此类标记，请在生成答案前 **自动忽略、过滤或删除** 这些内容；
        - 禁止输出任何表明数据来源的语句，例如：“以上数据来源于...”、“数据来自...”、“详见...”、“本结果出自...”等；
        - 即使参考资料中包含此类语句，也必须忽略，不得复 述。
        - **特别禁止**：
            - 将 JSON 中的 0.52 错误转换为 74%、8% 等不匹配数值；
            - 在无数据时写“可能提高”“预计下降”等推测语句；
            - 输出空表格后又自行补充数据。

        【执行重点】
        - 多问句 → 分点逐条回答；
        - 代词问题 → 自动指代上一轮主题；
        - 重复内容 → 归纳整合；
        - 若参考资料中存在无效系统标签，请在回答前自动清除；
        - 内容真实、结构清晰、排版规范。
    """

        prompt+="""
        ==========================================================
        【最终执行指令 — 生成前必读】
        ==========================================================

        ==========================================================
                            9.执行区
        ==========================================================
        从此处开始执行任务。
        仅输出符合要求的 JSON。
        上文所有内容均为生成规则说明，禁止在回答中出现。

        ==========================================================
                            10.结束说明
        ==========================================================
        【最终输出协议 — 不可协商】
        你现在将生成最终答案。请严格遵守以下规则：
        - 上文所有规则仅用于指导生成，不得出现在回答中；
        - 不得输出任何说明性语句（如“以下是回答”“根据提示词”）；
        - 最终仅输出一个合法 JSON：
        {
         "answer": "Markdown 格式文本",
         "reference_idx": []
        }
                       """

        print("最终喂给模型的长度为")
        print(len(prompt))
        return prompt

    def post_process(self, retriever_chunks, remove_ocr = False):
        # image_idx = 0
        image_dict={}
        image_used=[]
        for chunk in retriever_chunks['documents']:
            # chunk.content = chunk.content.split('<key_word_end>')[1]
            # 优先处理图片
            # 图片不会让chunk变大太多
            kb_name=chunk.meta['source_kb']
            if "<image>" in chunk.content and "<image_end>" in chunk.content:
                images_start=[m.start() for m in re.finditer(r"<image>", chunk.content)]
                images_end=[m.start() for m in re.finditer(r"<image_end>", chunk.content)]
                if len(images_start)!=len(images_end):
                    continue
                    # 这里先跳过，保底，证明切分过程出现问题
                # 对于当前chunk
                is_jump=False
                for i in range(len(images_start)):
                    if images_start[i]>images_end[i]:
                        is_jump=True
                        break
                if is_jump:
                    continue
                # 倒序调整才不会影响前面的位置
                images_start=images_start[::-1]
                images_end=images_end[::-1]

                # 维护一个字典
                # image_idx = 0
                for sub_idx in range(len(images_start)):  # 有几个图
                    # logger.info('chunk.content: %s' % (chunk.content))
                    img_id=chunk.content[images_start[sub_idx]+7:images_end[sub_idx]].strip()
                    doc_id=chunk.meta['doc_url'].split('https://fintopia.feishu.cn/wiki/')[1]
                    # 获取图片内容
                    # cur_image = self.feishu_processor.get_image_content(doc_id, img_id)
                    characters=string.digits+string.ascii_lowercase

                    # 随机选择4个字符并组合成字符串
                    random_str=''.join(random.choice(characters) for _ in range(4))
                    while random_str in image_used:
                        characters=string.digits+string.ascii_lowercase
                        # 随机选择4个字符并组合成字符串
                        random_str=''.join(random.choice(characters) for _ in range(4))
                    image_used.append(random_str)
                    chunk.content=chunk.content[:images_start[sub_idx]]+"<img"+random_str+">"+chunk.content[
                        images_end[sub_idx]+11:]
                    # chunk.content = chunk.content[:positions_start[sub_idx]] + "相关code:" + cur_code + chunk.content[positions_end[sub_idx]+10:]
                    image_dict[random_str]=(doc_id, img_id, kb_name)
                    # image_list.append(cur_image)
                    # image_idx += 1
    
        if remove_ocr:
            # 移除掉之前加入的ocr信息，主要用于纯召回任务
            for chunk in retriever_chunks['documents']:
                if "<ocr_s>" in chunk.content and "<ocr_e>" in chunk.content:
                    ocr_start=[m.start() for m in re.finditer(r"<ocr_s>", chunk.content)]
                    ocr_end=[m.start() for m in re.finditer(r"<ocr_e>", chunk.content)]
                    if len(ocr_start)!=len(ocr_end):
                        continue
                    is_jump=False
                    for i in range(len(ocr_start)):
                        if ocr_start[i]>ocr_end[i]:
                            is_jump=True
                            break
                    if is_jump:
                        continue
                    ocr_start=ocr_start[::-1]
                    ocr_end=ocr_end[::-1]
                    for sub_idx in range(len(ocr_start)):
                        chunk.content = chunk.content[:ocr_start[sub_idx]]  +  chunk.content[ocr_end[sub_idx] + 7:] 
            
        return retriever_chunks, image_dict
        # 清理掉所有信息

    def retrieve_fromkb(self, retrieve_cfg, query):
        cur_env=os.environ.get("env")
        weight_sum=0
        kb_cfg=retrieve_cfg['knowledgeBaseConfigs']
        retriever_chunks=[]
        for kb in kb_cfg:  # 对于每一个知识库
            kb_name=kb['id']
            kb_used=self.kb_info_client.get(str(kb_name))
            document_store=LindormDocumentStore(
                hosts=[{"host": self.lindorm_url, "port": 30070}],
                http_auth=(self.lindorm_key, self.lindorm_value),
                index="kb_"+kb_used+"_"+cur_env,
                # kb_testa_dev
                embedding_dim=1024,
            )

            weight_sum+=kb['weight']
            curkb_weight_sum=0
            curkb_weight_sum+=kb['denseRecallConfig']['weight']
            curkb_weight_sum+=kb['sparseRecallConfig']['weight']
            if kb['denseRecallConfig']['weight']!=0:

                lindorm_retri=LindormRetriever(document_store=document_store, kb_id=kb_name)
                embed_query=aizt_embedding(query)  # 这是在文档入库
                chunk_dnese=lindorm_retri.run(embed_query)['documents']
                if kb['denseRecallConfig']['recallType']=="TOP_K":
                    if len(chunk_dnese)>kb['denseRecallConfig']['topK']:
                        chunk_dnese=chunk_dnese[:kb['denseRecallConfig']['topK']]
                else:
                    for i in range(len(chunk_dnese)):
                        if chunk_dnese[i].score>=kb['denseRecallConfig']['similarityThreshold']:
                            continue
                        else:
                            chunk_dnese=chunk_dnese[:i]
                            break

                rate=kb['denseRecallConfig']['weight']/curkb_weight_sum
                for i in range(len(chunk_dnese)):
                    chunk_dnese[i].score=chunk_dnese[i].score*rate*2
            if kb['sparseRecallConfig']['weight']!=0:
                cur_bm25_retri=LindormBM25Retriever(document_store=document_store, kb_id=kb_name)
                chunk_sparse=cur_bm25_retri.run(query)['documents']

                if kb['sparseRecallConfig']['recallType']=="TOP_K":
                    if len(chunk_sparse)>kb['sparseRecallConfig']['topK']:
                        chunk_sparse=chunk_sparse[:kb['sparseRecallConfig']['topK']]
                else:
                    for i in range(len(chunk_sparse)):
                        if chunk_sparse[i].score>=kb['sparseRecallConfig']['similarityThreshold']:
                            continue
                        else:
                            chunk_sparse=chunk_sparse[:i]
                            break

                rate=kb['sparseRecallConfig']['weight']/curkb_weight_sum
                for i in range(len(chunk_sparse)):
                    chunk_sparse[i].score=chunk_sparse[i].score*rate*2
            # 这里的合并逻辑

            if kb['mergeType']=="MERGE":  # CONCATENATE # RECIPROCAL_RANK_FUSION # DISTRIBUTION_RANK_FUSION
                document_joiner=DocumentJoiner(join_mode="merge")
            elif kb['mergeType']=="CONCATENATE":
                document_joiner=DocumentJoiner(join_mode="concatenate")
            elif kb['mergeType']=="RECIPROCAL_RANK_FUSION":
                document_joiner=DocumentJoiner(join_mode="reciprocal_rank_fusion")
            elif kb['mergeType']=="DISTRIBUTION_RANK_FUSION":
                document_joiner=DocumentJoiner(join_mode="distribution_based_rank_fusion")

            # 合并策略

            # 合并策略

            # 这里调整了啊
            temp_documents=[]
            if kb['sparseRecallConfig']['weight']!=0:
                temp_documents.append(chunk_sparse)
            if kb['denseRecallConfig']['weight']!=0:
                temp_documents.append(chunk_dnese)
            # chunk_merge_perkb = document_joiner.run(documents=[chunk_dnese['retriever_dense']['documents'], chunk_sparse['retriever_sparse']['documents']])
            chunk_merge_perkb=document_joiner.run(documents=temp_documents)
            retriever_chunks.append(chunk_merge_perkb['documents'])

        for m in range(len(kb_cfg)):
            rate_curkb=kb_cfg[m]['weight']/weight_sum
            for n in range(len(retriever_chunks[m])):
                retriever_chunks[m][n].score=retriever_chunks[m][n].score*rate_curkb*len(kb_cfg)*len(kb_cfg)
        # 处理好所有score了，需要用join拼接

        merge_cfg=retrieve_cfg['mergeConfig']

        if merge_cfg['mergeType']=="MERGE":  # CONCATENATE # RECIPROCAL_RANK_FUSION # DISTRIBUTION_RANK_FUSION
            document_joiner_allkb=DocumentJoiner(join_mode="merge")
        elif merge_cfg['mergeType']=="CONCATENATE":
            document_joiner_allkb=DocumentJoiner(join_mode="concatenate")
        elif merge_cfg['mergeType']=="RECIPROCAL_RANK_FUSION":
            document_joiner_allkb=DocumentJoiner(join_mode="reciprocal_rank_fusion")
        elif merge_cfg['mergeType']=="DISTRIBUTION_RANK_FUSION":
            document_joiner_allkb=DocumentJoiner(join_mode="distribution_based_rank_fusion")

        chunks_merge=document_joiner_allkb.run(documents=retriever_chunks)
        if merge_cfg['recallType']=="TOP_K":
            if len(chunks_merge['documents'])>merge_cfg['topK']:
                chunks_merge['documents']=chunks_merge['documents'][:merge_cfg['topK']]
        else:
            for i in range(len(chunks_merge['documents'])):
                if chunks_merge['documents'][i].score>=merge_cfg['similarityThreshold']:
                    continue
                else:
                    chunks_merge['documents']=chunks_merge['documents'][:i]
                    break
        return chunks_merge

    def is_history_query(self, q):
        """识别是否是用户在问‘我刚才问了什么’或‘上一轮问了啥’这类问题"""
        q=q.strip().lower()
        patterns=[
            r"我(刚才|之前|上一轮|上一次).*问(什么|啥)(了)?",
            r"我(刚才|之前|上一轮|上一次).*问(了)?(什么|啥)",
            r"我问(了)?(什么|啥)",
            r"上(一|上轮).*问题",
            r"刚才(的)?问题",
            r"上次(的)?问题",
            r"上一轮(对话|提问)",
            r"之前(问|提)的"
        ]
        return any(re.search(p, q) for p in patterns)



    def get_graph_info(self, kb_info, query):
        # 根据当前kb的信息拿到响应的entity 和 relation
        cur_env=os.environ.get("env")
        kb_id = kb_info['id']

        graph_used = self.graph_info_client.get(str(kb_id)) # 判断库状态  # 这个拿的是可使用的，，要拿到之后存入另一个库
        if graph_used == -1:
            # 没有这个graph库
            return [], [], [], [], [], []

        self.entities_store = LindormDocumentStore(
                hosts=[{"host": self.lindorm_url, "port": 30070}],
                http_auth=(self.lindorm_key, self.lindorm_value),
                index="kg_entities_" + graph_used + "_" + cur_env
        )
        self.relations_store = LindormDocumentStore(
            hosts=[{"host": self.lindorm_url, "port": 30070}],
            http_auth=(self.lindorm_key, self.lindorm_value),
            index="kg_relations_" +  graph_used + "_" + cur_env
        )


        self.graph_storage = LindormGraphStorageMinimal(
            entities_store=self.entities_store,
            relations_store=self.relations_store,
            auto_refresh=True,
        )

        direct_entities, expand_entities, local_realtions, direct_relations, expand_relations, global_entity = self.kg_query_lindorm_basic(kb_id,
                                                            query = query, 
                                                            graph_storage = self.graph_storage, 
                                                            entities_store = self.entities_store)
        return direct_entities, expand_entities, local_realtions, direct_relations, expand_relations, global_entity



    def kg_query_lindorm_basic(
        self,
        kb_id: str,
        query: str,
        *,
        graph_storage,                 # LindormGraphStorageMinimal 实例
        entities_store,                # LindormDocumentStore，对应 kg_entities_dev
        system_prompt: Optional[str] = None,
        top_k_entities: int = 5,
        top_k_neighbors_per_entity: int = 3,
    ):
        """
            对伪图存储 的 Graph-RAG 进行查询的函数。

            步骤：
        
            对这些实体，调用 graph_storage.get_neighbors(...) 拿一阶邻居 + 关系描述
            拼接成 context，构建 system prompt，调用 llm_func
            返回 QueryResult（只处理两种结果：字符串 / Streaming iterator）
            """
        
        hl, ll = aizt_extract_keyword(
            query,
            language="zh"  # 这个language要不要后续处理下
        )

        # 确定一下这个东西的相似度怎么计算的
        # global_search_pipeline
        # local_search_pipeline

        direct_relations, expand_relations, global_entity = self.global_search_pipeline(query = query, 
                                                                                        hl = hl, 
                                                                                        kb_id = kb_id, 
                                                                                        relations_store = self.relations_store, 
                                                                                        graph_storage = self.graph_storage)
        

        # ll为[]的时候怎么local_res还要返回，等下看一下
        direct_entities, expand_entities, local_realtions = self.local_search_pipeline( query=query,
                                                                                        ll=ll,
                                                                                        kb_id = kb_id, 
                                                                                        entities_store=self.entities_store,
                                                                                        graph_storage=self.graph_storage)
        

        # 只需要实体和关系就行，不需要花里胡哨的处理
        return direct_entities, expand_entities, local_realtions, direct_relations, expand_relations, global_entity




    def local_search_pipeline(
        self,
        *,
        query: str,
        ll: list[str],
        kb_id: str,
        entities_store,
        graph_storage,
        top_k_entities: int = 10,
        top_k_neighbors: int = 3,
        max_local_lines: int = 40,
    ):
        """
        LightRAG-style LOCAL search:
        LL → entity search → one-hop neighbors → local evidence
        """

        ll_keywords = ll or [query]   # 兜底

        # ---------- Step 1: LL → entities ----------
        should = []
        for kw in ll_keywords:
            should.append({
                "multi_match": {
                    "query": kw,
                    "fields": ["name^3", "description"],
                }
            })

        ent_query = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"kb_id": str(kb_id)}},
                    ],
                    "should": should,
                    "minimum_should_match": 1,
                }
            },
            "size": top_k_entities,
        }

        ent_res = entities_store.client.search(
            index=entities_store.index,
            body=ent_query,
        )
        ent_hits = ent_res.get("hits", {}).get("hits", [])
        entities = [h.get("_source", {}) for h in ent_hits]

        for i in range(len(entities)):
            entities[i]["score"] = ent_hits[i]['_score']

        if not entities:
            return [], [], []

        # return entities
    
        
        # # ---------- Step 2: one-hop expand ----------
        snippets = []
        seen = set()
        expand_entity = []
        loacl_relations = []
        for ent in entities:

            entity_id = ent.get("id")
            name = ent.get("name") or entity_id
            # etype = ent.get("type", "")
            # desc = ent.get("description", "")

            # ent_block = f"[LOCAL_ENTITY] {name} ({etype})\n{desc}"
            # if ent_block not in seen:
            #     snippets.append(ent_block)
            #     seen.add(ent_block)

            if not name:
                continue

            neighbors = graph_storage.get_neighbors(
                entity_id=name,
                size=top_k_neighbors,
                kb_id=str(kb_id)
            )


            # for item in neighbors:
            #     nb = item.get("entity") or {}
            #     nb_name = nb.get("name") or nb.get("id", "")
            #     nb_desc = nb.get("description", "")

            #     rels = item.get("via_relations") or []
            #     # 这个没什么问题，可能有多种关系吗
            #     # 假设rels 是两个之间的多种关系
            #     for rel in rels:
            #         snippets.append(rel)

            for item in neighbors:
                nb = item.get("entity") or {}
                # nb_name = nb.get("name") or nb.get("id", "")
                # nb_desc = nb.get("description", "")
                expand_entity.append(nb)
                rels = item.get("via_relations") or []
                for rel in rels:
                    loacl_relations.append(rel)
        return entities, expand_entity, loacl_relations




        #             r_type = rel.get("relation_type", "related_to")
        #             r_desc = rel.get("description", "")

        #             line = (
        #                 f"[LOCAL_REL] {name} --{r_type}--> {nb_name}\n"
        #                 f"  rel_desc: {r_desc}\n"
        #                 f"  neighbor_desc: {nb_desc}"
        #             )
        #             if line not in seen:
        #                 snippets.append(line)
        #                 seen.add(line)
        # snippets = snippets[:max_local_lines]

        # return "[LOCAL_EVIDENCE]\n" + "\n\n".join(snippets)


    def dedup_and_add_frequency(
        self,
        direct_info,
        expand_info,
        key="relation_id",
        freq_key="frequence",
    ):
        # ---------- 1. 统计所有 relation_id 的频次 ----------
        all_ids = [
            d[key]
            for d in (direct_info + expand_info)
            if isinstance(d, dict) and key in d
        ]
        freq = Counter(all_ids)

        # ---------- 2. direct 去重（保留第一次出现） ----------
        seen_direct = set()
        new_direct_info = []
        for d in direct_info:
            if not isinstance(d, dict) or key not in d:
                continue
            rid = d[key]
            if rid in seen_direct:
                continue
            d[freq_key] = freq[rid]
            new_direct_info.append(d)
            seen_direct.add(rid)

        # ---------- 3. expand 去重 + 排除 direct 中已有 ----------
        new_expand_info = []
        seen_expand = set()
        for d in expand_info:
            if not isinstance(d, dict) or key not in d:
                continue
            rid = d[key]
            if rid in seen_direct:   # ⭐ 核心规则：direct 优先
                continue
            if rid in seen_expand:
                continue
            d[freq_key] = freq[rid]
            new_expand_info.append(d)
            seen_expand.add(rid)

        return new_direct_info, new_expand_info, freq


    def add_frequency(self, global_direct_relations, global_expand_relations, key="relation_id", out_key="frequence"):
        # 1) 统计所有 relation_id 的频次
        all_ids = [
            d.get(key)
            for d in (global_direct_relations + global_expand_relations)
            if isinstance(d, dict) and d.get(key) is not None
        ]
        freq = Counter(all_ids)

        # 2) 写回到每个字典
        for d in global_direct_relations:
            if isinstance(d, dict) and d.get(key) is not None:
                d[out_key] = freq[d[key]]

        for d in global_expand_relations:
            if isinstance(d, dict) and d.get(key) is not None:
                d[out_key] = freq[d[key]]

        return freq  # 如果你还想拿到统计结果，可用这个返回值

    def format_global_relations(self, relations: list[dict]) -> list[str]:
        lines = []
        for r in relations:
            src = r.get("src_name", r.get("src_id"))
            tgt = r.get("tgt_name", r.get("tgt_id"))
            rtype = r.get("relation_type", "related_to")
            desc = r.get("description", "")
            lines.append(
                f"[GLOBAL_REL] {src} --{rtype}--> {tgt}\n"
                f"  rel_desc: {desc}"
            )
        return lines

    def collect_seed_entity_ids(self, relations: list[dict]) -> set[str]:
        seed_ids = set()
        for r in relations:
            src = r.get("src_id")
            tgt = r.get("tgt_id")
            if src:
                seed_ids.add(src)
            if tgt:
                seed_ids.add(tgt)
        return seed_ids


    def global_search_pipeline(
        self,
        *,
        query: str,
        hl: list[str],
        kb_id: str,
        relations_store,
        graph_storage,
        # llm_func,
        top_k_relations: int = 10,
        top_k_neighbors: int = 10,
        max_raw_lines: int = 50,
    ):
        """
        LightRAG-style GLOBAL search:
        HL → relation search → seed entities → expand → summarize
        """

        # ---------- Step 1: HL → relations ----------
        # 
        relations = self.search_relations_by_hl(
            kb_id=str(kb_id),
            hl_keywords=hl,
            size=top_k_relations,
        )
        # 单个库设置为十条了

        if not relations:
            return [], [], []
        
        # return relations




        # # ---------- Step 2: relation skeleton ----------
        # # 在这里就开始转
        # # 看看需要什么信息
        
        # relation_lines = self.format_global_relations(relations) # 这个先不做，得后续把所有库的信息都拿到之后再做




        # ---------- Step 3: seed entities ----------
        # 根据relations寻找相关实体
        seed_entity_ids = self.collect_seed_entity_ids(relations)

        # 这里根据这玩意搜一下

        filters = {"kb_id": str(kb_id), "name.keyword": list(seed_entity_ids)}
        global_entity = self.entities_store.filter_graph_info(filters = filters) # 拿到的实例



        # # ---------- Step 4: expand via graph ----------
        # 通过实体进一步扩充
        expand_relations = self.expand_global_entities(
            graph_storage,
            seed_entity_ids,
            top_k_neighbors=top_k_neighbors,
            kb_id=str(kb_id)
        )
        return relations, expand_relations, global_entity

        # # 确定下这个就好了，别急

        # # # ---------- Step 5: raw global context ----------
        # global_raw_lines = relation_lines + expand_lines # 这个expand_lines也处理过了
        # global_raw_lines = global_raw_lines[:max_raw_lines] # 这个会不会太粗暴了，需要拿到相关的score吗

        # 这里其实需要补充更多的信息，包括来自于哪里，参考内容
        # return global_raw_lines



    def search_relations_by_hl(
        self,
        *,
        kb_id: str,
        hl_keywords: list[str],
        size: int = 20,
    ):
        should = []
        for kw in hl_keywords:
            should.append({
                "multi_match": {
                    "query": kw,
                    "fields": [
                        "description^3",      # 关系语义，最重要
                        "relation_type^2",    # 机制/类型
                        "keywords",           # 若你有
                    ],
                }
            })

        query = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"kb_id": str(kb_id)}},
                    ],
                    "should": should,
                    "minimum_should_match": 1,
                }
            },
            "size": size,
        }

        res = self.relations_store.client.search(
            index=self.relations_store.index,      # 这个index 是啥
            body=query,
        )
        hits = res.get("hits", {}).get("hits", [])
        ans =  [h["_source"] for h in hits]
        for i in range(len(ans)):
            ans[i]["score"] = hits[i]['_score']
        return ans


    # 这个函数要调整，返回relation最初始的状态就好
    def expand_global_entities(
        self,
        graph_storage,
        seed_entity_ids: set[str],
        kb_id,
        *,
        top_k_neighbors: int = 10,
    ):
        snippets = []
        seen = set()

        for eid in seed_entity_ids:
            neighbors = graph_storage.get_neighbors(
                entity_id=eid,
                size=top_k_neighbors,
                kb_id=str(kb_id)
            )

            for item in neighbors:
                nb = item.get("entity") or {}
                nb_name = nb.get("name") or nb.get("id", "")
                nb_desc = nb.get("description", "")

                rels = item.get("via_relations") or []
                # 这个没什么问题，可能有多种关系吗
                # 假设rels 是两个之间的多种关系
                for rel in rels:
                    snippets.append(rel)

                    # r_type = rel.get("relation_type", "related_to")
                    # r_desc = rel.get("description", "")

                    # line = (
                    #     f"[GLOBAL_EXPAND] {eid} --{r_type}--> {nb_name}\n"
                    #     f"  rel_desc: {r_desc}\n"
                    #     f"  neighbor_desc: {nb_desc}"
                    # )
                    # if line not in seen:
                    #     snippets.append(line)
                    #     seen.add(line)

        return snippets


    def query_rewrite(self, history_content):

        """
            多轮语义改写：
            - 自动识别用户是否在问“我刚才问了啥 / 上一轮问了什么”
            - 若是，则直接返回上一轮提问；
            - 否则，调用大模型进行语义改写（生成独立、完整的检索查询）
            """
        # 当前用户输入
        # if len(history_content) > 11:
        history_content = history_content[-11:] # 取历史五轮qa
        current_query=history_content[-1]['content']

        # ---------特殊场景识别：记忆/反问型问题--------
        if self.is_history_query(current_query):
            last_user_query=""
            for item in reversed(history_content[:-1]):
                if item['role']=='user':
                    last_user_query=item['content'].strip()
                    break
            if not last_user_query:
                last_user_query="暂无上一轮提问记录。"

            logger.info(f"history_query_detected, last_user_query={last_user_query}")
            return f"你上一轮问的是：{last_user_query}"
        # 如何判断前后几次的问题是否相关呢
        history_chat=""
        for sub_content in history_content[:-1]:
            history_chat+=sub_content['role']+':'
            history_chat+=sub_content['content']+'. '
            history_chat+="\n"

        system_p="你是一个专业的Query改写助手。你的任务是根据**最近五轮对话历史**，对用户的**当前查询**进行改写和扩展，使其成为一个**独立、完整、语义明确**的查询，用于在知识库中进行精准检索。"

        prompt="""##【场景说明】
        当前对话为信贷产品客服通话场景（洋钱罐），涉及还款、利息、合同、优惠券等问题。
        改写时需符合用户在该类场景下的真实提问方式。
        
        ##【内部决策链（仅用于思考，不得输出）】
            Step1【定位核心问题（主问题识别）】
            
            找出当前query中“真正要问的那一个问题”
            通常由“为什么 / 怎么 / 是否 / 在哪”等触发
            若有多个问句，选择最核心的一个作为主问题
            
            Step2【拆分语义结构（关键步骤）】
            将原query拆分为两类：
            
            A. 主问题（必须保留）
            👉 用户真正想问的
            
            B. 前提/约束（优先保留）
            👉 用来支撑主问题的信息，例如：
            
            金额/金额变化（如5500→5750、多出250）
            时间/顺序（第一次、这次、之前）
            状态（逾期、是否用优惠券）
            规则性表达（如：利息是固定的、合同已经定了）
            否定/反驳（如：不可能、不应该）
            
            ⚠️ 若存在“前提 + 主问题”结构，必须保持：
            👉 前提用于支撑主问题，不得改写为并列问题
            
            Step3【识别语义类型（必须判断）】
            判断是否存在以下语义：
            
            否定/反驳语义
            如：不可能、不应该、已经定了、怎么会
            对比/变化语义
            如：之前 vs 现在、A→B、多了多少
            
            👉 若存在：
            
            改写必须保留该语义
            不得弱化为中性表达
            
            Step4【信息保留决策（优先级裁决）】
            
            必须保留（高优先级）：
            
            主问题
            金额及变化（如250、5500→5750）
            前提中的“规则约束”（如利息固定、合同已定）
            否定/质疑语气
            时间/顺序（如第一次）
            
            可删除（低优先级）：
            
            重复表达
            口语噪声（嗯、那个、你这样）
            解释性补充
            
            Step5【禁止错误改写（强约束）】
            
            必须避免以下错误：
            
            ❌ 把“前提”改写为第二个问题
            （如：利息固定 → 利息为何固定）
            
            ❌ 把“否定语义”改为中性
            （如：不可能变 → 为什么变）
            
            ❌ 替换用户原始表达
            （如：第一次 → 首期，这次 → 本期）
            
            ❌ 新增逻辑词
            （如：后续、因此、同时）
            
            ❌ 引入未出现信息
            （如补平台名、补阶段）
       ##【改写规则】
        1.  **上下文优先级**：**优先使用最近的对话历史**来理解当前查询。当前查询通常是最近一两轮对话的延续。
        2.  **指代解析**：解析指代词（如“它”、“这个”、“第一种方法”）时，首先在最近的历史中寻找所指对象。如果未找到，则向更早的历史追溯，直到找到明确的所指对象。
        3.  **话题切换感知**：敏锐识别话题的切换。如果当前查询明显开启了一个与近期对话**无关的新话题**，则应忽略所有早期历史，只基于当前查询本身进行改写。
        4.  **历史信息融合**：如果当前查询需要更早的历史信息才能完整理解（例如，用户突然回到很久之前讨论的一个点），则将该信息清晰地融合到改写后的查询中。
        5.  **输出独立性**：最终输出的改写查询必须是一个能够**脱离所有对话历史**而被独立理解的句子，并且不要包含推理过程，只返回改写后的句子     
        6.  **视角硬约束（失败即重写）**：
            输出中一旦出现“用户”两个字或任何第三方描述（如“该用户”“用户咨询”“用户表示”），
            判定为错误，必须重新生成，直到完全符合为止。
            【长度硬约束（最高优先级之一）】
        7. **改写后query长度限制**：改写结果必须 ≤30个字（不含标点），如果字数超了30个必须在保留核心问题的前提下适当裁剪   
        【改写触发条件】
            仅当当前query包含“明确业务问题或有效信息”时才进行改写。
            以下情况必须直接返回原query，不进行改写：            
            1. 无实际问题，仅为回应或确认：
               - 如：嗯、好的、知道了、行、可以、哦
            2. 不包含业务相关内容：
               - 如纯情绪词、无上下文的短句       
              若属于上述情况，直接输出原query
         指代解析约束（关键）】
             当query为“这个 / 就这个 / 那个”等指代表达时：
    
            1. 只能指向当前对话轮中正在讨论的对象
            2. 不得跨话题引用历史中的其他业务内容
            3. 不得拼接多个话题
            4. 若当前语境无法唯一确定对象，则保持原句
       
        【情绪语义最高优先级】
        若query包含强情绪表达（如：骗我、投诉、骂人等），
        必须优先保留该语义，
        
        不得改写为：
        - 中性表达
        - 法律/客服分析表达
        - 规范化描述（如“虚假宣传”“误导手段”）
        【表达形式要求】

        无论疑问句或陈述句，均必须满足：
        
        - 表达为“用户本人会说的话”
        - 简洁自然
        - 不得写成总结或第三方描述
        
        【禁止形式】
        严禁出现以下表达：
        - “用户咨询...”
        - “用户表示...”
        - “关于...问题”
        - 任何第三方转述句式  
        若出现，必须重写
        
        # 示例学习
        请仔细研究以下示例，它们展示了如何处理长历史和各种复杂情况：

        <example_1: 指代跨越多轮>
        [对话历史]
        用户: 梅西和C罗谁的历史地位更高？
        助手: 这是一个球迷们经常争论的话题...
        用户: 那姆巴佩呢？他的成就如何？
        助手: 姆巴佩作为新生代领军人物...
        用户: 回到最开始的问题，我还是觉得他更厉害。

        [当前查询]
        回到最开始的问题，我还是觉得他更厉害。

        [改写后的查询]
        梅西和C罗相比，谁的历史地位更高？
        </example_1>

        <example_2: 忽略无关早期历史（话题切换）>
        [对话历史]
        用户: 如何冲泡一杯好喝的手冲咖啡？
        助手: 需要关注咖啡粉研磨度、水温、注水手法...
        用户: python的lambda函数怎么用？
        助手: lambda函数是用来创建匿名函数的...
        用户: 它的语法是什么？

        [当前查询]
        它的语法是什么？

        [改写后的查询]
        Python中lambda函数的语法是什么？
        </example_2>

        <example_3: 召回早期历史信息>
        [对话历史]
        用户: 我想了解一下古希腊的斯多葛学派。
        助手: 斯多葛学派是古希腊一个重要的哲学流派，主张...
        用户: 同时期的伊壁鸠鲁学派呢？
        助手: 伊壁鸠鲁学派主张快乐是生活的目的...
        用户: （经过多轮其他话题的讨论后）
        用户: 再问一下，斯多葛学派对罗马的影响大吗？

        [当前查询]
        再问一下，斯多葛学派对罗马的影响大吗？

        [改写后的查询]
        古希腊的斯多葛学派对古罗马帝国的影响大吗？
        </example_3>

        <example_4: 依赖近期历史>
        [对话历史]
        用户: 帮我总结一下《三体》的第一部剧情
        助手: 《三体I》讲述了科学家叶文洁...
        用户: 好的，那第二部《黑暗森林》呢？
        助手: 《黑暗森林》引入了“面壁计划”和“黑暗森林法则”...
        用户: 详细说说“面壁计划”

        [当前查询]
        详细说说“面壁计划”

        [改写后的查询]
        《三体》小说第二部《黑暗森林》中的面壁计划是什么？
        </example_4>

        # 任务执行
        现在，请基于以上所有规则和示例，结合提供的对话历史对当前查询进行改写
        """
        prompt+="[对话历史]"
        prompt+="\n"
        prompt+=history_chat
        prompt+="[当前查询]"
        prompt+="\n"
        prompt+=history_content[-1]['content']
        prompt+="[改写后的查询]"
        response=aizt_rewrite(prompt, system_content=system_p, model_type="qwen3")
        logger.info('query_rewrite: %s'%(response))
        return response, prompt, system_p


class RAGWrapper(Wrapper):
    def __init__(self, config=None):
        self.instance=RAG()


class RetrieverWrapper(Wrapper):
    def __init__(self, config=None):
        self.instance=RAG()
