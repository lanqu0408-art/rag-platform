import json
import argparse
import lark_oapi as lark
from lark_oapi.api.docx.v1 import *
import json
import os
import lark_oapi as lark
import time
from lark_oapi.api.drive.v1 import *
from lark_oapi.api.wiki.v2 import *
from lark_oapi.api.sheets.v3 import *
from collections import deque
import requests
import logging
from core.config import config
# from client.Client import Client
import random
logger = logging.getLogger(__name__)
map_dict = {
    1:"page",       # 页面（根） Block
    2:"text",       # 文本 Block
    3:"heading1",   # 一级标题 Block
    4:"heading2",   # 二级标题 Block
    5:"heading3",   # 三级标题 Block
    6:"heading4",   # 四级标题 Block
    7:"heading5",   # 五级标题 Block
    8:"heading6",   # 六级标题 Block
    9:"heading7",   # 七级标题 Block
    10:"heading8",  # 八级标题 Block
    11:"heading9",  # 九级标题 Block
    12:"bullet",    # 无序列表 Block
    13:"ordered",   # 有序列表 Block
    14:"code"       # 代码部分
}




class Feishu_Process:
    """处理飞书的组件"""
    def __init__(self):
        # client = Client()
        self.app_id = config.get("app_id")
        self.app_secret = config.get("app_secret")
        self.client = lark.Client.builder() \
            .app_id(self.app_id ) \
            .app_secret(self.app_secret) \
            .log_level(lark.LogLevel.DEBUG) \
            .build()

        

    # 所有调用飞书接口的地方都加个重试机制



    def request_with_retry(self, method, url, headers, max_retries=5):
        """带重试的请求封装（指数退避 + 抖动）"""
        # headers = kwargs.pop("headers", {})
        # headers["Authorization"] = f"Bearer {self.tenant_access_token}"
        retry = 0
        backoff = 1

        while True:
            resp = requests.request(method, url, headers=headers)
            try:
                data = resp.json()
            except Exception:
                resp.raise_for_status()
                return resp  # 返回原始 resp

            # 频控错误
            if data.get("code") == 99991400:
                retry += 1
                if retry > max_retries:
                    raise Exception(f"超过最大重试次数 {max_retries}，仍然触发频控: {data}")
                wait_time = backoff + random.uniform(0, 0.5)
                print(f"触发频控，等待 {wait_time:.2f}s 后重试（第 {retry} 次）... url={url}")
                time.sleep(wait_time)
                backoff *= 2
                continue
            return data

    def get_board_content(self, whiteboard_id):
        # 获取 tenant_access_token
        auth_url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        auth_resp = requests.post(auth_url, json={
            "app_id": self.app_id,
            "app_secret": self.app_secret
        })
        tenant_access_token = auth_resp.json()["tenant_access_token"]
        headers = {"Authorization": f"Bearer {tenant_access_token}"}
        url = f"https://open.feishu.cn/open-apis/board/v1/whiteboards/{whiteboard_id}/nodes"
        data = self.request_with_retry("GET", url, headers=headers)
        return data.get('data', {}).get('nodes',{})





    def get_table_content(self, doc_token, block_id):
        processed_block = []
        # 获取 tenant_access_token
        auth_url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        auth_resp = requests.post(auth_url, json={
            "app_id": self.app_id,
            "app_secret": self.app_secret
        })
        tenant_access_token = auth_resp.json()["tenant_access_token"]

        headers = {"Authorization": f"Bearer {tenant_access_token}"}

        # 获取表格 children（每个 cell）
        url = f"https://open.feishu.cn/open-apis/docx/v1/documents/{doc_token}/blocks/{block_id}/children"
        data = self.request_with_retry("GET", url, headers=headers)
        # resp = requests.get(url, headers=headers)
        if not isinstance(data, dict) or data.get("code") != 0:
            raise Exception(f"请求飞书 API 失败: {data}")
        # if resp.status_code != 200:
        #     raise Exception(f"请求飞书 API 失败: {resp.text}")
        # data = resp.json()

        cell_blocks = data["data"]["items"]
        # 先获取 row/col 信息
        # ⚠️ 注意：这里 col_count/row_count 需要额外查一次 table block
        table_url = f"https://open.feishu.cn/open-apis/docx/v1/documents/{doc_token}/blocks/{block_id}"
        table_data = self.request_with_retry("GET", table_url, headers=headers)
        if not isinstance(table_data, dict) or table_data.get("code") != 0:
            raise Exception(f"获取 table block 失败: {table_data}")
        # table_resp = requests.get(table_url, headers=headers).json()
        # table = table_resp["data"]["block"]["table"]
        table = table_data["data"]["block"]["table"]
        row_count = table["property"]["row_size"]
        col_count = table["property"]["column_size"]
        grid = [["" for _ in range(col_count)] for _ in range(row_count)]
        for idx, cell_block in enumerate(cell_blocks):
            processed_block.append(cell_block['block_id'])
            row = idx // col_count
            col = idx % col_count
            text_parts = []
            for child_id in cell_block.get("children", []):
                processed_block.append(child_id)
                child_url = f"https://open.feishu.cn/open-apis/docx/v1/documents/{doc_token}/blocks/{child_id}"
                # 有可能是这里除了问题
                child_resp = self.request_with_retry("GET", child_url, headers)
                child_block = child_resp["data"]["block"]
                if child_block["block_type"] > 1 and child_block["block_type"] < 14:
                    for e in child_block[map_dict[child_block["block_type"]]].get("elements", []):
                        if "text_run" in e:
                            text_parts.append(e["text_run"]["content"])    
                            text_parts.append('\n')
                elif child_block["block_type"] == 14:
                    text_parts.append("<code>")
                    text_parts.append(child_block['block_id'])
                    text_parts.append("<code_end>")
            grid[row][col] = "".join(text_parts).strip()
        if not grid:
            return ""
        markdown = "| " + " | ".join(grid[0]) + " |\n"
        markdown += "| " + " | ".join(["---"] * col_count) + " |\n"
        for row in grid[1:]:
            markdown += "| " + " | ".join(row) + " |\n"
        return markdown



    def get_user_name(self, user_id):
        token = self.get_tenant_token(self.app_id, self.app_secret)
        url = f"https://open.feishu.cn/open-apis/contact/v3/users/{user_id}"
        headers = {"Authorization": f"Bearer {token}"}

        data = self.request_with_retry("GET", url, headers=headers, max_retries=5)
        # request_with_retry 可能返回 parsed json(dict) 或者原 resp（在异常分支）
        if not isinstance(data, dict):
            try:
                data = data.json()
            except Exception as e:
                raise RuntimeError(f"无法解析响应为 JSON: {e}")

        # 常见返回结构： {"code":0, "msg":"ok", "data": { "user": {...} } }
        if data.get("code") != 0:
            raise RuntimeError(f"获取用户信息失败: {data}")

        user_obj = data.get("data", {}).get("user") or data.get("data")
        if not user_obj:
            raise RuntimeError(f"响应中未包含 user 字段: {data}")

        # 兼容不同字段名
        name = (
            user_obj.get("name")
            or user_obj.get("employee_name")
            or user_obj.get("zh_name")
            or user_obj.get("en_name")
            or user_obj.get("email")
        )
        return name


    def download_whiteboard_png(self, whiteboard_token: str) -> str:
        token = self.get_tenant_token(self.app_id, self.app_secret)
        headers = {"Authorization": f"Bearer {token}"}

        url = f"https://open.feishu.cn/open-apis/board/v1/whiteboards/{whiteboard_token}/download_as_image?type=png"
        # type 支持 png/jpeg/gif/svg，默认我们要 png 

        resp = self.request_with_retry("GET", url, headers=headers)
        if resp.status_code != 200:
            raise Exception(f"下载白板图片失败: status_code={resp.status_code}, text={getattr(resp,'text', '')}")

        return resp.content


    def extract_flowchart_from_block(self, record_str):
        # record_str = block["add_ons"]["record"]  # 这是字符串
        record = json.loads(record_str)          # 转成 dict

        raw_data = record.get("data", "")

        # 清洗无关字符
        cleaned = (
            raw_data
            .replace("\\u003e", ">")
            .replace("\xa0", " ")
            .replace("\\n", "\n")
        )

        return cleaned.strip()

    def get_document_info(self, document_id):
        token = self.get_tenant_token(self.app_id, self.app_secret)
        url = f"https://open.feishu.cn/open-apis/docx/v1/documents/{document_id}"
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()


    # def extract_flowchart_from_doc_blocks(self, board_token):
    #     import pdb
    #     pdb.set_trace()
    #     token = self.get_tenant_token(self.app_id, self.app_secret)
    #     headers = {"Authorization": f"Bearer {token}"}
    #     url = f"https://open.feishu.cn/open-apis/drive/explorer/v2/file/blocks?file_token={board_token}"

    #     resp = requests.get(url, headers=headers)
    #     if resp.status_code == 404:
    #         # 尝试旧接口
    #         url = f"https://open.feishu.cn/open-apis/drive/v1/files/{board_token}/raw/blocks"
    #         resp = requests.get(url, headers=headers)

    #     # url = f"https://open.feishu.cn/open-apis/mindnotes/v2/mindnotes/{board_token}/content"
    #     # headers = {"Authorization": f"Bearer {token}"}

    #     # resp = requests.get(url, headers=headers)
    #     import pdb
    #     pdb.set_trace()
    #     print(resp.status_code, resp.text)

    #     # # 3. 调用 Board API 获取流程图所有元素
    #     # url = f"https://open.feishu.cn/open-apis/board/v1/boards/{board_token}/blocks"
    #     # headers = {"Authorization": f"Bearer {token}"}

    #     # resp = requests.get(url, headers=headers)
    #     resp.raise_for_status()

    #     board_items = resp.json().get("data", {}).get("items", [])

    #     # 4. 解析流程图节点与连线
    #     nodes = {}
    #     edges = []

    #     for item in board_items:
    #         t = item.get("type")

    #         # 节点：矩形、圆形、菱形等都属于 shape
    #         if t == "shape":
    #             node_id = item.get("id")
    #             text = item.get("shape", {}).get("text", "").strip()
    #             nodes[node_id] = text

    #         # 连线
    #         elif t == "line":
    #             line = item.get("line", {})
    #             edges.append({
    #                 "from": line.get("source"),
    #                 "to": line.get("target"),
    #                 "text": line.get("text", "").strip()
    #             })

    #     # 5. 组织 RAG 文本输出
    #     output_lines = []

    #     output_lines.append("流程图内容：")

    #     # 节点
    #     for nid, text in nodes.items():
    #         output_lines.append(f"- 节点 {nid}: {text}")

    #     # 连线
    #     for e in edges:
    #         start = nodes.get(e["from"], e["from"])
    #         end = nodes.get(e["to"], e["to"])
    #         edge_note = e["text"]
    #         if edge_note:
    #             output_lines.append(f"  {start} → {end}    (备注: {edge_note})")
    #         else:
    #             output_lines.append(f"  {start} → {end}")

    #     return "\n".join(output_lines)

    # def parse_flowchart(self, board_data):
    #     nodes = {}
    #     edges = []

    #     for item in board_data["items"]:
    #         t = item["type"]

    #         if t == "shape":  # 流程节点
    #             nodes[item["id"]] = item["shape"]["text"]

    #         elif t == "line":  # 连线
    #             edges.append({
    #                 "from": item["line"]["source"],
    #                 "to": item["line"]["target"],
    #                 "text": item["line"].get("text", "")
    #             })

    #     return nodes, edges



    def get_image_content(self, doc_token: str, block_id: str):
        token = self.get_tenant_token(self.app_id, self.app_secret)
        headers = {"Authorization": f"Bearer {token}"}
        # Step 1: 获取块信息
        block_url = f"https://open.feishu.cn/open-apis/docx/v1/documents/{doc_token}/blocks/{block_id}"
        data = self.request_with_retry("GET", block_url, headers=headers)
        if not isinstance(data, dict) or data.get("code") != 0:
            raise Exception(f"获取图片 block 信息失败: {data}")
        block = data["data"]["block"]
        file_token = block["image"]["token"]
        download_url = f"https://open.feishu.cn/open-apis/drive/v1/medias/{file_token}/download"
        resp = self.request_with_retry("GET", download_url, headers=headers)
        if resp.status_code != 200:
            raise Exception(f"下载图片失败")
        return resp.content

    def get_code_content(self, document_id: str, block_id: str):
        token = self.get_tenant_token(self.app_id, self.app_secret)
        url = f"https://open.feishu.cn/open-apis/docx/v1/documents/{document_id}/blocks/{block_id}"
        headers = {"Authorization": f"Bearer {token}"}

        data = self.request_with_retry("GET", url, headers=headers)

        if not isinstance(data, dict) or data.get("code") != 0:
            raise Exception(f"获取代码块失败: {data}")

        code = ""
        for element in data['data']['block']['code']['elements']:
            code += element['text_run']['content']
        return code


    def get_tenant_token(self, app_id, app_secret):
        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        resp = requests.post(url, json={"app_id": app_id, "app_secret": app_secret})
        resp.raise_for_status()
        j = resp.json()
        if j.get("code") != 0:
            raise RuntimeError("获取 tenant_access_token 失败: " + str(j))
        return j["tenant_access_token"]

    def get_faq_content(self, faq_token, sheet_id, query_name, answer_name):
        node_info, _ = self.get_node(faq_token)
        faq_obj_token = node_info['obj_token']
        token = self.get_tenant_token(self.app_id, self.app_secret)
        headers = {"Authorization": f"Bearer {token}"}
        meta_url = f"https://open.feishu.cn/open-apis/sheet/v2/spreadsheets/{faq_obj_token}/metainfo"
        # meta_resp = requests.get(meta_url, headers=headers)
        # meta_resp.raise_for_status()
        # meta = meta_resp.json()
        meta = self.request_with_retry("GET", meta_url, headers=headers)
        if not isinstance(meta, dict) or meta.get("code") != 0:
            raise RuntimeError("metainfo 请求失败: " + str(meta))
        title = meta['data']['properties']['title']
        # if meta.get("code") != 0:
        #     raise RuntimeError("metainfo 请求失败: " + str(meta))

        if not sheet_id:
            sheet_id = meta['data']['sheets'][0]['sheetId']
            row_sum = meta['data']['sheets'][0]['rowCount']
        else:
            for i in range(len(meta['data']['sheets'])):
                if meta['data']['sheets'][i]['sheetId'] == sheet_id:
                    row_sum = meta['data']['sheets'][i]['rowCount']
                    break

        # 读区表头
        range_str = f"{sheet_id}!A1:Z1"
        val_url = f"https://open.feishu.cn/open-apis/sheet/v2/spreadsheets/{faq_obj_token}/values/{range_str}"

        # r = requests.get(val_url, headers=headers)
        # r.raise_for_status()
        # j = r.json()
        j = self.request_with_retry("GET", val_url, headers=headers)
        # if j.get("code") != 0:
        #     print(f"读取 {sheet_id} 失败:", j)
        #     values = []
        # else:
        #     values = j.get("data", {}).get("valueRange", {}).get("values", [])

        if not isinstance(j, dict) or j.get("code") != 0:
            print(f"读取 {sheet_id} 失败:", j)
            values = []
        else:
            values = j.get("data", {}).get("valueRange", {}).get("values", [])


        # 找到相应的列名字
        answer_col = -1
        query_col = -1
        for col_idx in range(len(values[0])):
            # cur_col_name = str(values[0][col_idx]) if values[0][col_idx] 
            if str(values[0][col_idx]) == query_name:
                query_col = chr(ord('A') + col_idx)
            elif str(values[0][col_idx]) == answer_name:
                answer_col = chr(ord('A') + col_idx)
            if query_col != -1 and answer_col != -1:
                break
        if answer_col == -1 or query_col == -1:
            raise RuntimeError("faq 文档qa列名错误，找不到对应列名") 
        query_range_str = f"{sheet_id}!"+query_col+"1:"+query_col+str(row_sum)
        answer_range_str = f"{sheet_id}!"+answer_col+"1:"+answer_col+str(row_sum)
        query_val_url = f"https://open.feishu.cn/open-apis/sheet/v2/spreadsheets/{faq_obj_token}/values/{query_range_str}"
        answer_val_url = f"https://open.feishu.cn/open-apis/sheet/v2/spreadsheets/{faq_obj_token}/values/{answer_range_str}"
        # r_q = requests.get(query_val_url, headers=headers)
        # r_q.raise_for_status()
        # j = r_q.json()
        j = self.request_with_retry("GET", query_val_url, headers=headers)


        if j.get("code") != 0:
            print(f"读取 {sheet_id} 失败:", j)
            query_values = []
        else:
            query_values = j.get("data", {}).get("valueRange", {}).get("values", [])

        # r_a = requests.get(answer_val_url, headers=headers)
        # r_a.raise_for_status()
        # j = r_a.json()
        j = self.request_with_retry("GET", answer_val_url, headers=headers)

        if j.get("code") != 0:
            print(f"读取 {sheet_id} 失败:", j)
            answer_values = []
        else:
            answer_values = j.get("data", {}).get("valueRange", {}).get("values", [])

        # 转字符串

        for m in range(1,len(query_values)):
            query_str = ""
            if not isinstance(query_values[m][0], list):
                continue
            if not query_values[m][0]:
                continue
            for n in range(len(query_values[m][0])):
                if query_values[m][0][n]['type'] == 'text':
                    query_str += query_values[m][0][n]['text']
            query_values[m][0] = query_str

        for m in range(1,len(answer_values)):
            answer_str = ""
            if not isinstance(answer_values[m][0], list):
                continue
            if not answer_values[m][0]:
                continue
            for n in range(len(answer_values[m][0])):
                if answer_values[m][0][n]['type'] == 'text':
                    answer_str += answer_values[m][0][n]['text']
            answer_values[m][0] = answer_str
        return query_values, answer_values, title

    # 这个要接入限频
    def get_block_content(self, wiki_token, page_token=None):
        content_list = []
        if not page_token:
            request: ListDocumentBlockRequest = ListDocumentBlockRequest.builder() \
                .document_id(wiki_token) \
                .page_size(500) \
                .document_revision_id(-1) \
                .build()
        else:
            request: ListDocumentBlockRequest = ListDocumentBlockRequest.builder() \
                .document_id(wiki_token) \
                .page_size(500) \
                .page_token(page_token) \
                .document_revision_id(-1) \
                .build()
    
        response: ListDocumentBlockResponse = self.client.docx.v1.document_block.list(request)
        # 也有可能是这个接口除了问题
        if not response.success():
            lark.logger.error(
                f"client.docx.v1.document_block.list failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
            return
        ans = json.loads(lark.JSON.marshal(response.data))
        content_list += ans['items']
        if ans['has_more']:
            other_page_list = self.get_block_content(wiki_token, page_token=ans['page_token'])
            content_list += other_page_list
        return content_list

    def search_subnode_info(self, space_id, parent_node_token, page_token=None):
        node_list = []

        if not page_token:

            request: ListSpaceNodeRequest = ListSpaceNodeRequest.builder() \
                .space_id(space_id) \
                .parent_node_token(parent_node_token) \
                .build()
        else:
            request: ListSpaceNodeRequest = ListSpaceNodeRequest.builder() \
                .space_id(space_id) \
                .page_token(page_token) \
                .parent_node_token(parent_node_token) \
                .build()
        # 发起请求
        response: ListSpaceNodeResponse = self.client.wiki.v2.space_node.list(request)

        # 处理失败返回
        if not response.success():
            lark.logger.error(
                f"client.wiki.v2.space_node.list failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
            return

        node_info = json.loads(lark.JSON.marshal(response.data))
        node_list += node_info['items']
        if node_info['has_more']:
            sub_node_list = self.search_subnode_info(space_id, parent_node_token, page_token=node_info['page_token'])
            node_list += sub_node_list
        return node_list
    

    def get_node(self, wiki_token):

        # 构造请求对象
        request: GetNodeSpaceRequest = GetNodeSpaceRequest.builder() \
            .token(wiki_token) \
            .obj_type("wiki") \
            .build()

        # 发起请求
        response: GetNodeSpaceResponse = self.client.wiki.v2.space.get_node(request)

        # 处理失败返回
        if not response.success():
            node_info = None
            status = False
            lark.logger.error(
                f"client.wiki.v2.space.get_node failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
            return node_info, status
        node_info = json.loads(lark.JSON.marshal(response.data))['node']
        status = True
        return node_info, status