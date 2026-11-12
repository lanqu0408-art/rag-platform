from haystack.components.preprocessors import DocumentSplitter, RecursiveDocumentSplitter
from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
import re
from more_itertools import windowed
from haystack import Document, component, logging
from haystack.components.preprocessors.sentence_tokenizer import Language, SentenceSplitter, nltk_imports
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.utils import deserialize_callable, serialize_callable
from haystack import Document, component, logging
from ...external_api.aizt_api import aizt_chat
logger = logging.getLogger(__name__)
import time
# mapping of split by character, 'function' and 'sentence' don't split by character
_CHARACTER_SPLIT_BY_MAPPING = {
    "page": "\f", "passage": "\n\n", "period": ".", "word": " ", "line": "\n"}

special_tokens = {
    '<file_path>',
    '<file_path_end>',
    '<chunk_info>',
    '<chunk_info_end>',
    '<image>',
    '<image_end>',
    '<code>',
    '<code_end>',
    '<tab_s>',
    '<tab_e>',
    '<chart_s>',
    '<chart_e>',
    '<ocr_s>',
    '<ocr_e>'
}

special_pairs = [('<file_path>', '<file_path_end>'), 
                 ('<chunk_info>', '<chunk_info_end>'),
                 ('<image>', '<image_end>'),
                 ('<code>', '<code_end>'),
                 ('<tab_s>', '<tab_e>'),
                 ('<chart_s>', '<chart_e>'),
                 ('<ocr_s>', '<ocr_e>')]

@component
class chinese_RecrucuitSplitter(RecursiveDocumentSplitter):
    def __init__(self, *args, **kwargs):
        super(chinese_RecrucuitSplitter, self).__init__(*args, 
                                                        split_length = kwargs['split_length'],
                                                        # split_overlap = kwargs['split_overlap'],
                                                        split_overlap = 0,
                                                        split_unit = "char",
                                                        separators = kwargs['separators'])
        # self.split_overlap = kwargs['split_overlap']
        self.split_overlap = 0
        self.chinese_tokenizer_coarse = kwargs['chinese_tokenizer_coarse']
        self.chinese_tokenizer_fine = kwargs['chinese_tokenizer_fine']
        self.model = kwargs['model_chat']
        self.tokenizer = kwargs['chat_tokenizer']
        self.paragraphAugment = kwargs['paragraphAugment']
        self.documentAugment =  kwargs['documentAugment']
        self.keywordCount = kwargs['keywordCount']


    def merge_special_sign(self, tokens, token_pairs=special_pairs):
        merged = []
        in_special = None  # 标记当前在哪个特殊区间内（存储起始标记）
        buffer = []

        for token in tokens:
            # 检查是否进入特殊区间
            for start, end in token_pairs:
                if token == start:
                    in_special = start
                    buffer.append(token)
                    break
            else:
                # 检查是否退出特殊区间
                if in_special is not None:
                    # 找到对应的结束标记
                    end_token = next(end for start, end in token_pairs if start == in_special)
                    if token == end_token:
                        buffer.append(token)
                        merged.append(''.join(buffer))
                        buffer = []
                        in_special = None
                    else:
                        buffer.append(token)
                else:
                    merged.append(token)
        # 处理未闭合的特殊标记
        if buffer:
            merged.extend(buffer)
        return merged

    def add_addition_info(self, chunk):
        text_splits_add_chunk_info = []  # 用于最后返回
        file_path_info = chunk[0].split('<file_path_end>')[0].split('<file_path>')[1]


        if '/' in file_path_info: 
            pre_chunk_info = file_path_info.split('/')[1]
        else:
            pre_chunk_info = file_path_info
        # chunk_info 从文件名开始
        for sub_content in chunk:
            pre_info = ''
            if '<file_path>' in sub_content: # 证明是第一个
                sub_content = sub_content.split('<file_path_end>')[1]
            pre_info = '<file_path>' + file_path_info + '<file_path_end>'
            # 同时维护pre_chunk_info更新
            if self.paragraphAugment:
                if "<chunk_info>" not in sub_content: # 完全是上一个chunk，直接拼接
                    pre_info += '<chunk_info>' + pre_chunk_info + '<chunk_info_end>'
                else:
                    start_pos =  [match.start () for match in re.finditer ('<chunk_info>', sub_content)]
                    end_pos = [match.end() for match in re.finditer ('<chunk_info_end>', sub_content)]
                    pairs = []
                    for i in range(len(start_pos)):
                        pairs.append([start_pos[i], end_pos[i]])
                    
                    if sub_content.startswith('<chunk_info>'):
                        cur_chunk_info = ''
                    else:
                        cur_chunk_info = pre_chunk_info
                    # for s, e in pairs:
                    #     chunk_info += ("&" + sub_content[s:e].split('<chunk_info_end>')[0].split('<chunk_info>')[1])
                    cur_chunk_info = '<chunk_info>' + cur_chunk_info + '<chunk_info_end>'
                    
                    # chunk_info = sub_content[start_pos[-1]:end_pos[-1]].split('<chunk_info_end>')[0].split('<chunk_info>')[1]
                    # for s, e in pairs[::-1]:
                    #     if e >= len(sub_content):
                    #         e = len(sub_content) - 1
                    #     sub_content = sub_content[:s] + sub_content[e+1:]
                    pre_chunk_info = sub_content[start_pos[-1]:end_pos[-1]].split('<chunk_info_end>')[0].split('<chunk_info>')[1]
                    pre_info += cur_chunk_info
            else:
                pre_info += '<chunk_info> '  + '<chunk_info_end>'

            if self.keywordCount != 0:
                # 判断是否添加关键字
                prompt = "根据下面这段文字，提取" + str(self.keywordCount) + "个关键词，只返回相应个数的关键词，不用返回其他内容。文字内容为"
                prompt += sub_content
                time_start = time.time()
                response = aizt_chat(prompt, model_type="qwen3")
                time_end = time.time() 
                logger.info('time：提取关键词用时: %s' % (time_end - time_start ))
                # messages = [
                #     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Your task is to extract keywords accurately."},
                #     {"role": "user", "content": prompt}
                # ]
                # # 生成聊天模板
                # text = self.tokenizer.apply_chat_template(
                #     messages,
                #     tokenize=False,
                #     add_generation_prompt=True
                # )

                # # 转换为模型输入
                # model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

                # # 生成回复，增加更多控制参数
                # generated_ids = self.model.generate(
                #     **model_inputs,
                #     max_new_tokens=512
                # )

                # # 提取生成的部分（排除输入部分）
                # generated_ids = [
                #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                # ]

                # # 解码得到结果

                # response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            else:
                response = " "

            pre_info += "<key_word>" + response + "<key_word_end>"
            sub_content = pre_info + sub_content
            text_splits_add_chunk_info.append(sub_content)
        return text_splits_add_chunk_info

    def _run_one(self, doc: Document) -> List[Document]:
        chunks = self._chunk_text(doc.content)
        chunks = self.add_addition_info(chunks)
        chunks = chunks[:-1] if len(chunks[-1]) == 0 else chunks  # remove last empty chunk if it exists
        current_position = 0
        current_page = 1

        new_docs = []

        for split_nr, chunk in enumerate(chunks):
            meta = deepcopy(doc.meta)
            meta["parent_id"] = doc.id
            meta["split_id"] = split_nr
            meta["split_idx_start"] = current_position
            meta["_split_overlap"] = [] if self.split_overlap > 0 else None
            new_doc = Document(content=chunk, meta=meta)

            # add overlap information to the previous and current doc
            if split_nr > 0 and self.split_overlap > 0:
                self._add_overlap_info(current_position, new_doc, new_docs)

            # count page breaks in the chunk
            current_page += chunk.count("\f")

            # if there are consecutive page breaks at the end with no more text, adjust the page number
            # e.g: "text\f\f\f" -> 3 page breaks, but current_page should be 1
            consecutive_page_breaks = len(chunk) - len(chunk.rstrip("\f"))

            if consecutive_page_breaks > 0:
                new_doc.meta["page_number"] = current_page - consecutive_page_breaks
            else:
                new_doc.meta["page_number"] = current_page

            # keep the new chunk doc and update the current position
            new_docs.append(new_doc)
            current_position += len(chunk) - (self.split_overlap if split_nr < len(chunks) - 1 else 0)

        return new_docs

    # def _get_overlap(self, overlapped_chunks: list[str]) -> tuple[str, str]:
    #     """Get the previous overlapped chunk instead of the original chunk."""
    #     prev_chunk = overlapped_chunks[-1]
    #     overlap_start = max(0, self._chunk_length(prev_chunk) - self.split_overlap)

    #     if self.split_units == "word":
    #         word_chunks = prev_chunk.split()
    #         overlap = " ".join(word_chunks[overlap_start:])
    #     elif self.split_units == "token":
    #         # For token-based splitting, handle overlap at token level
    #         # at this point we know that the tokenizer is already initialized
    #         tokens = self.tiktoken_tokenizer.encode(prev_chunk)  # type: ignore
    #         overlap_tokens = tokens[overlap_start:]
    #         overlap = self.tiktoken_tokenizer.decode(overlap_tokens)  # type: ignore
    #     else:  # char
    #         overlap = prev_chunk[overlap_start:]

    #     return overlap, prev_chunk


    # def _apply_overlap(self, chunks: list[str]) -> list[str]:
    #     """
    #     Applies an overlap between consecutive chunks if the chunk_overlap attribute is greater than zero.

    #     Works for both word- and character-level splitting. It trims the last chunk if it exceeds the split_length and
    #     adds the trimmed content to the next chunk. If the last chunk is still too long after trimming, it splits it
    #     and adds the first chunk to the list. This process continues until the last chunk is within the split_length.

    #     :param chunks: A list of text chunks.
    #     :returns:
    #         A list of text chunks with the overlap applied.
    #     """
    #     overlapped_chunks: list[str] = []

    #     for idx, chunk in enumerate(chunks):
    #         if idx == 0:
    #             overlapped_chunks.append(chunk)
    #             continue

    #         # get the overlap between the current and previous chunk
    #         overlap, prev_chunk = self._get_overlap(overlapped_chunks)
    #         if overlap == prev_chunk:
    #             logger.warning(
    #                 "Overlap is the same as the previous chunk. "
    #                 "Consider increasing the `split_length` parameter or decreasing the `split_overlap` parameter."
    #             )

    #         current_chunk = self._create_chunk_starting_with_overlap(chunk, overlap)

    #         # if this new chunk exceeds 'split_length', trim it and move the remaining text to the next chunk
    #         # if this is the last chunk, another new chunk will contain the trimmed text preceded by the overlap
    #         # of the last chunk
    #         if self._chunk_length(current_chunk) > self.split_length:
    #             current_chunk, remaining_text = self._split_chunk(current_chunk)
    #             if idx < len(chunks) - 1:
    #                 if self.split_units == "word":
    #                     chunks[idx + 1] = remaining_text + " " + chunks[idx + 1]
    #                 elif self.split_units == "token":
    #                     # For token-based splitting, combine at token level
    #                     # at this point we know that the tokenizer is already initialized
    #                     remaining_tokens = self.tiktoken_tokenizer.encode(remaining_text)  # type: ignore
    #                     next_chunk_tokens = self.tiktoken_tokenizer.encode(chunks[idx + 1])  # type: ignore
    #                     chunks[idx + 1] = self.tiktoken_tokenizer.decode(remaining_tokens + next_chunk_tokens)  # type: ignore
    #                 else:  # char
    #                     chunks[idx + 1] = remaining_text + chunks[idx + 1]
    #             elif remaining_text:
    #                 # create a new chunk with the trimmed text preceded by the overlap of the last chunk
    #                 overlapped_chunks.append(current_chunk)
    #                 chunk = remaining_text
    #                 overlap, _ = self._get_overlap(overlapped_chunks)
    #                 current_chunk = self._create_chunk_starting_with_overlap(chunk, overlap)

    #         overlapped_chunks.append(current_chunk)

    #         # it can still be that the new last chunk exceeds the 'split_length'
    #         # continue splitting until the last chunk is within 'split_length'
    #         if idx == len(chunks) - 1 and self._chunk_length(current_chunk) > self.split_length:
    #             last_chunk = overlapped_chunks.pop()
    #             first_chunk, remaining_chunk = self._split_chunk(last_chunk)
    #             overlapped_chunks.append(first_chunk)

    #             while remaining_chunk:
    #                 # combine overlap with remaining chunk
    #                 overlap, _ = self._get_overlap(overlapped_chunks)
    #                 current = self._create_chunk_starting_with_overlap(remaining_chunk, overlap)

    #                 # if it fits within split_length we are done
    #                 if self._chunk_length(current) <= self.split_length:
    #                     overlapped_chunks.append(current)
    #                     break

    #                 # otherwise split it again
    #                 first_chunk, remaining_chunk = self._split_chunk(current)
    #                 overlapped_chunks.append(first_chunk)

    #     return overlapped_chunks


    # def _create_chunk_starting_with_overlap(self, chunk, overlap):
    #     if self.split_units == "word":
    #         current_chunk = overlap + " " + chunk
    #     elif self.split_units == "token":
    #         # For token-based splitting, combine at token level
    #         # at this point we know that the tokenizer is already initialized
    #         overlap_tokens = self.tiktoken_tokenizer.encode(overlap)  # type: ignore
    #         chunk_tokens = self.tiktoken_tokenizer.encode(chunk)  # type: ignore
    #         current_chunk = self.tiktoken_tokenizer.decode(overlap_tokens + chunk_tokens)  # type: ignore
    #     else:  # char
    #         current_chunk = overlap + chunk
    #     return current_chunk


    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Split a list of documents into documents with smaller chunks of text.

        :param documents: List of Documents to split.
        :returns:
            A dictionary containing a key "documents" with a List of Documents with smaller chunks of text corresponding
            to the input documents.

        :raises RuntimeError: If the component wasn't warmed up but requires it for sentence splitting or tokenization.
        """
        if not self._is_warmed_up and ("sentence" in self.separators or self.split_units == "token"):
            raise RuntimeError(
                "The component RecursiveDocumentSplitter wasn't warmed up but requires it "
                "for sentence splitting or tokenization. Call 'warm_up()' before calling 'run()'."
            )
        docs = []
        for doc in documents:
            if not doc.content or doc.content == "":
                logger.warning("Document ID {doc_id} has an empty content. Skipping this document.", doc_id=doc.id)
                continue
            docs.extend(self._run_one(doc))
        return {"documents": docs}
