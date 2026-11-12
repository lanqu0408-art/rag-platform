# '''
# haystack-ai == 2.12.1

# '''


from haystack.components.preprocessors import DocumentSplitter
from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
import re
from more_itertools import windowed
from haystack import Document, component, logging
from haystack.components.preprocessors.sentence_tokenizer import Language, SentenceSplitter, nltk_imports
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.utils import deserialize_callable, serialize_callable

# from ...external_api.aizt_api import aizt_chat
import time
from ...external_api.aizt_api import aizt_chat
from core.hanlp_runtime import ensure_hanlp_env, get_sentence_splitter
logger = logging.getLogger(__name__)
ensure_hanlp_env()

# mapping of split by character, 'function' and 'sentence' don't split by character
_CHARACTER_SPLIT_BY_MAPPING = {
    "page": "\f", "passage": "\n\n", "period": ".", "word": " ", "line": "\n"}
# chinese_tokenizer_coarse = hanlp.load(
#     hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
# chinese_tokenizer_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
# 这里补special_tokens
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

# 将特殊标记添加到自定义词典，确保不被拆分
# 对于Transformer-based模型，使用add_words方法
# for special_token in special_tokens:

# 还是会被切碎


# .dict_combine(special_tokens)



@component
class chinese_DocumentSpliter(DocumentSplitter):
    def __init__(self, *args, **kwargs):
        super(chinese_DocumentSpliter, self).__init__(*args, 
                                                      split_length = kwargs['split_length'],
                                                      split_overlap = 0,
                                                      respect_sentence_boundary = kwargs['respectSentenceBoundary'])
        self.language = 'zh'
        self.particle_size = "coarse"
        self.chinese_tokenizer_coarse = kwargs['chinese_tokenizer_coarse']
        self.chinese_tokenizer_fine = kwargs['chinese_tokenizer_fine']
        self.chinese_tokenizer_coarse.dict_force = special_tokens
        self.splitThreshold = kwargs['splitThreshold']
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

    def _split_by_character(self, doc) -> List[Document]:
        split_at = _CHARACTER_SPLIT_BY_MAPPING[self.split_by]
        if self.language == 'zh' and self.particle_size == "coarse":  # 只会走这种
            units = self.chinese_tokenizer_coarse(doc.content) 
            units = self.merge_special_sign(units)
        if self.language == 'zh' and self.particle_size == "fine":
            units = self.chinese_tokenizer_fine(doc.content)
        if self.language == 'en':
            units = doc.content.split(split_at)
            # Add the delimiter back to all units except the last one
        for i in range(len(units) - 1):
            units[i] += split_at
        text_splits, splits_pages, splits_start_idxs, text_splits_with_info = self._concatenate_units(
            units, self.split_length, self.split_overlap, self.split_threshold
        )
        metadata = deepcopy(doc.meta)
        metadata["source_id"] = doc.id
        return self._create_docs_from_splits(
            text_splits=text_splits_with_info, splits_pages=splits_pages, splits_start_idxs=splits_start_idxs, meta=metadata
        )

    # 中文这么切就好，但是如何添加白名单
    # 定义一个函数用于处理中文分句
    def chinese_sentence_split(self, text: str) -> list:
        # 分句
        split_sent = get_sentence_splitter()
        sentences = split_sent(text)

        # 整理格式
        results = []
        start = 0
        for sentence in sentences:
            start = text.find(sentence, start)
            end = start + len(sentence)
            results.append({
                'sentence': sentence + '\n',
                'start': start,
                'end': end
            })
            start = end

        return results

    def _split_document(self, doc: Document) -> List[Document]:

        if self.split_by == "sentence" or self.respect_sentence_boundary:
            return self._split_by_nltk_sentence(doc)
            # 如果尊重句子边界的话，会走nltk_sen

        return self._split_by_character(doc)

    @staticmethod
    def _concatenate_sentences_based_on_word_amount(
        self, sentences: List[str], split_length: int, split_overlap: int, language: str, particle_size: str
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        Groups the sentences into chunks of `split_length` words while respecting sentence boundaries.

        This function is only used when splitting by `word` and `respect_sentence_boundary` is set to `True`, i.e.:
        with NLTK sentence tokenizer.

        :param sentences: The list of sentences to split.
        :param split_length: The maximum number of words in each split.
        :param split_overlap: The number of overlapping words in each split.
        :returns: A tuple containing the concatenated sentences, the start page numbers, and the start indices.
        """
        # chunk information
        chunk_word_count = 0
        chunk_starting_page_number = 1
        chunk_start_idx = 0
        current_chunk: List[str] = []
        # output lists
        split_start_page_numbers = []
        list_of_splits: List[List[str]] = []
        split_start_indices = []
        # chinese_tokenizer_coarse = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        # chinese_tokenizer_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
        for sentence_idx, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            if language == 'zh' and particle_size == "coarse":
                chunk_word_count += len(self.chinese_tokenizer_coarse(sentence))
                next_sentence_word_count = (
                    len(self.chinese_tokenizer_coarse(
                        sentences[sentence_idx + 1])) if sentence_idx < len(sentences) - 1 else 0
                )
            if language == 'zh' and particle_size == "fine":
                chunk_word_count += len(self.chinese_tokenizer_fine(sentence))
                next_sentence_word_count = (
                    len(self.chinese_tokenizer_fine(
                        sentences[sentence_idx + 1])) if sentence_idx < len(sentences) - 1 else 0
                )

            # Number of words in the current chunk plus the next sentence is larger than the split_length,
            # or we reached the last sentence
            if (chunk_word_count + next_sentence_word_count) > split_length or sentence_idx == len(sentences) - 1:
                #  Save current chunk and start a new one
                list_of_splits.append(current_chunk)
                split_start_page_numbers.append(chunk_starting_page_number)
                split_start_indices.append(chunk_start_idx)

                # Get the number of sentences that overlap with the next chunk
                num_sentences_to_keep = chinese_DocumentSpliter._number_of_sentences_to_keep(
                    sentences=current_chunk, split_length=split_length, split_overlap=split_overlap, language=language, particle_size=particle_size
                )
                # Set up information for the new chunk
                if num_sentences_to_keep > 0:
                    # Processed sentences are the ones that are not overlapping with the next chunk
                    processed_sentences = current_chunk[:-
                                                        num_sentences_to_keep]
                    chunk_starting_page_number += sum(sent.count("\f")
                                                      for sent in processed_sentences)
                    chunk_start_idx += len("".join(processed_sentences))
                    # Next chunk starts with the sentences that were overlapping with the previous chunk
                    current_chunk = current_chunk[-num_sentences_to_keep:]
                    chunk_word_count = sum(len(s.split())
                                           for s in current_chunk)
                else:
                    # Here processed_sentences is the same as current_chunk since there is no overlap
                    chunk_starting_page_number += sum(sent.count("\f")
                                                      for sent in current_chunk)
                    chunk_start_idx += len("".join(current_chunk))
                    current_chunk = []
                    chunk_word_count = 0

        # Concatenate the sentences together within each split
        text_splits = []
        for split in list_of_splits:
            text = "".join(split)
            if len(text) > 0:
                text_splits.append(text)

        return text_splits, split_start_page_numbers, split_start_indices


    # 增加中文句子切分，通过languge == "zh"，进行启用
    def _split_by_nltk_sentence(self, doc: Document) -> List[Document]:
        split_docs = []
        if self.language == 'zh':
            result = self.chinese_sentence_split(doc.content)
        if self.language == 'en':
            result = self.sentence_splitter.split_sentences(
                doc.content)  # type: ignore # None check is done in run()

        units = [sentence["sentence"] for sentence in result]

        if self.respect_sentence_boundary:
            text_splits, splits_pages, splits_start_idxs = self._concatenate_sentences_based_on_word_amount(
                sentences=units, split_length=self.split_length, split_overlap=self.split_overlap, language=self.language,
                particle_size=self.particle_size)
        else:
            text_splits, splits_pages, splits_start_idxs, text_split_with_info = self._concatenate_units(
                elements=units,
                split_length=self.split_length,
                split_overlap=self.split_overlap,
                split_threshold=self.split_threshold,
            )
        metadata = deepcopy(doc.meta)
        metadata["source_id"] = doc.id
        split_docs += self._create_docs_from_splits(
            text_splits=text_split_with_info, splits_pages=splits_pages, splits_start_idxs=splits_start_idxs, meta=metadata
        )

        return split_docs

    def _concatenate_units(
        self, elements: List[str], split_length: int, split_overlap: int, split_threshold: int
    ) -> Tuple[List[str], List[int], List[int]]:
        # 这个地方就已经切了
        """
        Concatenates the elements into parts of split_length units.

        Keeps track of the original page number that each element belongs. If the length of the current units is less
        than the pre-defined `split_threshold`, it does not create a new split. Instead, it concatenates the current
        units with the last split, preventing the creation of excessively small splits.
        """
        text_splits: List[str] = []
        splits_pages: List[int] = []
        splits_start_idxs: List[int] = []
        cur_start_idx = 0
        cur_page = 1
        segments = windowed(elements, n=split_length,
                            step=split_length - split_overlap)

        for seg in segments:
            current_units = [unit for unit in seg if unit is not None]
            txt = "".join(current_units)

            # check if length of current units is below split_threshold
            if len(current_units) < split_threshold and len(text_splits) > 0:
                # concatenate the last split with the current one
                text_splits[-1] += txt

            # NOTE: This line skips documents that have content=""
            elif len(txt) > 0:
                text_splits.append(txt)
                splits_pages.append(cur_page)
                splits_start_idxs.append(cur_start_idx)

            processed_units = current_units[: split_length - split_overlap]
            cur_start_idx += len("".join(processed_units))

            if self.split_by == "page":
                num_page_breaks = len(processed_units)
            else:
                num_page_breaks = sum(processed_unit.count("\f")
                                      for processed_unit in processed_units)

            cur_page += num_page_breaks
        text_splits_add_chunk_info = []  # 用于最后返回
        file_path_info = text_splits[0].split('<file_path_end>')[0].split('<file_path>')[1]



        if '/' in file_path_info: 
            pre_chunk_info = file_path_info.split('/')[1]
        else:
            pre_chunk_info = file_path_info # pre_chunk_info 是上一个chunk的ZUI后一个chunkinfo
        # import pdb
        # pdb.set_trace()
        for sub_content in text_splits:
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

                prompt = "根据下面这段文字，提取"  + str(self.keywordCount) + "个关键词，只返回关键词，不用返回其他内容，返回格式为：关键词1/关键词2···。文字内容为"
                prompt += sub_content
                # response = aizt_chat(user_content = prompt, stream = False)
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

                pre_info += "<key_word>" + response + "<key_word_end>"
            else:
                pre_info += "<key_word> " + "<key_word_end>"
            sub_content = pre_info + sub_content
            text_splits_add_chunk_info.append(sub_content)
        return text_splits, splits_pages, splits_start_idxs, text_splits_add_chunk_info

    def _create_docs_from_splits(
        self, text_splits: List[str], splits_pages: List[int], splits_start_idxs: List[int], meta: Dict[str, Any]
    ) -> List[Document]:
        """
        Creates Document objects from splits enriching them with page number and the metadata of the original document.
        """
        documents: List[Document] = []

        for i, (txt, split_idx) in enumerate(zip(text_splits, splits_start_idxs)):
            copied_meta = deepcopy(meta)
            copied_meta["page_number"] = splits_pages[i]
            copied_meta["split_id"] = i
            copied_meta["split_idx_start"] = split_idx
            doc = Document(content=txt, meta=copied_meta) # 这里面记录的idx不准了
            documents.append(doc)

            if self.split_overlap <= 0:
                continue

            doc.meta["_split_overlap"] = []

            if i == 0:
                continue

            doc_start_idx = splits_start_idxs[i]
            previous_doc = documents[i - 1]
            previous_doc_start_idx = splits_start_idxs[i - 1]
            self._add_split_overlap_information(
                doc, doc_start_idx, previous_doc, previous_doc_start_idx)

        for d in documents:
            d.content=d.content.replace(" ","")
        return documents 

    @staticmethod
    def _add_split_overlap_information(
        self, current_doc: Document, current_doc_start_idx: int, previous_doc: Document, previous_doc_start_idx: int
    ):
        """
        Adds split overlap information to the current and previous Document's meta.

        :param current_doc: The Document that is being split.
        :param current_doc_start_idx: The starting index of the current Document.
        :param previous_doc: The Document that was split before the current Document.
        :param previous_doc_start_idx: The starting index of the previous Document.
        """
        overlapping_range = (current_doc_start_idx - previous_doc_start_idx,
                             len(previous_doc.content))  # type: ignore

        if overlapping_range[0] < overlapping_range[1]:
            # type: ignore
            overlapping_str = previous_doc.content[overlapping_range[0]: overlapping_range[1]]

            if current_doc.content.startswith(overlapping_str):  # type: ignore
                # add split overlap information to this Document regarding the previous Document
                current_doc.meta["_split_overlap"].append(
                    {"doc_id": previous_doc.id, "range": overlapping_range})

                # add split overlap information to previous Document regarding this Document
                overlapping_range = (
                    0, overlapping_range[1] - overlapping_range[0])
                previous_doc.meta["_split_overlap"].append(
                    {"doc_id": current_doc.id, "range": overlapping_range})

    @staticmethod
    def _number_of_sentences_to_keep(self, sentences: List[str], split_length: int, split_overlap: int, language: str, particle_size: str) -> int:
        """
        Returns the number of sentences to keep in the next chunk based on the `split_overlap` and `split_length`.

        :param sentences: The list of sentences to split.
        :param split_length: The maximum number of words in each split.
        :param split_overlap: The number of overlapping words in each split.
        :returns: The number of sentences to keep in the next chunk.
        """
        # If the split_overlap is 0, we don't need to keep any sentences
        if split_overlap == 0:
            return 0

        num_sentences_to_keep = 0
        num_words = 0
        # chinese_tokenizer_coarse = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        # chinese_tokenizer_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
        # Next overlapping Document should not start exactly the same as the previous one, so we skip the first sentence
        for sent in reversed(sentences[1:]):
            if language == 'zh' and particle_size == "coarse":
                num_words += len(self.chinese_tokenizer_coarse(sent))
                # num_words += len(sent.split())
            if language == 'zh' and particle_size == "fine":
                num_words += len(self.chinese_tokenizer_fine(sent))
            # If the number of words is larger than the split_length then don't add any more sentences
            if num_words > split_length:
                break
            num_sentences_to_keep += 1
            if num_words > split_overlap:
                break
        return num_sentences_to_keep


    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        """
        Split documents into smaller parts.

        Splits documents by the unit expressed in `split_by`, with a length of `split_length`
        and an overlap of `split_overlap`.

        :param documents: The documents to split.
        :returns: A dictionary with the following key:
            - `documents`: List of documents with the split texts. Each document includes:
                - A metadata field `source_id` to track the original document.
                - A metadata field `page_number` to track the original page number.
                - All other metadata copied from the original document.

        :raises TypeError: if the input is not a list of Documents.
        :raises ValueError: if the content of a document is None.
        """

        if self._use_sentence_splitter and self.sentence_splitter is None:
            raise RuntimeError(
                "The component DocumentSplitter wasn't warmed up. Run 'warm_up()' before calling 'run()'."
            )

        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            raise TypeError("DocumentSplitter expects a List of Documents as input.")

        split_docs: list[Document] = []
        for doc in documents:
            if doc.content is None:
                raise ValueError(
                    f"DocumentSplitter only works with text documents but content for document ID {doc.id} is None."
                )
            if doc.content == "" and self.skip_empty_documents:
                logger.warning("Document ID {doc_id} has an empty content. Skipping this document.", doc_id=doc.id)
                continue

            split_docs += self._split_document(doc)
        return {"documents": split_docs}
