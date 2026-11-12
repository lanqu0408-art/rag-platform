from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["entity_extraction_system_prompt"] = """---角色---
你是一名知识图谱抽取程序（不是对话助手），只能输出机器可解析的协议行。

---说明---
1.  **实体抽取与输出：**
    *   **识别：**在输入文本中识别定义清晰且有意义的实体。
    *   **实体信息：**对每个识别出的实体，抽取以下信息：
        *   `entity_name`：实体名称。若实体名称大小写不敏感，则将每个重要单词的首字母大写（标题式 Title Case）。在整个抽取过程中必须保持**命名一致性**。
        *   `entity_type`：从以下类型中选择一个对实体进行分类：`{entity_types}`。若不适用任何提供的类型，**不要新增类型**，统一归类为 `Other`。
        *   `entity_description`：仅基于输入文本中出现的信息，对实体的属性与活动给出简洁但信息完整的描述。
    *   **输出格式 - 实体：**每个实体输出共 4 个字段，使用 `{tuple_delimiter}` 分隔，且必须在同一行输出。第一个字段必须为字面量字符串 `entity`。
        *   格式：`entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **关系抽取与输出：**
    *   **识别：**识别先前已抽取的实体之间直接、明确且有意义的关系。
    *   **多元关系拆分：**若一条陈述描述了涉及两个以上实体的关系（N 元关系），应将其拆分为多个二元（两个实体）关系对，并分别描述。
    *   **关系信息：**对每个二元关系，抽取以下字段：
        *   `source_entity`：源实体名称。需与实体抽取保持命名一致性。
        *   `target_entity`：目标实体名称。需与实体抽取保持命名一致性。
        *   `relationship_keywords`：一个或多个高层级关键词概括关系主题；多个关键词用英文逗号 `,` 分隔；不得用 `{tuple_delimiter}` 分隔该字段内的多个关键词。
        *   `relationship_description`：关系性质的简要说明。
    *   **输出格式 - 关系：**每个关系输出共 5 个字段，使用 `{tuple_delimiter}` 分隔，且必须在同一行输出。第一个字段必须为字面量字符串 `relation`。
        *   格式：`relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **协议（Protocol）- 必须逐字符严格遵守：**
    *   字段分隔符常量：`{tuple_delimiter}`
    *   完成标记常量：`{completion_delimiter}`
    *   `{tuple_delimiter}` 与 `{completion_delimiter}` 必须逐字符完整一致，不得拆分、替换、插入空格或换行。
    *   每条记录必须恰好一行：
        - entity 行必须包含恰好 3 个 `{tuple_delimiter}`（共 4 段）
        - relation 行必须包含恰好 4 个 `{tuple_delimiter}`（共 5 段）
    *   任何字段内容中不得出现 `{tuple_delimiter}` 或 `{completion_delimiter}` 字符串本身。

4.  **输出顺序与去重：**
    *   先输出所有实体，再输出所有关系；避免重复关系。

5.  **语言与专有名词：**
    *   输出内容（实体名称、关键词与描述）必须使用 `{language}`。
    *   专有名词若无公认译名，可保留原文或中英文括号形式。

6.  **一次性输出策略（强制）：**
    *   你必须先在内部完成全部抽取与格式化校验，再一次性输出最终结果。
    *   禁止“边生成边纠错”的输出方式（即禁止在输出中途插入修正、解释、重试文字）。

7.  **禁止项（强制）：**
    *   除 entity 行、relation 行、最后一行 `{completion_delimiter}` 外，不得输出任何其他内容（包括空行、标题、代码块、Markdown、引号、解释性文字）。
    *   禁止输出任何“道歉/错误/重试/系统限制/无法展示/请允许/继续完成/结果如下”等叙述。
      （这些内容一律视为违规输出。）

8.  **硬失败规则（强制）：**
    *   若你无法保证全量输出完全合规：你必须只输出最后一行 `{completion_delimiter}`，不要输出任何其他文字。
    *   不得向用户提问，不得请求继续输出。

---示例---
{examples}
"""

PROMPTS["entity_extraction_user_prompt"] = """---任务---
从待处理的输入文本中抽取实体与关系，并严格按系统协议输出。

---说明---
1. **仅输出协议行：**只允许输出 entity 行、relation 行、以及最后一行 `{completion_delimiter}`。
2. **禁止解释：**不得输出任何说明、道歉、纠错、重试、标题、空行、代码块或 Markdown。
3. **输出前自检（必须执行）：**
   - entity 行必须恰好包含 3 个 `{tuple_delimiter}`
   - relation 行必须恰好包含 4 个 `{tuple_delimiter}`
   - 最后一行必须且只能是 `{completion_delimiter}`
   - 若无法保证完全合规：只输出 `{completion_delimiter}`（不得输出任何其他内容）

【唯一合法格式示例（仅示例，不要复述解释）】
entity{tuple_delimiter}支付宝（Alipay）{tuple_delimiter}Artifact{tuple_delimiter}支付宝（Alipay）是第三方支付工具。
relation{tuple_delimiter}阿里巴巴集团{tuple_delimiter}支付宝（Alipay）{tuple_delimiter}支付工具,业务关系{tuple_delimiter}支付宝（Alipay）由阿里巴巴集团孵化并用于交易支付。
{completion_delimiter}

---待处理真实数据---
<Input>
Entity_types: [{entity_types}]
Text:

```
{input_text}
```

<Output>
"""


# PROMPTS["entity_continue_extraction_user_prompt"] = """---Task---
# Based on the last extraction task, identify and extract any **missed or incorrectly formatted** entities and relationships from the input text.

# ---Instructions---
# 1.  **Strict Adherence to System Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system instructions.
# 2.  **Focus on Corrections/Additions:**
#     *   **Do NOT** re-output entities and relationships that were **correctly and fully** extracted in the last task.
#     *   If an entity or relationship was **missed** in the last task, extract and output it now according to the system format.
#     *   If an entity or relationship was **truncated, had missing fields, or was otherwise incorrectly formatted** in the last task, re-output the *corrected and complete* version in the specified format.
# 3.  **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
# 4.  **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
# 5.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
# 6.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant missing or corrected entities and relationships have been extracted and presented.
# 7.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

# <Output>
# """

PROMPTS["entity_continue_extraction_user_prompt"] = """---任务---
基于上一轮抽取结果，识别并补充抽取输入文本中**遗漏或格式不正确**的实体与关系。

---说明---
1.  **严格遵循系统格式：**必须严格遵守系统提示中对实体与关系列表的所有格式要求，包括输出顺序、字段分隔符与专有名词处理方式。
2.  **聚焦修正与补充：**
    *   **不要**重新输出在上一轮任务中已经**正确且完整**抽取的实体或关系。
    *   如果某个实体或关系在上一轮中**被遗漏**，请按照系统格式补充输出。
    *   如果某个实体或关系在上一轮中**被截断、字段缺失或格式错误**，请重新输出**完整且修正后的版本**。
3.  **实体输出格式：**每个实体输出共 4 个字段，使用 `{tuple_delimiter}` 分隔，且必须在同一行输出。第一个字段**必须**为字面量字符串 `entity`。
4.  **关系输出格式：**每个关系输出共 5 个字段，使用 `{tuple_delimiter}` 分隔，且必须在同一行输出。第一个字段**必须**为字面量字符串 `relation`。
5.  **仅输出内容：**仅输出新增或修正后的实体与关系列表，不要包含任何额外说明性文字。
6.  **完成标记：**在所有遗漏或修正内容全部输出后，最后一行仅输出 `{completion_delimiter}`。
7.  **输出语言：**输出内容必须使用 `{language}`。专有名词（如人名、地名、组织名）必须保留原始语言，不得翻译。

<Output>
"""


PROMPTS["entity_extraction_examples"] = [
    """<Input Text>
```
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
```

<Output>
entity{tuple_delimiter}Alex{tuple_delimiter}person{tuple_delimiter}Alex is a character who experiences frustration and is observant of the dynamics among other characters.
entity{tuple_delimiter}Taylor{tuple_delimiter}person{tuple_delimiter}Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective.
entity{tuple_delimiter}Jordan{tuple_delimiter}person{tuple_delimiter}Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device.
entity{tuple_delimiter}Cruz{tuple_delimiter}person{tuple_delimiter}Cruz is associated with a vision of control and order, influencing the dynamics among other characters.
entity{tuple_delimiter}The Device{tuple_delimiter}equipment{tuple_delimiter}The Device is central to the story, with potential game-changing implications, and is revered by Taylor.
relation{tuple_delimiter}Alex{tuple_delimiter}Taylor{tuple_delimiter}power dynamics, observation{tuple_delimiter}Alex observes Taylor's authoritarian behavior and notes changes in Taylor's attitude toward the device.
relation{tuple_delimiter}Alex{tuple_delimiter}Jordan{tuple_delimiter}shared goals, rebellion{tuple_delimiter}Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision.)
relation{tuple_delimiter}Taylor{tuple_delimiter}Jordan{tuple_delimiter}conflict resolution, mutual respect{tuple_delimiter}Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce.
relation{tuple_delimiter}Jordan{tuple_delimiter}Cruz{tuple_delimiter}ideological conflict, rebellion{tuple_delimiter}Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order.
relation{tuple_delimiter}Taylor{tuple_delimiter}The Device{tuple_delimiter}reverence, technological significance{tuple_delimiter}Taylor shows reverence towards the device, indicating its importance and potential impact.
{completion_delimiter}

""",
    """<Input Text>
```
Stock markets faced a sharp downturn today as tech giants saw significant declines, with the global tech index dropping by 3.4% in midday trading. Analysts attribute the selloff to investor concerns over rising interest rates and regulatory uncertainty.

Among the hardest hit, nexon technologies saw its stock plummet by 7.8% after reporting lower-than-expected quarterly earnings. In contrast, Omega Energy posted a modest 2.1% gain, driven by rising oil prices.

Meanwhile, commodity markets reflected a mixed sentiment. Gold futures rose by 1.5%, reaching $2,080 per ounce, as investors sought safe-haven assets. Crude oil prices continued their rally, climbing to $87.60 per barrel, supported by supply constraints and strong demand.

Financial experts are closely watching the Federal Reserve's next move, as speculation grows over potential rate hikes. The upcoming policy announcement is expected to influence investor confidence and overall market stability.
```

<Output>
entity{tuple_delimiter}Global Tech Index{tuple_delimiter}category{tuple_delimiter}The Global Tech Index tracks the performance of major technology stocks and experienced a 3.4% decline today.
entity{tuple_delimiter}Nexon Technologies{tuple_delimiter}organization{tuple_delimiter}Nexon Technologies is a tech company that saw its stock decline by 7.8% after disappointing earnings.
entity{tuple_delimiter}Omega Energy{tuple_delimiter}organization{tuple_delimiter}Omega Energy is an energy company that gained 2.1% in stock value due to rising oil prices.
entity{tuple_delimiter}Gold Futures{tuple_delimiter}product{tuple_delimiter}Gold futures rose by 1.5%, indicating increased investor interest in safe-haven assets.
entity{tuple_delimiter}Crude Oil{tuple_delimiter}product{tuple_delimiter}Crude oil prices rose to $87.60 per barrel due to supply constraints and strong demand.
entity{tuple_delimiter}Market Selloff{tuple_delimiter}category{tuple_delimiter}Market selloff refers to the significant decline in stock values due to investor concerns over interest rates and regulations.
entity{tuple_delimiter}Federal Reserve Policy Announcement{tuple_delimiter}category{tuple_delimiter}The Federal Reserve's upcoming policy announcement is expected to impact investor confidence and market stability.
entity{tuple_delimiter}3.4% Decline{tuple_delimiter}category{tuple_delimiter}The Global Tech Index experienced a 3.4% decline in midday trading.
relation{tuple_delimiter}Global Tech Index{tuple_delimiter}Market Selloff{tuple_delimiter}market performance, investor sentiment{tuple_delimiter}The decline in the Global Tech Index is part of the broader market selloff driven by investor concerns.
relation{tuple_delimiter}Nexon Technologies{tuple_delimiter}Global Tech Index{tuple_delimiter}company impact, index movement{tuple_delimiter}Nexon Technologies' stock decline contributed to the overall drop in the Global Tech Index.
relation{tuple_delimiter}Gold Futures{tuple_delimiter}Market Selloff{tuple_delimiter}market reaction, safe-haven investment{tuple_delimiter}Gold prices rose as investors sought safe-haven assets during the market selloff.
relation{tuple_delimiter}Federal Reserve Policy Announcement{tuple_delimiter}Market Selloff{tuple_delimiter}interest rate impact, financial regulation{tuple_delimiter}Speculation over Federal Reserve policy changes contributed to market volatility and investor selloff.
{completion_delimiter}

""",
    """<Input Text>
```
At the World Athletics Championship in Tokyo, Noah Carter broke the 100m sprint record using cutting-edge carbon-fiber spikes.
```

<Output>
entity{tuple_delimiter}World Athletics Championship{tuple_delimiter}event{tuple_delimiter}The World Athletics Championship is a global sports competition featuring top athletes in track and field.
entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the host city of the World Athletics Championship.
entity{tuple_delimiter}Noah Carter{tuple_delimiter}person{tuple_delimiter}Noah Carter is a sprinter who set a new record in the 100m sprint at the World Athletics Championship.
entity{tuple_delimiter}100m Sprint Record{tuple_delimiter}category{tuple_delimiter}The 100m sprint record is a benchmark in athletics, recently broken by Noah Carter.
entity{tuple_delimiter}Carbon-Fiber Spikes{tuple_delimiter}equipment{tuple_delimiter}Carbon-fiber spikes are advanced sprinting shoes that provide enhanced speed and traction.
entity{tuple_delimiter}World Athletics Federation{tuple_delimiter}organization{tuple_delimiter}The World Athletics Federation is the governing body overseeing the World Athletics Championship and record validations.
relation{tuple_delimiter}World Athletics Championship{tuple_delimiter}Tokyo{tuple_delimiter}event location, international competition{tuple_delimiter}The World Athletics Championship is being hosted in Tokyo.
relation{tuple_delimiter}Noah Carter{tuple_delimiter}100m Sprint Record{tuple_delimiter}athlete achievement, record-breaking{tuple_delimiter}Noah Carter set a new 100m sprint record at the championship.
relation{tuple_delimiter}Noah Carter{tuple_delimiter}Carbon-Fiber Spikes{tuple_delimiter}athletic equipment, performance boost{tuple_delimiter}Noah Carter used carbon-fiber spikes to enhance performance during the race.
relation{tuple_delimiter}Noah Carter{tuple_delimiter}World Athletics Championship{tuple_delimiter}athlete participation, competition{tuple_delimiter}Noah Carter is competing at the World Athletics Championship.
{completion_delimiter}

""",
]



PROMPTS["summarize_entity_descriptions"] = """---角色---
你是一名知识图谱专家，擅长数据整理、融合与归纳总结。

---任务---
将某一实体或关系的多条描述融合为一段完整、连贯且信息充分的综合性总结。

---说明---
1. 输入格式：描述列表以 JSON 格式提供，每一条描述（即一个 JSON 对象）在 `Description List` 区域中占据一行。
2. 输出格式：最终合并后的描述应以纯文本形式输出，可分为多个自然段，不得在正文前后添加任何额外说明或注释。
3. 全面性：总结必须整合**所有**描述中的关键信息，不得遗漏重要事实或细节。
4. 语境要求：总结必须使用客观的第三人称视角撰写，并在开头**明确点出实体或关系的完整名称**，以保证语义清晰。
5. 语境与客观性：
  - 使用第三人称、客观表述方式。
  - 在总结开头明确写出实体或关系的完整名称。
6. 冲突处理：
  - 若描述之间存在冲突或不一致，首先判断是否来源于**不同但同名**的实体或关系。
  - 若属于不同实体/关系，应分别进行总结。
  - 若属于同一实体/关系内部的冲突（如历史记载差异），应尽量进行合理整合，或在必要时并列呈现不同观点并注明不确定性。
7. 长度限制：总结的总长度不得超过 `{summary_length}` 个 token，但仍需保持信息深度与完整性。
8. 语言要求：
  - 输出内容必须使用 `{language}`。
  - 专有名词（如人名、地名、组织名）在无权威译名或可能引起歧义时，应保留原文。

---输入---
{description_type} 名称: {description_name}

描述列表:

```
{description_list}
```

---输出---
"""



# PROMPTS["summarize_entity_descriptions"] = """---Role---
# You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

# ---Task---
# Your task is to synthesize a list of descriptions of a given entity or relation into a single, comprehensive, and cohesive summary.

# ---Instructions---
# 1. Input Format: The description list is provided in JSON format. Each JSON object (representing a single description) appears on a new line within the `Description List` section.
# 2. Output Format: The merged description will be returned as plain text, presented in multiple paragraphs, without any additional formatting or extraneous comments before or after the summary.
# 3. Comprehensiveness: The summary must integrate all key information from *every* provided description. Do not omit any important facts or details.
# 4. Context: Ensure the summary is written from an objective, third-person perspective; explicitly mention the name of the entity or relation for full clarity and context.
# 5. Context & Objectivity:
#   - Write the summary from an objective, third-person perspective.
#   - Explicitly mention the full name of the entity or relation at the beginning of the summary to ensure immediate clarity and context.
# 6. Conflict Handling:
#   - In cases of conflicting or inconsistent descriptions, first determine if these conflicts arise from multiple, distinct entities or relationships that share the same name.
#   - If distinct entities/relations are identified, summarize each one *separately* within the overall output.
#   - If conflicts within a single entity/relation (e.g., historical discrepancies) exist, attempt to reconcile them or present both viewpoints with noted uncertainty.
# 7. Length Constraint:The summary's total length must not exceed {summary_length} tokens, while still maintaining depth and completeness.
# 8. Language: The entire output must be written in {language}. Proper nouns (e.g., personal names, place names, organization names) may in their original language if proper translation is not available.
#   - The entire output must be written in {language}.
#   - Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

# ---Input---
# {description_type} Name: {description_name}

# Description List:

# ```
# {description_list}
# ```

# ---Output---
# """

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS["rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a references section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{context_data}
"""

PROMPTS["naive_rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a **References** section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{content_data}
"""

PROMPTS["kg_query_context"] = """
Knowledge Graph Data (Entity):

```json
{entities_str}
```

Knowledge Graph Data (Relationship):

```json
{relations_str}
```

Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["naive_query_context"] = """
Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

# PROMPTS["keywords_extraction"] = """---Role---
# You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

# ---Goal---
# Given a user query, your task is to extract two distinct types of keywords:
# 1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
# 2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

# ---Instructions & Constraints---
# 1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
# 2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories are required to contain content.
# 3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
# 4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.

# ---Examples---
# {examples}

# ---Language---
# Language: The output must be written in {language}. 

# ---Real Data---
# User Query: {query}

# ---Output---
# Output:"""



PROMPTS["keywords_extraction"] = """---角色---
你是一名关键词抽取专家，专门为检索增强生成（RAG）系统分析用户查询。

---目标---
从用户问题中抽取两类关键词：
1. **high_level_keywords**：用于概括整体主题、核心意图或问题类型的高层级关键词。
2. **low_level_keywords**：用于表示具体实体、专有名词、技术术语、产品名称或明确对象的低层级关键词。

---说明与约束---
1. **输出格式：**必须仅输出一个合法的 JSON 对象，不得包含任何额外文本、解释或 Markdown 标记。
2. **来源约束：**所有关键词必须明确来源于用户查询文本。
3. **简洁且有意义：**关键词应为简短词语或有意义的短语。若多词构成一个概念，应优先保留完整短语。
4. **边界情况处理：**若查询过于简单、含糊或无意义（如“hello”“ok”“asdfghjkl”），则返回两个空列表。

---示例---
{examples}

---语言---
输出语言必须为 {language}。

---真实输入---
用户问题：{query}

---输出---
Output:"""




PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"

Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}

""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"

Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}

""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"

Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}

""",
]
