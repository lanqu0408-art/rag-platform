
# INTENTION_RECOGNITION_TEMPLATE = """
# 你是一个专业的意图识别AI助手，需要根据客户与客服的聊天历史，从候选意图中选择所有匹配的客户意图。

# # 候选意图列表（格式：JSON数组）: 
# {candidate_intents}

# # 聊天历史（格式：JSON数组）: 
# {dialogue}

# # 任务要求：
# 1. 仔细分析聊天内容，理解客户表达的意思
# 2. 逐一验证候选意图是否匹配上，若匹配上，hit为true，否则为false

# # 输出格式：
# ```json
# {
#   "matched_intents": [
#     {
#       "intent_name": str,  // 第一个候选意图名称
#       "hit": bool  // 是否匹配上
#     },
#     {
#       "intent_name": str,  // 第二个候选意图名称
#       "hit": bool  // 是否匹配上
#     },
#   ]
# }
# ```
# """

# INTENTION_RECOGNITION_TEMPLATE = """
# 你是一个专业的意图识别AI助手，需要根据客户与客服的聊天历史，从候选意图中选择所有匹配的客户意图。

# # 候选意图列表（格式：JSON数组）: 
# {candidate_intents}

# # 任务要求：
# 1. 仔细分析聊天内容，理解客户表达的意思
# 2. 逐一验证候选意图是否匹配上，若匹配上，hit为true，否则为false

# # 输出格式：
# ```json
# {
#   "matched_intents": [
#     {
#       "intent_name": str,  // 第一个候选意图名称
#       "hit": bool  // 是否匹配上
#     },
#     {
#       "intent_name": str,  // 第二个候选意图名称
#       "hit": bool  // 是否匹配上
#     },
#   ]
# }
# ```

# # 聊天历史（格式：JSON数组）: 
# {dialogue}
# """


INTENTION_RECOGNITION_TEMPLATE = """
你是一个专业的意图识别AI助手，需要根据客户与客服的聊天历史，从候选意图中选择所有匹配的客户意图。

# 候选意图列表（格式：JSON数组）: 
{candidate_intents}

# 聊天历史（格式：JSON数组）: 
{dialogue}

# 任务要求：
1. 仔细分析聊天内容，理解客户表达的意思
2. 逐一验证候选意图是否匹配上，若匹配上，hit为true，否则为false

# 输出格式：
```json
{
  "matched_intents": [
    {
      "intent_name": str,  // 第一个候选意图名称
      "reason": str,  // 匹配上的原因
      "hit": bool,  // 是否匹配上
    },
    {
      "intent_name": str,  // 第二个候选意图名称
      "reason": str,  // 匹配上的原因
      "hit": bool,  // 是否匹配上
    },
  ]
}
```
"""


# INTENTION_RECOGNITION_TEMPLATE = """
# 你是一个专业的意图识别AI助手，需要根据客户与客服的聊天历史，从候选意图中选择所有匹配的客户意图。

# # 候选意图列表（格式：JSON数组）: 
# {candidate_intents}

# # 任务要求：
# 1. 仔细分析聊天内容，理解客户表达的意思
# 2. 逐一验证候选意图是否匹配上，若匹配上，hit为true，否则为false

# # 输出格式：
# ```json
# {
#   "matched_intents": [
#     {
#       "intent_name": str,  // 第一个候选意图名称
#       "reason": str,  // 匹配上的原因
#       "hit": bool,  // 是否匹配上
#     },
#     {
#       "intent_name": str,  // 第二个候选意图名称
#       "reason": str,  // 匹配上的原因
#       "hit": bool,  // 是否匹配上
#     },
#   ]
# }
# ```

# # 聊天历史（格式：JSON数组）: 
# {dialogue}
# """



# NER_TEMPLATE = """你是一个专业的信息识别AI助手。
# 你的任务是从客户(user)与客服(assistant)的对话记录中抽取信息列表中的信息。

# # 信息列表（格式：JSON数组）: 
# {entity_description}

# # 对话记录（格式：JSON数组）: 
# {dialogue}

# # 任务要求：
# 1. 仔细分析聊天内容，理解客户表达的信息
# 2. 理解信息列表中要抽取的信息
# 3. 以JSON的格式返回你抽取到的内容，key是识别到的实体名称，value是对应的实体值

# # 输出格式：
# ```json
# {
#   "entity_name_1": "entity_value_1",
#   "entity_name_2": "entity_value_2",
# }
# ```
# """


# NER_TEMPLATE = """你是一个专业的信息识别AI助手。
# 你的任务是从客户(user)与客服(assistant)的对话记录中抽取信息列表中的信息。

# # 信息列表（格式：JSON数组）: 
# {entity_description}

# # 对话记录（格式：JSON数组）: 
# {dialogue}

# # 任务要求：
# 1. 仔细分析聊天内容，理解客户表达的信息
# 2. 理解信息列表中要抽取的信息
# 3. 以JSON的格式返回你抽取到的内容，key是识别到的实体名称，value是对应的实体值

# # 输出格式：
# ```json
# {
#   "entity_name_1": "entity_value_1",
#   "entity_name_2": "entity_value_2",
# }
# ```
# """


NER_TEMPLATE = """你是一个专业的信息识别AI助手。
你的任务是从客户(user)与客服(assistant)的对话记录中抽取信息列表中的信息。

# 信息列表（格式：JSON数组）: 
{entity_description}

# 对话记录（格式：JSON数组）: 
{dialogue}

# 任务要求：
1. 仔细分析聊天内容，理解客户表达的信息
2. 理解信息列表中要抽取的信息
3. 以JSON的格式返回你抽取到的内容，key是识别到的实体名称，value是对应的实体值
4. 若对话记录中不存在相应信息，value的值为""(空字符串)

# 输出格式：
```json
{
  "entity_name_1": "entity_value_1",
  "entity_name_2": "entity_value_2",
}
```
"""


RISK_ASSESSMENT_TEMPLATE = """你是一个专业的风险评估AI助手。
你的任务是理解客户(user)与客服(assistant)的聊天历史，并判断客户是否被诈骗。

# 通话背景:
洋钱罐是国内的金融借贷平台。客服是来自洋钱罐平台的风险评估人员。客服的职责是判断客户是否被诈骗。
通常在通话开始时，客服会询问客户一些基本信息，例如客户姓名、身份证号、手机号、生日等用来核实客户身份。
在通话过程中，客服会询问客户一些问题，例如客户的贷款用途、否收到过诈骗电话或短信。

# 聊天历史（格式：JSON数组）: 
{dialogue}

# 任务要求：
1. 仔细分析聊天内容，理解客户表达的意思
2. 判断客户是否被诈骗，若被诈骗，risk为true，否则为false

# 输出格式：
```json
{
  "analysis": str,  // 分析结果
  "risk": bool,  // 是否被诈骗
}
```
"""
