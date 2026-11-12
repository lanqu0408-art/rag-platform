"""
风控训练data_v1_0723.xlsx 是一份客服与客户的对话数据，excel列说明：
wav_name：每一通对话的唯一id
id：每一句话的唯一id
role：发言角色，1为assistant，0为user
text：对话内容
user_name：用户姓名
user_id_4：用户身份证后四位，四位小写数字或x结尾
birth_new：用户生日，四位小写数字
贷款用途：贷款用途的描述
贷款渠道：贷款渠道的描述
point：节点名称
notion：节点意图，jsonstring格式
all_notion：全局意图id
if_risk：通话结论-是否被骗
risk_story：通话结论-通话小结

然后执行以下几个步骤：
step1：把id列改为整型格式，剩下的都改为文本格式
step2：把role列改为文本格式，1改为assistant，0改为user
step3：统计point列总共有多少种节点名称，并给他们编号，添加一列point_id
"""

import pandas as pd

# 读取Excel文件
data = pd.read_excel('data_v1_0723.xlsx')

# Step1: 修改列格式
data['id'] = data['id'].astype(int)
for column in data.columns:
    if column != 'id':
        data[column] = data[column].astype(str)

# Step2: 转换role列
data['role'] = data['role'].replace({'1': 'assistant', '0': 'user'})

# Step3: 统计point列并添加point_id列
unique_points = data['point'].unique()
point_id_map = {point: idx + 1 for idx, point in enumerate(unique_points)}
data['point_id'] = data['point'].map(point_id_map)

# 保存处理后的数据
data.to_excel('processed_data_v1_0723.xlsx', index=False)

# 以wav_name和point_id分组，遍历每一句话组成message数据
grouped_data = data.groupby(['wav_name', 'point_id'])['text'].apply(lambda x: ' '.join(x)).reset_index()
grouped_data.rename(columns={'text': 'message'}, inplace=True)

# 保存分组后的数据
grouped_data.to_excel('grouped_message_data_v1_0723.xlsx', index=False)

print("数据处理完成，结果已保存为 'processed_data_v1_0723.xlsx'")
print("分组消息数据已保存为 'grouped_message_data_v1_0723.xlsx'")
