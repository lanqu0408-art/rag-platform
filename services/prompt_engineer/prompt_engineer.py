import os
import json
import traceback
from typing import Optional, Union, Dict, List

from utils.wrapper import Wrapper
from services.external_api import gpt_chat, qwen_chat
from services.prompt_engineer.template import (
    INTENTION_RECOGNITION_TEMPLATE, 
    NER_TEMPLATE,
    RISK_ASSESSMENT_TEMPLATE
    )


class BasePromptEngineer:
    def __init__(self):
        self.role_map = {
            "assist": "客服",
            "assistant": "客服",
            "bot": "客服",
            "customer": "客户",
            "user": "客户"
        }

    # def _get_dialogue(self, messages: list) -> str:
    #     """获取对话内容"""
    #     dialogue = ''
    #     for msg in messages:
    #         dialogue += f"{msg['role']}: {msg['content']}\n"
    #     return dialogue

    def _llm_chat(
        self,
        user_content: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        system_content: str = "You are a helpful assistant.",
        to_json: bool = False,
        model_type: str = None,
        **kwargs
    ):
        if 'gpt' in model_type.lower():
            response = gpt_chat(
                user_content=user_content, 
                messages=messages,
                system_content=system_content,
                to_json=to_json, 
                model_type=model_type, 
                **kwargs)
        elif 'qwen' in model_type.lower():
            response = qwen_chat(
                user_content=user_content, 
                messages=messages,
                system_content=system_content,
                to_json=to_json, 
                model_type=model_type, 
                **kwargs)
        else:
            raise Exception(f"model_type: {model_type} is not supported")
        # print(f"\nLLM Response: \n{response}\n")  # 测试用
        return response
        

    def _get_dialogue(self, messages: list) -> str:
        """获取对话内容"""
        return json.dumps(messages, ensure_ascii=False, indent=2)


class IntentPromptEngineer(BasePromptEngineer):
    # def _get_intention_description(self, candidate_intent: list) -> str:
    #     intention_description = ''
    #     for intent in candidate_intent:
    #         intention_description += f"意图名称: {intent['name']}\n意图描述: {intent['desc']}\n\n"
    #     return intention_description

    def _get_intention_description(self, candidate_intent: list) -> str:
        intent_list = [
            {
                "intent_name": item["name"],
                "description": item["desc"]
            }
            for item in candidate_intent
        ]
        return json.dumps(intent_list, ensure_ascii=False, indent=2)

    def prepare_response_data(self, info_carrier):
        outputs = {
            'model_type': info_carrier['model_type'],
            'hit_intent': info_carrier['hit_intent'],
        }
        return outputs

    def process(self, info_carrier: dict) -> dict:
        try:
            intent = info_carrier['intent']
            messages = info_carrier['messages']
            model_type = info_carrier['model_type']
            print(f"model_type: {model_type} \n")

            candidate_intents = self._get_intention_description(intent)
            dialogue = self._get_dialogue(messages)

            # Component prompt words
            prompt = INTENTION_RECOGNITION_TEMPLATE
            prompt = prompt.replace('{candidate_intents}', candidate_intents)
            prompt = prompt.replace('{dialogue}', dialogue)
            print(f"\nprompt: \n{prompt}\n")  # 测试用

            response = self._llm_chat(user_content=prompt, to_json=True, model_type=model_type)
            info_carrier['llm_chat_response'] = response
            print(f"LLM Response: \n{response}\n")  # 测试用

            # matched_intents = {}
            # for data in response['matched_intents']:
            #     matched_intents[data['intent_name']] = data['hit']
            # hit_intent = {}
            # for data in intent:
            #     if data['name'] in matched_intents:
            #         hit = matched_intents[data['name']]
            #         hit_intent[data['id']] = '1' if hit else '0'
            #     else:
            #         hit_intent[data['id']] = '0'

            intent_name_lts = [data['name'] for data in intent]
            intent_name2type = {data['name']: data.get('type', None) for data in intent}
            hit_intent = []
            for data in response['matched_intents']:
                intent_name = data['intent_name']
                if intent_name in intent_name_lts and data['hit']:
                    hit_intent.append({
                        "intent_name": intent_name,
                        "intent_type": intent_name2type[intent_name]
                    })
                
            info_carrier['hit_intent'] = hit_intent
            print(f"hit_intent: \n{hit_intent}\n")  # 测试用

            outputs = {
                'model_type': info_carrier['model_type'],
                'hit_intent': info_carrier['hit_intent'],
            }
            info_carrier['outputs'] = outputs
            return info_carrier

        except Exception as e:
            print(f"\n{traceback.format_exc()}\n")  # 测试用
            info_carrier['error'] = f"IntentPromptEngineer error: {str(e)}"
            raise Exception(f"IntentPromptEngineer error: {str(e)} ")


class IntentPromptEngineerWrapper(Wrapper):
    def __init__(self, config=None):
        self.instance = IntentPromptEngineer()


class NERPromptEngineer(BasePromptEngineer):
    def _get_entity_description(self, candidate_entity: list) -> str:
        entity_description = ''
        for entity in candidate_entity:
            entity_description += f"信息名称: {entity['name']}\n信息描述: {entity['desc']}\n\n"
        return entity_description

    def process(self, info_carrier: dict) -> dict:
        try:
            entity = info_carrier['entity']
            messages = info_carrier['messages']
            model_type = info_carrier['model_type']

            entity_description = self._get_entity_description(entity)
            dialogue = self._get_dialogue(messages)

            # Component prompt words
            prompt = NER_TEMPLATE
            prompt = prompt.replace('{dialogue}', dialogue)
            prompt = prompt.replace('{entity_description}', entity_description)
            print(f"\nprompt: \n{prompt}\n")  # 测试用

            response = self._llm_chat(user_content=prompt, to_json=True, model_type=model_type)
            info_carrier['llm_chat_response'] = response

            hit_entity = {}
            
            # entity_names = [data['name'] for data in entity]
            # for k,v in response.items():
            #     if k in entity_names:
            #         hit_entity[k] = v

            name2id = {data['name']: data['id'] for data in entity}
            for data in entity:
                entity_name = data['name']
                entity_value = response.get(entity_name, None)
                if not entity_value:
                    entity_value = ""
                # hit_entity[name2id[entity_name]] = entity_value
                hit_entity[entity_name] = entity_value

            info_carrier['hit_entity'] = hit_entity
            print(f"hit_entity: \n{hit_entity}\n")  # 测试用

            return info_carrier

        except Exception as e:
            print(f"\n{traceback.format_exc()}\n")  # 测试用
            info_carrier['error'] = f"NERPromptEngineer error: {str(e)} "
            raise Exception(f"NERPromptEngineer error: {str(e)} ")


class NERPromptEngineerWrapper(Wrapper):
    def __init__(self, config=None):
        self.instance = NERPromptEngineer()


class RiskAssessmentPromptEngineer(BasePromptEngineer):

    def process(self, info_carrier: dict) -> dict:
        try:
            messages = info_carrier['messages']
            model_type = info_carrier['model_type']
            print(f"model_type: {model_type} \n")

            dialogue = self._get_dialogue(messages)

            # Component prompt words
            prompt = RISK_ASSESSMENT_TEMPLATE
            prompt = prompt.replace('{dialogue}', dialogue)
            print(f"\nprompt: \n{prompt}\n")  # 测试用

            response = self._llm_chat(user_content=prompt, to_json=True, model_type=model_type)
            info_carrier['llm_response'] = response
            print(f"LLM Response: \n{response}\n")  # 测试用

            info_carrier['outputs'] = response
            return info_carrier

        except Exception as e:
            print(f"\n{traceback.format_exc()}\n")  # 测试用
            info_carrier['error'] = f"IntentPromptEngineer error: {str(e)}"
            raise Exception(f"IntentPromptEngineer error: {str(e)} ")

class RiskAssessmentPromptEngineerWrapper(Wrapper):
    def __init__(self, config=None):
        self.instance = RiskAssessmentPromptEngineer()
