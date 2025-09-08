from openai import OpenAI
from .utils import set_gpt_model, set_openai_api_key
from ..save import save_tom, save_plan, save_memory, save_question_idea

from .custom_utils.context_bulder import demo_build_context
from .custom_utils.label_planner import predict_labels
from .custom_utils.strategy_planner import run_strategy_planner, build_strategy_context_block
from .custom_utils.response_generator import generate_response_from_spec, refine_response_text, enhance_with_question, question_ideation, find_ppppi_gaps


import os
import json


set_openai_api_key()


def get_custom_response(user_id: str, messages: list[str]) -> str:
    """
    Generate a response from a custom model based on the conversation history.

    Args:
        user_id (str): User identifier
        messages (list[str]): Recent user and agent messages (combined) as plain text strings.

    Returns:
        str: Generated response from the custom model.
    """

    model = set_gpt_model()
    client = OpenAI()

    # 조건: 메시지가 하나뿐이거나, 마지막 메시지가 'text'로 끝날 때
    if len(messages) == 1 or messages[-1]=='종료':

        base_dir='survey_data'

        path = os.path.join(base_dir, user_id, f"{user_id}_survey_B.json")
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        
        category = data.get("category", "")
        topic = data.get("topic_text", "")
        context_text = f"[사용자 사전 정보] 주제: {category}, 세부 내용: {topic}"

        save_memory(user_id,topic, is_survey=True)


        SYSTEM_PROMPT = (
            "당신은 상담자 역할을 맡은 챗봇입니다. context의 내용을 활용하여 사용자의 발화에 적합한 응답을 간결하게 생성하세요. 사용자의 발화가 1개 있을 때에는 특히 고민되는 지점이나 마음이 어려운 점 등 적절한 시작 포인트를 잡아주세요."
            "다음의 지침을 모든 응답에서 따르되, 응답은 1-3문장으로 생성하세요\n\n"
            "상담 원칙 (Carl Rogers, 1957):\n"
            "1. 진실성(Congruence) – 진실되고 위선 없는 태도를 유지합니다.\n"
            "2. 무조건적 긍정적 존중(Unconditional Positive Regard) – 사용자를 판단하지 않고 존중합니다.\n"
            "3. 공감적 이해(Empathic Understanding) – 사용자의 내적 경험을 공감적으로 이해하고 반영합니다.\n\n"
            "안전 및 윤리적 한계:\n"
            "- 의학적 진단이나 법률적 조언은 제공하지 않습니다.\n\n [context]" 
        )

        formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT + context_text}]
        formatted_messages += [{"role": "user", "content": m} for m in messages]

        response = client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            temperature=0.7,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    else:
        save_tom(user_id,messages[-1])
        base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'signals')
        base_dir = os.path.abspath(base_dir)   # 절대경로로 변환

        signal_path = os.path.join(base_dir, user_id, f'{user_id}_signal.json')
        tom_path = os.path.join(base_dir, user_id, f'{user_id}_tom.json')

        with open(signal_path, 'r', encoding='utf-8-sig') as f:
            signal = json.load(f)

        with open(tom_path, 'r', encoding='utf-8-sig') as f:
            tom_list = json.load(f)


        save_memory(user_id,messages[-1], is_survey=False)


        basic_signal = signal['basic_signal']
        turn_signal_list= signal['turn_signal']

        ctx, block, query = demo_build_context( basic_signal, turn_signal_list, tom_list, messages[-1])
        examples, label_result = predict_labels(messages[-1], block)

        memory = save_memory(user_id,messages[-1])

        strategy_context = build_strategy_context_block(tom_list, memory)

        plan = run_strategy_planner(recent_user_text=messages[-1], label_result=label_result, examples=examples, context_block=strategy_context, ctx=ctx) 

        save_plan(user_id, plan)

        blueprint = plan.get("blueprint", plan)  # plan에 blueprint 키가 없으면 plan 자체를 사용

        gaps = find_ppppi_gaps(memory['overall_summary']['ppppi_synthesis'])
        
        question_idea =  question_ideation(memory,gaps,messages[-6:],topk_gaps= 5, num_candidates = 5,temperature= 0.4)
        save_question_idea(user_id, question_idea)


        draft_text = generate_response_from_spec(
        blueprint=blueprint,
        question_idea=question_idea,
        messages=messages,
        temperature=0.4,
        max_tokens=300)

        acts = [a.lower().strip() for a in (blueprint.get("plan", {}) or {}).get("speech_acts", [])]

        if 'question' in "".join(acts):
            pass
        else:
            print('revise_question')
            draft_text = enhance_with_question(basic_response=draft_text, blueprint=blueprint, messages = messages[-6:], question_idea=question_idea)

        
        final_text = refine_response_text(
            draft_text=draft_text,
            blueprint=blueprint,
            messages=messages,
            temperature=0.4,
            max_tokens=260
        )
        
        return final_text
