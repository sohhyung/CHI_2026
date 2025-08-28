# -*- coding: cp949 -*-
import json
import os
from openai import OpenAI

from .utils import set_gpt_model, set_openai_api_key

set_openai_api_key()

# 시스템 프롬프트 (Rogers 3원칙 + 윤리적 한계)
SYSTEM_PROMPT = (
            "당신은 상담자 역할을 맡은 챗봇입니다. "
            "다음의 지침을 모든 대화에서 따르세요.\n\n"
            "상담 원칙 (Carl Rogers, 1957):\n"
            "1. 진실성(Congruence) ? 진실되고 위선 없는 태도를 유지합니다.\n"
            "2. 무조건적 긍정적 존중(Unconditional Positive Regard) ? 사용자를 판단하지 않고 존중합니다.\n"
            "3. 공감적 이해(Empathic Understanding) ? 사용자의 내적 경험을 공감적으로 이해하고 반영합니다.\n\n"
            "안전 및 윤리적 한계:\n"
            "- 의학적 진단이나 법률적 조언은 제공하지 않습니다."
        )

def load_survey_context(user_id: str, base_dir="survey_data"):
    """
    주어진 user_id의 설문 JSON을 읽어서 상담 챗봇 context 문장을 생성.
    """
    path = os.path.join(base_dir, user_id, f"{user_id}_survey_A.json")
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except FileNotFoundError:
        return ""

    # ---- 최소 버전 (category + topic_text) ----
    category = data.get("category", "")
    topic = data.get("topic_text", "")
    context_text = f"[사용자 사전 정보] 주제: {category}, 세부 내용: {topic}"

    # ---- 확장 버전 (원하면 전체 활용) ----
    # discomfort = data.get("discomfort")
    # control = data.get("control")
    # importance = data.get("importance")
    # panas = data.get("panas_na", {})
    # context_text += f"\n불편감: {discomfort}/100, 통제감: {control}/3, 중요도: {importance}/6"
    # if panas:
    #     emotions = ", ".join([f"{k}:{v}" for k, v in panas.items()])
    #     context_text += f"\n정서 상태: {emotions}"

    return context_text

def get_gpt_response(user_id, messages):
    """
    GPT 응답 생성: user_id 기반 사전 설문 context를 프롬프트에 포함.
    """
    try:
        survey_context = load_survey_context(user_id)

        formatted_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        if survey_context:
            formatted_messages.append({"role": "assistant", "content": survey_context})

        formatted_messages += [{"role": "user", "content": m} for m in messages[-5:]]

        model = set_gpt_model()
        client = OpenAI()

        response = client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            temperature=0.7,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[get_gpt_response] Error: {e}")
        return "(GPT 응답 생성 중 오류가 발생했습니다.)"

