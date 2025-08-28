# -*- coding: cp949 -*-
import json
import os
from openai import OpenAI

from .utils import set_gpt_model, set_openai_api_key

set_openai_api_key()

# �ý��� ������Ʈ (Rogers 3��Ģ + ������ �Ѱ�)
SYSTEM_PROMPT = (
            "����� ����� ������ ���� ê���Դϴ�. "
            "������ ��ħ�� ��� ��ȭ���� ��������.\n\n"
            "��� ��Ģ (Carl Rogers, 1957):\n"
            "1. ���Ǽ�(Congruence) ? ���ǵǰ� ���� ���� �µ��� �����մϴ�.\n"
            "2. �������� ������ ����(Unconditional Positive Regard) ? ����ڸ� �Ǵ����� �ʰ� �����մϴ�.\n"
            "3. ������ ����(Empathic Understanding) ? ������� ���� ������ ���������� �����ϰ� �ݿ��մϴ�.\n\n"
            "���� �� ������ �Ѱ�:\n"
            "- ������ �����̳� ������ ������ �������� �ʽ��ϴ�."
        )

def load_survey_context(user_id: str, base_dir="survey_data"):
    """
    �־��� user_id�� ���� JSON�� �о ��� ê�� context ������ ����.
    """
    path = os.path.join(base_dir, user_id, f"{user_id}_survey_A.json")
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except FileNotFoundError:
        return ""

    # ---- �ּ� ���� (category + topic_text) ----
    category = data.get("category", "")
    topic = data.get("topic_text", "")
    context_text = f"[����� ���� ����] ����: {category}, ���� ����: {topic}"

    # ---- Ȯ�� ���� (���ϸ� ��ü Ȱ��) ----
    # discomfort = data.get("discomfort")
    # control = data.get("control")
    # importance = data.get("importance")
    # panas = data.get("panas_na", {})
    # context_text += f"\n����: {discomfort}/100, ������: {control}/3, �߿䵵: {importance}/6"
    # if panas:
    #     emotions = ", ".join([f"{k}:{v}" for k, v in panas.items()])
    #     context_text += f"\n���� ����: {emotions}"

    return context_text

def get_gpt_response(user_id, messages):
    """
    GPT ���� ����: user_id ��� ���� ���� context�� ������Ʈ�� ����.
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
        return "(GPT ���� ���� �� ������ �߻��߽��ϴ�.)"

