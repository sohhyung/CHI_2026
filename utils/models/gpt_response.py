# -*- coding: cp949 -*-
import os
import openai
from openai import OpenAI

from .utils import set_gpt_model, set_openai_api_key
import time
import random

# Initialize OpenAI API key
set_openai_api_key()



def get_gpt_response(messages):
    """
    Generates a response from GPT-4 based on the latest chat messages.
    
    Parameters:
        messages (list[str]): Recent user and agent messages (as plain text)

    Returns:
        str: GPT-4 generated response
    """
    try:
        # �ý��� ������Ʈ (����� ���� - �߸���, ģ��)
        system_prompt = (
            "����� �߸����̰� ģ���� ������Դϴ�. "
            "������� ������ �����ϰ�, ����ϰ� ��ȭ�� �̲��� ������. "
            "���� �м��̳� ���𺸴ٴ�, ����ڰ� �ڽ��� ������ �����Ӱ� ǥ���� �� �ֵ��� �����ּ���."
        )

        # �޽��� ���� ���� (�ֱ� 5��)
        formatted_messages = [{"role": "system", "content": system_prompt}] + [
            {"role": "user", "content": m} for m in messages[-5:]
        ]
        
        model = set_gpt_model()
        client = OpenAI()
        # GPT ȣ��
        response = client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[get_gpt_response] Error: {e}")
        return "(GPT ���� ���� �� ������ �߻��߽��ϴ�.)"
