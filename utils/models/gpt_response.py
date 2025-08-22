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
        # 시스템 프롬프트 (상담자 역할 - 중립적, 친절)
        system_prompt = (
            "당신은 중립적이고 친절한 상담자입니다. "
            "사용자의 감정을 존중하고, 편안하게 대화를 이끌어 가세요. "
            "깊은 분석이나 조언보다는, 사용자가 자신의 감정을 자유롭게 표현할 수 있도록 도와주세요."
        )

        # 메시지 구조 생성 (최근 5개)
        formatted_messages = [{"role": "system", "content": system_prompt}] + [
            {"role": "user", "content": m} for m in messages[-5:]
        ]
        
        model = set_gpt_model()
        client = OpenAI()
        # GPT 호출
        response = client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[get_gpt_response] Error: {e}")
        return "(GPT 응답 생성 중 오류가 발생했습니다.)"
