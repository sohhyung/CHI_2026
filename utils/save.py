# utils/save_chat.py
import json
import os

from utils.models.custom_utils.state_builder import build_baseline_signals,build_turn_signals
from utils.models.custom_utils.tom_reasoner import process_turn
from utils.models.custom_utils.memory_manager import create_empty_memory, build_memory

import json, os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional


def _now_kst_iso() -> str:
    KST = timezone(timedelta(hours=9))
    return datetime.now(KST).isoformat(timespec="seconds")

def save_chat_to_json(user_id: str, mode: str, messages: list, BASE_DIR = 'chat_logs'):
    os.makedirs(BASE_DIR, exist_ok=True)
    user_dir = os.path.join(BASE_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

    filename = f'{user_id}_chat_{mode}.json'
    filepath = os.path.join(user_dir, filename)

    try:
        with open(filepath, 'w', encoding='utf-8-sig') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f'[save_chat_to_json] Failed to save {filepath}: {e}')



def save_user_survey(user_id: str, mode: str, survey_dict: dict, BASE_DIR: str = 'survey_data'):
    """
    survey_dict 예시:
    {
      "categories": ["감정/스트레스"],
      "topic_text": "최근에 ...",
      "discomfort": 3,
      "panas_na": {"짜증난다": 2, "긴장된다": 4, ...}
    }
    """
    os.makedirs(BASE_DIR, exist_ok=True)
    user_dir = os.path.join(BASE_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

    filename = f'{user_id}_survey_{mode}.json'
    filepath = os.path.join(user_dir, filename)

    try:
        with open(filepath, 'w', encoding='utf-8-sig') as f:
            json.dump(survey_dict, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f'[save_user_survey] Failed to save {filepath}: {e}')

def save_initial_signal_from_survey(user_id: str, survey_result: dict, BASE_DIR:str='signals') :
    os.makedirs(BASE_DIR,exist_ok=True)

    user_dir = os.path.join(BASE_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    filename = f'{user_id}_signal.json'
    filepath = os.path.join(user_dir,filename)


    basic_signal = build_baseline_signals(survey_result)


    signal = {
        "basic_signal": basic_signal,
        "turn_signal": [build_turn_signals((survey_result.get("topic_text") or "").strip())],
    }

    with open(filepath, "w", encoding="utf-8-sig") as f:
        json.dump(signal, f, ensure_ascii=False, indent=2)

def save_signal(user_id: str, response: str, BASE_DIR: str = 'signals'):
    # 파일 경로 정의
    user_dir = os.path.join(BASE_DIR, user_id)
    filename = f"{user_id}_signal.json"
    filepath = os.path.join(user_dir, filename)

    # 기존 파일 읽기 (무조건 존재한다고 가정)
    with open(filepath, "r", encoding="utf-8-sig") as f:
        signal = json.load(f)

    # turn_signal에 새로운 응답 추가
    signal["turn_signal"].append(build_turn_signals((response or "").strip()))

    # 파일 다시 저장
    with open(filepath, "w", encoding="utf-8-sig") as f:
        json.dump(signal, f, ensure_ascii=False, indent=2)


def save_initial_tom_from_survey(user_id: str,raw_text: str, BASE_DIR: str = "signals"):

    user_dir = os.path.join(BASE_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

    signal_path = os.path.join(user_dir, f"{user_id}_signal.json")
    tom_path = os.path.join(user_dir, f"{user_id}_tom.json")

    with open(signal_path, "r", encoding="utf-8-sig") as f:
        signal = json.load(f)

    basic_signal = signal["basic_signal"]
    first_turn = signal["turn_signal"][0]

    # 2) 1회차 처리
    turn_record = process_turn( basic_signal=basic_signal,turn_signal=first_turn, raw_text=raw_text.strip(),ppppi_prev=None)
    out = [turn_record]

    with open(tom_path, "w", encoding="utf-8-sig") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def save_tom(user_id, raw_text, BASE_DIR='signals'):
    user_dir = os.path.join(BASE_DIR, user_id)

    # 기존 파일 읽기 (무조건 존재한다고 가정)
    signal_path = os.path.join(user_dir, f"{user_id}_signal.json")
    tom_path = os.path.join(user_dir, f"{user_id}_tom.json")

    with open(signal_path, "r", encoding="utf-8-sig") as f:
        signal = json.load(f)
    
    basic_signal = signal["basic_signal"]
    last_turn = signal["turn_signal"][-1]

    with open(tom_path, "r", encoding="utf-8-sig") as f:
        tom = json.load(f)

    tom.append(process_turn( basic_signal=basic_signal,turn_signal=last_turn, raw_text=raw_text.strip(),ppppi_prev=tom[-1]))

    with open(tom_path, "w", encoding="utf-8-sig") as f:
        json.dump(tom, f, ensure_ascii=False, indent=2)

def save_plan(user_id, plan, BASE_DIR='signals'):
    user_dir = os.path.join(BASE_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

    plan_path = os.path.join(user_dir, f"{user_id}_plan.json")

    if os.path.exists(plan_path):
        with open(plan_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        # 혹시 과거에 단일 dict로 저장된 경우도 대비
        plans = data
    else:
        plans = []

    plans.append(plan)

    with open(plan_path, "w", encoding="utf-8-sig") as f:
        json.dump(plans, f, ensure_ascii=False, indent=2)


def save_memory(user_id: str,raw_text: str, BASE_DIR: str = "signals", is_survey=False):
    user_dir = os.path.join(BASE_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

    tom_path = os.path.join(user_dir, f"{user_id}_tom.json")
    memory_path = os.path.join(user_dir, f"{user_id}_memory.json")

    if is_survey:
        memory = create_empty_memory()
    else:
        with open(memory_path, "r", encoding="utf-8-sig") as f:
            memory= json.load(f)


    with open(tom_path, "r", encoding="utf-8-sig") as f:
        tom = json.load(f)

    turn_tom = tom[-1]

    memory = build_memory(raw_text, turn_tom, memory)

    with open(memory_path, "w", encoding="utf-8-sig") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

    return memory


def save_question_idea(user_id, question_idea, BASE_DIR: str = "signals"):
    user_dir = os.path.join(BASE_DIR, user_id)
    filepath = os.path.join(user_dir, f"{user_id}_question.json")
    
    # 사용자 디렉토리가 없으면 생성
    os.makedirs(user_dir, exist_ok=True)
    
    # 기존 파일이 있으면 불러오고, 없으면 빈 리스트로 시작
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            questions = json.load(f)
    else:
        questions = []
    
    # 새 question_idea 추가
    questions.append(question_idea)
    
    # 파일에 저장
    with open(filepath, 'w', encoding='utf-8-sig') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)