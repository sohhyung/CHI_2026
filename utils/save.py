# utils/save_chat.py
import json
import os


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