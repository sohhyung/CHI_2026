# utils/save_chat.py
import json
import os


BASE_DIR = 'chat_logs'
os.makedirs(BASE_DIR, exist_ok=True)

def save_chat_to_json(user_id: str, mode: str, messages: list):
    user_dir = os.path.join(BASE_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

    filename = f'{user_id}_chat_{mode}.json'
    filepath = os.path.join(user_dir, filename)

    try:
        with open(filepath, 'w', encoding='utf-8-sig') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f'[save_chat_to_json] Failed to save {filepath}: {e}')
