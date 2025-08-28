# -*- coding: cp949 -*-
# pages/admin.py
from nicegui import ui
from state import app
import json
import os


@ui.page('/admin/show_info')
def admin_show_info_page(uid: str = '', room: str = ''):

    ui.label('Admin: User Info').classes('text-xl font-bold mb-4')
    
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'survey_data')
    base_dir = os.path.abspath(base_dir)   # 절대경로로 변환

    filepath = os.path.join(base_dir, uid, f'{uid}_survey_C.json')

    if not os.path.exists(filepath):
        ui.label(f'설문 파일 없음: {filepath}').classes('text-red-500')
        return
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)

    # 하위호환: category 우선, 없으면 categories[0]
    category = data.get('category')
    if not category:
        cats = data.get('categories', [])
        category = cats[0] if isinstance(cats, list) and cats else ''

    with ui.card().classes('max-w-2xl w-full space-y-2'):
        ui.label(f'UID: {uid} / MODE: C').classes('font-medium')
        ui.label(f"카테고리: {category or '-'}")
        ui.label(f"주제: {data.get('topic_text', '-')}")
        ui.label(f"SUS(불쾌감): {data.get('discomfort', '-')}")
        ui.label(f"통제감(Control): {data.get('control', '-')}")
        ui.label(f"중요도(Importance): {data.get('importance', '-')}")
        
        ui.label('PANAS-NA')
        panas = data.get('panas_na', {})
        if not panas:
            ui.label('-')
        else:
            for k, v in panas.items():
                ui.label(f'- {k}: {v}')
