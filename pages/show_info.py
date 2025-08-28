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
    base_dir = os.path.abspath(base_dir)   # �����η� ��ȯ

    filepath = os.path.join(base_dir, uid, f'{uid}_survey_C.json')

    if not os.path.exists(filepath):
        ui.label(f'���� ���� ����: {filepath}').classes('text-red-500')
        return
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)

    # ����ȣȯ: category �켱, ������ categories[0]
    category = data.get('category')
    if not category:
        cats = data.get('categories', [])
        category = cats[0] if isinstance(cats, list) and cats else ''

    with ui.card().classes('max-w-2xl w-full space-y-2'):
        ui.label(f'UID: {uid} / MODE: C').classes('font-medium')
        ui.label(f"ī�װ�: {category or '-'}")
        ui.label(f"����: {data.get('topic_text', '-')}")
        ui.label(f"SUS(���谨): {data.get('discomfort', '-')}")
        ui.label(f"������(Control): {data.get('control', '-')}")
        ui.label(f"�߿䵵(Importance): {data.get('importance', '-')}")
        
        ui.label('PANAS-NA')
        panas = data.get('panas_na', {})
        if not panas:
            ui.label('-')
        else:
            for k, v in panas.items():
                ui.label(f'- {k}: {v}')
