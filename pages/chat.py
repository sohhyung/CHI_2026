# pages/chat.py
from nicegui import ui
from datetime import datetime
from state import app
from utils.save import save_chat_to_json
from utils.response_router import get_response
import time
import random
import asyncio


@ui.page('/chat')
def chat(room: str, mode: str, sid: str):
    # Step 1: Recover user_id from sid
    user_id = app.storage.sessions.get(sid)
    if not user_id:
        ui.notify('Invalid session')
        ui.navigate.to('/')
        return

    chat_id = room

    # Step 2: Register chat room and state if not already set
    if user_id not in app.storage.chat_rooms:
        app.storage.chat_rooms[user_id] = chat_id
    app.storage.messages.setdefault(chat_id, [])
    app.storage.listeners.setdefault(chat_id, [])

    # Step 3: Chat UI
    ui.label(f'Chatting as {user_id} in room {chat_id} [Mode {mode}]').classes('text-lg font-bold mb-4')
    chat_area = ui.column().classes('w-full h-96 overflow-auto border rounded p-2')
    message_input = ui.input(placeholder='Type your message...').classes('w-full')

    last_message_count = 0
    def render_messages():
        nonlocal last_message_count

        current_messages = app.storage.messages.get(chat_id, [])
        current_count = len(current_messages)
        chat_partner_id = chat_id.replace('chatroom-', '')
        chat_area.clear()

        with chat_area:
            for i, msg in enumerate(current_messages):
                if user_id == 'admin':
                    prefix = 'You:' if msg['type'] == 'agent' else f'{chat_partner_id}:'
                else:
                    prefix = 'You:' if msg['type'] == 'user' else 'Agent:'

                label_class = 'mb-1 scroll-anchor' if i == current_count - 1 else 'mb-1'
                ui.label(f'{prefix} {msg["text"]} ({msg["time"]})').classes(label_class)

        # ? �޽��� ���� �þ�� ���� ��ũ��
        if current_count > last_message_count:
            ui.run_javascript("""
                const anchor = document.querySelector('.scroll-anchor');
                if (anchor) {
                    anchor.scrollIntoView({ behavior: 'smooth' });
                }
            """)

        last_message_count = current_count






    ui.timer(1.0, render_messages)

    def send_message():
        text = message_input.value.strip()
        if not text:
            return

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sender_type = 'agent' if user_id == 'admin' else 'user'

        # 1. ����� �޽��� ����
        app.storage.messages[chat_id].append({'type': sender_type, 'text': text, 'time': timestamp})
        save_chat_to_json(user_id, mode, app.storage.messages[chat_id],'chat_logs')

        message_input.value = ''
        render_messages()

        # 2. �ڵ����� (A, B ����� ����)
        if mode in ['A', 'B']:
            async def generate_response():
                await asyncio.sleep(random.uniform(5, 10))
                history = [msg['text'] for msg in app.storage.messages[chat_id][-5:]]
                response = get_response(mode, user_id, history)
                timestamp2 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                app.storage.messages[chat_id].append({'type': 'agent', 'text': response, 'time': timestamp2})
                save_chat_to_json(user_id, mode, app.storage.messages[chat_id],'chat_logs')
                render_messages()

            asyncio.create_task(generate_response())
        elif mode == 'C':
            # Human-to-human mode, do nothing: admin will respond separately
            pass

    def exit_chat():
        ui.notify('Exiting chat...')

        if user_id == 'admin':
            ui.navigate.to(f'/admin?sid={sid}')
        else:
            ui.navigate.to(f'/select-mode?sid={sid}')

        ui.navigate.reload()


    # Input area and buttons
    with ui.row().classes('w-full'):
        message_input.on('keydown.enter', lambda e: send_message())
        ui.button('Send', on_click=send_message)

    # Initial render
    render_messages()
