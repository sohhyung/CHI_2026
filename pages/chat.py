# pages/chat.py
from nicegui import ui
from datetime import datetime
from state import app
from utils.save_chat import save_chat_to_json


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
    
    def render_messages():
        chat_area.clear()

        chat_partner_id = chat_id.replace('chatroom-', '')

        with chat_area:
            for msg in app.storage.messages.get(chat_id, []):
                if user_id == 'admin':
                    prefix = 'You:' if msg['type'] == 'agent' else f'{chat_partner_id}:'
                else:
                    prefix = 'You:' if msg['type'] == 'user' else 'Agent:'

                ui.label(f'{prefix} {msg["text"]} ({msg["time"]})').classes('mb-1')


    ui.timer(1.0, render_messages)
    
    def send_message():
        text = message_input.value.strip()
        if not text:
            return

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sender_type = 'agent' if user_id == 'admin' else 'user'

        # 1. 사용자 메시지 저장
        app.storage.messages[chat_id].append({'type': sender_type, 'text': text, 'time': timestamp})
        save_chat_to_json(user_id, mode, app.storage.messages[chat_id])

        message_input.value = ''
        render_messages()

        # 2. 자동응답 (A, B 모드일 때만)
        if mode in ['A', 'B']:
            response = f'(Auto-response from Mode {mode})'
            app.storage.messages[chat_id].append({'type': 'agent', 'text': response, 'time': timestamp})
            save_chat_to_json(user_id, mode, app.storage.messages[chat_id])
            render_messages()
        elif mode == 'C':
            # Human-to-human mode, do nothing: admin will respond separately
            pass

    def exit_chat():
        ui.notify('Exiting chat...')
        ui.navigate.to(f'/select-mode?sid={sid}')
        ui.navigate.reload() 

    # Input area and buttons
    with ui.row().classes('w-full'):
        message_input.on('keydown.enter', lambda e: send_message())
        ui.button('Send', on_click=send_message)
        ui.button('Exit Chat', on_click=exit_chat).props('color=negative')

    # Initial render
    render_messages()
