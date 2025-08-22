# pages/select_mode.py
from nicegui import ui
from state import app

@ui.page('/select-mode')
def select_mode(sid: str):
    print('Received sid:', sid)
    print('All sessions:', app.storage.sessions)

    user_id = app.storage.sessions.get(sid)
    print('Retrieved user_id:', user_id)

    if not user_id:
        ui.notify('Invalid session')
        ui.navigate.to('/')
        return

    ui.label(f'Welcome, {user_id}').classes('text-lg font-bold mb-4')

    mode_select = ui.select(
        options=['A (GPT)', 'B (Custom Model)', 'C (Chat with Human)'],
        label='Select Chat Mode',
    ).classes('w-full')

    def on_submit():
        if not mode_select.value:
            ui.notify('Please select a mode')
            return

        mode_letter = mode_select.value[0]  # 'A', 'B', or 'C'
        chat_id = f'chatroom-{user_id}'

        # Save mode and chat room
        app.storage.chat_modes[user_id] = mode_letter
        app.storage.chat_rooms[user_id] = chat_id

        if chat_id in app.storage.messages:
            app.storage.messages[chat_id].clear()

        # Navigate to chat page
        ui.navigate.to(f'/survey/start?room={chat_id}&mode={mode_letter}&sid={sid}')
        ui.navigate.reload() 

    ui.button('Start Chat', on_click=on_submit).classes('mt-4')
