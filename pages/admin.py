# pages/admin.py
from nicegui import ui
from state import app

@ui.page('/admin')
def admin_page(sid: str):
    user_id = app.storage.sessions.get(sid)
    if user_id != 'admin':
        ui.notify('Admin access only')
        ui.navigate.to('/')
        return
    
    admin_sid = ui.context.client.id
    app.storage.sessions[admin_sid] = 'admin'
    
    ui.label('Admin Dashboard: C Mode Users').classes('text-xl font-bold mb-4')

    # Filter users in C mode
    c_mode_users = [
        uid for uid, mode in app.storage.chat_modes.items()
        if mode == 'C'
    ]

    if not c_mode_users:
        ui.label('No users are currently in C mode.').classes('text-gray-500')
        return

    # Show buttons for each user
    for uid in c_mode_users:
        chat_id = app.storage.chat_rooms.get(uid)
        with ui.row().classes('items-center my-2'):
            ui.label(f'{uid} (chatroom: {chat_id})')
            ui.button(
                'Enter Chat',
                on_click=lambda c=chat_id: ui.navigate.to(f'/chat?room={c}&mode=C&sid={admin_sid}')
            )
            