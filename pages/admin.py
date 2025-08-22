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

    def make_enter_handler(target_room: str):
        def handler():
            # 새 탭에서 클릭이 발생한 "그" 요청의 SID로 관리자 바인딩
            current_sid = ui.context.client.id
            app.storage.sessions[current_sid] = 'admin'
            ui.navigate.to(f'/chat?room={target_room}&mode=C&sid={current_sid}', new_tab=True)
        return handler

    def make_showinfo_handler(target_uid: str):
        def handler():
            current_sid = ui.context.client.id
            app.storage.sessions[current_sid] = 'admin'
            mode = app.storage.chat_modes.get(target_uid, 'C')
            ui.navigate.to(f'/admin/show_info?uid={target_uid}&mode={mode}', new_tab=True)
        return handler

    # 사용자별 행
    for uid in c_mode_users:
        chat_id = app.storage.chat_rooms.get(uid)  # 예: 'chatroom-<uid>'
        with ui.row().classes('items-center my-2'):
            ui.label(f'{uid} (chatroom: {chat_id})')
            ui.button('Enter Chat', on_click=make_enter_handler(chat_id))
            ui.button('Show Info', on_click=make_showinfo_handler(uid))
            