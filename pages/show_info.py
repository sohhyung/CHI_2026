# pages/admin.py
from nicegui import ui
from state import app


@ui.page('/admin/show_info')
def admin_show_info_page(uid: str = '', room: str = ''):

    ui.label('Admin: User Info').classes('text-xl font-bold mb-4')

    with ui.card().classes('max-w-xl'):
        ui.label(f'UID: {uid}')