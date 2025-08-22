from nicegui import ui
from pages import login, select_mode, admin, chat, show_info, survey
ui.run(host='0.0.0.0', port=8000)
ui.run(title='Multi-Mode Chat App')
