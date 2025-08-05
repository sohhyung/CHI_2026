from nicegui import ui
from state import app
import uuid

@ui.page('/')
def login():
    id_input = ui.input('Enter your ID')

    def on_login():
        user_id = id_input.value.strip()
        if not user_id:
            ui.notify('Please enter an ID')
            return

        # generate session key
        session_key = str(uuid.uuid4())
        app.storage.sessions[session_key] = user_id

        if user_id == 'admin':
            ui.navigate.to(f'/admin?sid={session_key}')
        else:
            ui.navigate.to(f'/select-mode?sid={session_key}')

        ui.navigate.reload() 

    ui.button('Log In', on_click=on_login)