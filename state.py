# state.py
from nicegui import app

# session_key (uuid str) ¡æ user_id
app.storage.sessions = {}

# user_id ¡æ mode
app.storage.chat_modes = {}

# user_id ¡æ chat_id
app.storage.chat_rooms = {}

# chat_id ¡æ message list
app.storage.messages = {}

# chat_id ¡æ UI update callbacks
app.storage.listeners = {}
