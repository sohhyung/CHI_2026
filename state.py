# state.py
from nicegui import app

# session_key (uuid str) �� user_id
app.storage.sessions = {}

# user_id �� mode
app.storage.chat_modes = {}

# user_id �� chat_id
app.storage.chat_rooms = {}

# chat_id �� message list
app.storage.messages = {}

# chat_id �� UI update callbacks
app.storage.listeners = {}
