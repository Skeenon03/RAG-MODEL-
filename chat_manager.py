import os
import json
import datetime

CHAT_HISTORY_FILE = "chat_sessions.json"

def load_chat_sessions():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_chat_sessions(data):
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def create_new_chat():
    chat_id = str(datetime.datetime.now().timestamp())
    sessions = load_chat_sessions()
    sessions[chat_id] = {"title": "New Chat", "messages": []}
    save_chat_sessions(sessions)
    return chat_id

def generate_title(text):
    return " ".join(text.split()[:5])