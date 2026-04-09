import streamlit as st
import os
import datetime
import json
import ingestion
import time

from chat_manager import *
from rag_engine import *

st.markdown("""
<style>

/* ---------- THEME ---------- */
.stApp {
    background: linear-gradient(135deg, #eef3ff, #f7f9fc);
    color: #1a1a1a;
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e0e6f0;
    padding: 10px;
}

/* ---------- CHAT BUBBLES ---------- */
[data-testid="stChatMessage"] {
    padding: 12px;
    border-radius: 14px;
    margin-bottom: 10px;
    max-width: 75%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    animation: fadeIn 0.4s ease-in-out;
}

/* Assistant (left) */
[data-testid="stChatMessage"]:not(:has(div[data-testid="stMarkdownContainer"] p strong)) {
    background-color: #ffffff;
    color: #333;
}

/* User (right) */
[data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"]) {
    background-color: #4a90e2;
    color: white;
    margin-left: auto;
}

/* ---------- ANIMATION ---------- */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ---------- BUTTONS ---------- */
button {
    background-color: #4a90e2 !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    transition: all 0.2s ease !important;
}

button:hover {
    background-color: #357abd !important;
    transform: scale(1.05);
}

/* ---------- INPUT BOX ---------- */
textarea {
    border-radius: 10px !important;
    border: 1px solid #d0d7e2 !important;
}

textarea:focus {
    border: 1px solid #4a90e2 !important;
    box-shadow: 0 0 6px rgba(74, 144, 226, 0.4);
}

/* ---------- SCROLLBAR ---------- */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-thumb {
    background: #c5d3f5;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_FOLDER = os.path.join(BASE_DIR, "docs")
LOG_FILE = "chat_history_log.txt"
FEEDBACK_FILE = "verified_answers.json"

os.makedirs(DOCS_FOLDER, exist_ok=True)

# ---------------- PROMPTS ----------------

STANDARD_PROMPT = """You are a strict data analyst. Use ONLY the provided context to answer the question.

RULES:
1. If the answer is found in the context, summarize it clearly.
2. If the answer is NOT in the context, simply state:
   "I cannot find this information in the provided documents."
3. Do not make up answers.
4. Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""

RETRY_PROMPT = """The previous answer was incorrect.

TASK:
Re-answer the question strictly using ONLY the provided context.
Do NOT use outside knowledge.

Context:
{context}

Question:
{question}

Corrected Answer:"""

DETAIL_PROMPT = """The previous answer lacked detail.

TASK:
Answer again with ALL relevant details from context only.

Context:
{context}

Question:
{question}

Improved Answer:"""

# ---------------- FEEDBACK ----------------

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_feedback(question, answer):
    data = load_feedback()
    data[question.lower().strip()] = answer
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def save_to_log(user_text, ai_text):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] User: {user_text}\n")
        f.write(f"[{timestamp}] AI: {ai_text}\n")
        f.write("-" * 40 + "\n")

# ---------------- PAGE ----------------

st.set_page_config(page_title="Pushpas AI", page_icon="📄", layout="wide")
st.title("RAG SLAVEEE𓀓𓀝")

# ---------------- SESSION ----------------

if "chat_id" not in st.session_state:
    st.session_state.chat_id = create_new_chat()

if "messages" not in st.session_state:
    sessions = load_chat_sessions()
    st.session_state.messages = sessions.get(st.session_state.chat_id, {}).get("messages", [])

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.header("📂 Manage your personal filess")

    uploaded_files = st.file_uploader("Upload", accept_multiple_files=True)

    if st.button("Process & Embedd"):
        if uploaded_files:
            for file in uploaded_files:
                with open(os.path.join(DOCS_FOLDER, file.name), "wb") as f:
                    f.write(file.getbuffer())
            ingestion.main()
            st.cache_resource.clear()
            st.success("Done! ✅")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        sessions = load_chat_sessions()
        sessions[st.session_state.chat_id]["messages"] = []
        save_chat_sessions(sessions)
        st.rerun()

    st.markdown("---")
    st.header("💬 Chat History")

    sessions = load_chat_sessions()

    if st.button("➕ New Chat"):
        st.session_state.chat_id = create_new_chat()
        st.session_state.messages = []
        st.rerun()

    for cid, data in sessions.items():
        if st.button(data["title"], key=cid):
            st.session_state.chat_id = cid
            st.session_state.messages = data["messages"]
            st.rerun()

# ---------------- MAIN ----------------

vectorstore, llm = load_resources()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- INPUT ----------------

if question := st.chat_input("Ask a question..."):

    st.session_state.messages.append({"role": "user", "content": question})

    sessions = load_chat_sessions()
    sessions[st.session_state.chat_id]["messages"] = st.session_state.messages

    if len(st.session_state.messages) == 1:
        sessions[st.session_state.chat_id]["title"] = generate_title(question)

    save_chat_sessions(sessions)

    with st.chat_message("user"):
        st.markdown(question)

    verified_data = load_feedback()
    clean_q = question.lower().strip()

    if clean_q in verified_data:
        ans = verified_data[clean_q]

        with st.chat_message("assistant"):
            st.markdown(ans)

        st.session_state.messages.append({"role": "assistant", "content": ans})

    elif vectorstore and llm:
        response = generate_answer(question, vectorstore, llm, STANDARD_PROMPT, k_val=5)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.last_answer = {"q": question, "a": response}

        save_to_log(question, response)

    else:
        st.error("Upload documents first.")

    sessions = load_chat_sessions()
    sessions[st.session_state.chat_id]["messages"] = st.session_state.messages
    save_chat_sessions(sessions)

# ---------------- FEEDBACK ----------------

if st.session_state.last_answer:
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("✅ Save"):
            save_feedback(st.session_state.last_answer["q"], st.session_state.last_answer["a"])
            st.session_state.last_answer = None
            st.rerun()

    with c2:
        if st.button("🔍 Detail"):
            q = st.session_state.last_answer["q"]
            new_ans = generate_answer(q, vectorstore, llm, DETAIL_PROMPT, k_val=8)
            st.session_state.messages.append({"role": "assistant", "content": new_ans})
            st.session_state.last_answer = {"q": q, "a": new_ans}
            st.rerun()

    with c3:
        if st.button("❌ Retry"):
            q = st.session_state.last_answer["q"]
            new_ans = generate_answer(q, vectorstore, llm, RETRY_PROMPT, k_val=5)
            st.session_state.messages.append({"role": "assistant", "content": new_ans})
            st.session_state.last_answer = {"q": q, "a": new_ans}
            st.rerun()