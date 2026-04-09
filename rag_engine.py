import os
import time
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "db", "faiss_index")

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-large")

    try:
        vectorstore = FAISS.load_local(
            DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)

        return vectorstore, llm

    except:
        return None, None


# ---------------- GENERATE ANSWER ----------------

def generate_answer(question, vectorstore, llm, prompt_template_string, k_val=4):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_val})
    docs = retriever.invoke(question)

    if not docs:
        return "I cannot find that information in your documents."

    context_text = "\n\n".join(d.page_content[:1500] for d in docs)

    prompt = ChatPromptTemplate.from_template(prompt_template_string)
    chain = prompt | llm | StrOutputParser()

    with st.chat_message("assistant"):
        status = st.empty()
        placeholder = st.empty()
        full_response = ""

        # ---------- STEP 1: THINKING ----------
        status.markdown("""
        🤖 Thinking<span class="dots"></span>

        <style>
        .dots::after {
            content: '';
            animation: dots 1.5s steps(3, end) infinite;
        }
        @keyframes dots {
            0% { content: ''; }
            33% { content: '.'; }
            66% { content: '..'; }
            100% { content: '...'; }
        }
        </style>
        """, unsafe_allow_html=True)

        time.sleep(0.4)

        # ---------- STEP 2: RETRIEVAL ----------
        status.markdown("📄 Searching documents...")
        time.sleep(0.4)

        # ---------- STEP 3: GENERATION ----------
        status.markdown("✍️ Generating answer...")

        # ---------- STREAM OUTPUT ----------
        for chunk in chain.stream({
            "context": context_text,
            "question": question
        }):
            full_response += chunk
            placeholder.markdown(full_response + "▌")

        # ---------- FINAL OUTPUT ----------
        status.empty()
        placeholder.markdown(full_response)

        # ---------- SOURCES ----------
        with st.expander("View Sources"):
            for d in docs:
                src = os.path.basename(d.metadata.get("source", "Unknown"))
                st.caption(f"{src} (Page {d.metadata.get('page', '?')})")

    return full_response