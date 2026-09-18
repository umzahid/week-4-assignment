import os
from datetime import datetime

import streamlit as st

from rag import chat, store
from rag.embeddings import embed_texts
from rag.eval import compute_metrics
from rag.lifecycle import sync
from rag.llm import OllamaError

st.set_page_config(page_title="Week 4 RAG Assignment", page_icon="📁", layout="wide")

FOLDER_DEFAULT = "sample_docs"
MANIFEST_PATH = "data/manifest.json"

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Week 4 RAG Assignment — folder-watching chat")

with st.sidebar:
    st.header("Watched folder")
    folder_path = st.text_input("Folder path", value=FOLDER_DEFAULT)
    folder_exists = os.path.isdir(folder_path)
    collection = store.get_collection()

    if not folder_exists:
        st.warning(f"Folder not found: {folder_path}")
    else:

        @st.fragment(run_every=5)
        def watch_folder():
            summary = sync(collection, folder_path, MANIFEST_PATH)
            st.caption(f"Last checked: {datetime.now().strftime('%H:%M:%S')}")
            if summary["added"]:
                st.write(f"✅ Added: {', '.join(summary['added'])}")
            if summary["modified"]:
                st.write(f"♻️ Updated: {', '.join(summary['modified'])}")
            if summary["deleted"]:
                st.write(f"❌ Removed: {', '.join(summary['deleted'])}")
            st.metric("Chunks indexed", collection.count())

        watch_folder()

chat_ready = folder_exists and collection.count() > 0

if not chat_ready:
    st.info("Point the sidebar at a folder containing .pdf/.docx files to start chatting.")


def render_report_card(sources: list[dict], metrics: dict) -> None:
    cols = st.columns(3)
    cols[0].metric("Retrieval relevance", f"{metrics['retrieval_relevance']:.2f}")
    cols[1].metric("Groundedness", f"{metrics['groundedness']:.2f}")
    cols[2].metric("Answer relevance", f"{metrics['answer_relevance']:.2f}")
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for i, source in enumerate(sources, start=1):
                st.markdown(f"**[{i}] {source['source']}** (chunk {source['chunk_index']})")
                st.caption(source["text"])


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("metrics"):
            render_report_card(message["sources"], message["metrics"])

question = st.chat_input("Ask a question about your documents...", disabled=not chat_ready)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = chat.ask(collection, question)
                metrics = compute_metrics(question, result["answer"], result["sources"], embed_texts)
                st.write(result["answer"])
                render_report_card(result["sources"], metrics)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "metrics": metrics,
                    }
                )
            except OllamaError as exc:
                error_message = f"Ollama isn't ready: {exc}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
