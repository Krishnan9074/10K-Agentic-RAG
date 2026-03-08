

import time
import uuid
import streamlit as st
from rag import RagService

MAX_REQUESTS_PER_MINUTE = 20


st.set_page_config(page_title="RAG QA System", layout="wide")

st.title("RAG Question Answering System")



model_choice = st.selectbox(
    "Select Groq Model",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]
)


if "current_model" not in st.session_state:
    st.session_state["current_model"] = None

if "rag_service" not in st.session_state:
    st.session_state["rag_service"] = None

# Reinitialize if model selection changes
if st.session_state["current_model"] != model_choice:

    with st.spinner("Initializing model..."):
        st.session_state["rag_service"] = RagService(model_type=model_choice)

    st.session_state["current_model"] = model_choice
    st.success("Model initialized successfully!")



# Unique session ID per browser session (fixes shared chat history)
if "session_id" not in st.session_state:
    st.session_state["session_id"] = "user_" + uuid.uuid4().hex[:12]

# Rate-limiting state
if "request_timestamps" not in st.session_state:
    st.session_state["request_timestamps"] = []

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question...")

if user_input:

    # Rate limiting: max requests per minute per session
    now = time.time()
    st.session_state["request_timestamps"] = [
        t for t in st.session_state["request_timestamps"] if now - t < 60
    ]
    if len(st.session_state["request_timestamps"]) >= MAX_REQUESTS_PER_MINUTE:
        st.error("Too many requests. Please wait a moment before asking another question.")
        st.stop()
    st.session_state["request_timestamps"].append(now)

    # Display user message
    st.session_state["messages"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Call the model
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            session_config = {
                "configurable": {
                    "session_id": st.session_state["session_id"]
                }
            }

            response = st.session_state["rag_service"].chain.invoke(
                {"input": user_input},
                config=session_config
            )

            st.markdown(response)

    st.session_state["messages"].append(
        {"role": "assistant", "content": response}
    )