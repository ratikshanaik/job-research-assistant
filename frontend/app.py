import streamlit as st
import requests

st.set_page_config(page_title="AI/ML Job Research Assistant", page_icon="🔍")
st.title("🔍 AI/ML Job Research Assistant")
st.caption("Ask questions about live AI/ML job postings, powered by RAG. Data sourced from Adzuna.")

BACKEND_URL = "http://127.0.0.1:8000/ask"

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask about AI/ML job postings...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching job postings..."):
            try:
                response = requests.post(BACKEND_URL, json={"question": question})
                data = response.json()
                answer = data["answer"]
                sources = data["sources"]

                st.markdown(answer)
                with st.expander("Sources"):
                    for s in sources:
                        st.write(f"- {s}")

                full_reply = answer + "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)
                st.session_state.messages.append({"role": "assistant", "content": full_reply})
            except Exception as e:
                st.error(f"Something went wrong: {e}")