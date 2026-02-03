import streamlit as st
import requests

API_BASE = "http://localhost:8001"

st.set_page_config(
    page_title="College Manual QA",
    page_icon="📘",
    layout="wide",
)

st.title("📘 College Manual RAG Assistant")

# Session state for chat history (frontend)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "manual_indexed" not in st.session_state:
    st.session_state["manual_indexed"] = False


st.sidebar.header("Manual PDF")
uploaded_file = st.sidebar.file_uploader(
    "Upload college manual (PDF)", type=["pdf"]
)
st.sidebar.subheader("📦 Index Management")

if st.sidebar.button("💾 Save Index"):
    with st.spinner("Saving index..."):
        try:
            r = requests.post(f"{API_BASE}/api/save-index")
            if r.status_code == 200:
                st.sidebar.success("Index saved locally.")
            else:
                st.sidebar.error(r.text)
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

if st.sidebar.button("📂 Load Saved Index"):
    with st.spinner("Loading index..."):
        try:
            r = requests.post(f"{API_BASE}/api/load-index")
            if r.status_code == 200:
                st.session_state["manual_indexed"] = True
                st.sidebar.success("Loaded saved index!")
            else:
                st.sidebar.error(r.text)
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

if st.sidebar.button("Upload & Index") and uploaded_file is not None:
    with st.spinner("Uploading and indexing PDF..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        try:
            resp = requests.post(f"{API_BASE}/api/upload-manual", files=files)
            if resp.status_code == 200:
                data = resp.json()
                st.sidebar.success(
                    f"Indexed: {data.get('filename', 'manual')}"
                )
                st.session_state["manual_indexed"] = True
                st.session_state["messages"] = []  # reset chat
            else:
                st.sidebar.error(f"Upload failed: {resp.text}")
        except Exception as e:
            st.sidebar.error(f"Error contacting backend: {e}")


if st.session_state["manual_indexed"]:
    st.sidebar.success("Manual is loaded ✅")
else:
    st.sidebar.info("Upload and index a manual to start.")


st.markdown("### 🗣 Chat with the manual")

# Display existing messages
for msg in st.session_state["messages"]:
    role = msg["role"]
    content = msg["content"]
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content)

# Chat input
user_input = st.chat_input("Ask a question about the college manual...")

if user_input:
    if not st.session_state["manual_indexed"]:
        st.warning("Please upload and index a manual first.")
    else:
        # Show user message immediately
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Call backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    payload = {"question": user_input, "reset_history": False}
                    resp = requests.post(f"{API_BASE}/api/ask", json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data["answer"]
                        chunks = data.get("chunks", [])
                    else:
                        answer = f"Error: {resp.text}"
                        chunks = []
                except Exception as e:
                    answer = f"Error contacting backend: {e}"
                    chunks = []

                st.markdown(answer)
                st.session_state["messages"].append({"role": "assistant", "content": answer})

                if chunks:
                    with st.expander("📄 Retrieved Relevant Document Sections"):
                        for i, c in enumerate(chunks, start=1):
                            st.markdown(f"**Chunk {i}:**")
                            st.write(c)
                            st.markdown("---")


st.caption(
    "Powered by Gemini + FAISS + Streamlit. Answers are strictly from the uploaded manual."
)
