import streamlit as st
import tempfile
import os

from rag import process_pdf, ask_question, collection


# Page Configuration
st.set_page_config(
    page_title="Local RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_ready" not in st.session_state:
    st.session_state.document_ready = collection.count() > 0

if "document_name" not in st.session_state:
    st.session_state.document_name = ""


# Main Heading
st.title("📊 AI Document Chatbot")

st.write(
    "Upload your PDF and ask questions about its content. "
    "The chatbot uses RAG to retrieve relevant information."
)

st.divider()


# Sidebar - PDF Upload
with st.sidebar:
    st.header("📂 Document Upload")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    if st.button("🚀 Process Document"):
        if uploaded_file:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(uploaded_file.getvalue())
                file_path = temp_file.name

            try:
                with st.spinner("Processing document..."):
                    count = process_pdf(file_path)

                if count > 0:
                    st.session_state.document_ready = True
                    st.session_state.document_name = uploaded_file.name
                    st.session_state.messages = []

                    st.success("PDF processed successfully!")
                    st.info(f"Total chunks: {count}")

                else:
                    st.error("No readable text found in PDF.")

            except Exception as e:
                st.error(f"Processing error: {e}")

            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)

        else:
            st.warning("Please upload a PDF.")

    st.divider()

    # Document Status
    st.subheader("📊 Document Status")

    if st.session_state.document_ready:
        st.success("Document Ready")
        st.write(f"File: {st.session_state.document_name}")
    else:
        st.warning("No document uploaded")

    st.divider()

    # Clear Chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# Chat Interface
st.subheader("💬 Chat with Your Document")

if not st.session_state.messages:
    st.info("Upload a PDF and start asking questions.")


# Display Previous Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Get User Question
question = st.chat_input("Ask something about your document...")

if question:

    if not st.session_state.document_ready:
        st.warning("Please upload and process a PDF first.")

    else:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        # Generate Answer
        with st.chat_message("assistant"):
            with st.spinner("Finding relevant information..."):
                answer = ask_question(question)
                st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })


# Footer
st.divider()

st.caption(
    "Built with Python | Streamlit | ChromaDB | "
    "Sentence Transformers | Hugging Face"
)