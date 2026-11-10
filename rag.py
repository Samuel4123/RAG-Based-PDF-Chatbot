import streamlit as st
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# Load AI Models
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    chatbot = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

    return embedding_model, chatbot


# Initialize ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="knowledge_base"
)


# Split Text into Chunks
def split_text(text, chunk_size=500, overlap=100):
    chunks = []

    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# Process PDF
def process_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"

    if not text.strip():
        return 0

    chunks = split_text(text)
    embedding_model, _ = load_models()

    embeddings = embedding_model.encode(chunks).tolist()

    old_ids = collection.get()["ids"]

    if old_ids:
        collection.delete(ids=old_ids)

    collection.add(
        ids=[f"doc_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )

    return len(chunks)


# Retrieve Relevant Information
def retrieve_context(question):
    embedding_model, _ = load_models()

    question_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(3, collection.count())
    )

    return "\n\n".join(results["documents"][0])


# Generate Answer
def ask_question(question):
    if collection.count() == 0:
        return "Please upload a PDF first."

    try:
        _, chatbot = load_models()
        context = retrieve_context(question)

        prompt = f"""
        Answer the question using only the context.
        Give a complete answer in simple English.
        Do not invent information.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        response = chatbot(
            prompt,
            max_new_tokens=150,
            do_sample=False,
            truncation=True
        )

        return response[0]["generated_text"].strip()

    except Exception as e:
        return f"Error: {str(e)}"