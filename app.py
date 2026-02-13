import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitters.recursive import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms.huggingface import HuggingFaceLLM

from transformers import pipeline
import tempfile
import os

st.set_page_config(page_title="RAG Summariser", layout="wide")
st.title("📄 RAG Document Summariser")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # Load document
    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    docs = loader.load()

    # Split document
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    st.info(f"Split into {len(chunks)} chunks.")

    # Embeddings
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Store in FAISS
    vectordb = FAISS.from_documents(chunks, embedder)

    # Load lightweight model
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=512,
        temperature=0.3
    )
    llm = HuggingFaceLLM(pipeline=pipe)

    # Create RAG chain
    rag = RetrievalQA.from_llm(llm=llm, retriever=vectordb.as_retriever())

    if st.button("Generate Summary"):
        with st.spinner("Summarising..."):
            result = rag.run("Summarise the document concisely.")
            st.markdown("### 📌 Summary")
            st.write(result)

    # remove tmp file
    os.remove(file_path)
