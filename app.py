import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

from transformers import pipeline
import tempfile
import os

st.set_page_config(page_title="RAG Summariser", layout="wide")
st.title("📄 RAG Document Summariser")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Load document
    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    texts = text_splitter.split_documents(documents)

    st.success(f"Document split into {len(texts)} chunks.")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store in FAISS
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Load simple LLM (lightweight)
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=512,
        temperature=0.3
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )

    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            summary = qa_chain.run(
                "Provide a concise summary of the document."
            )
        st.subheader("📌 Summary")
        st.write(summary)

    # Cleanup
    os.remove(file_path)
