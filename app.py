import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="RAG Document Summarizer", layout="wide")
st.title("📄 RAG-Based Document Summarizer")
st.write("Upload a PDF and generate a smart summary using Retrieval-Augmented Generation (RAG).")

# ---------------------------
# OPENAI API KEY
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("🔑 OpenAI API Key not found. Please set it in Streamlit secrets.")
    st.stop()

# ---------------------------
# FILE UPLOADER
# ---------------------------
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("File uploaded successfully!")

    # ---------------------------
    # LOAD PDF
    # ---------------------------
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # ---------------------------
    # SPLIT DOCUMENT INTO CHUNKS
    # ---------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)
    st.write(f"Document split into {len(docs)} chunks.")

    # ---------------------------
    # CREATE EMBEDDINGS
    # ---------------------------
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # ---------------------------
    # LLM & Prompt
    # ---------------------------
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)

    prompt = ChatPromptTemplate.from_template("""
    You are an expert document summarizer.

    Use ONLY the context below to generate a concise summary.

    Context:
    {context}

    Question:
    {question}
    """)

    # ---------------------------
    # RAG Pipeline
    # ---------------------------
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # ---------------------------
    # GENERATE SUMMARY
    # ---------------------------
    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            result = rag_chain.invoke(
                "Provide a structured summary including main topic, key arguments, findings, and conclusions."
            )
            st.subheader("📌 Summary")
            st.write(result)

    os.remove(tmp_path)
