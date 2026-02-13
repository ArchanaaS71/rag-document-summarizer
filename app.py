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
import time
from openai.error import RateLimitError

st.set_page_config(page_title="RAG Document Summarizer", layout="wide")
st.title("RAG-Based Document Summarizer")
st.write("Upload a PDF and generate a summary using Retrieval-Augmented Generation.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OpenAI API Key not found. Please set it in Streamlit secrets.")
    st.stop()

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    @st.cache_resource
    def get_vectorstore(docs):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
        return FAISS.from_documents(docs, embeddings)

    try:
        vectorstore = get_vectorstore(docs)
    except RateLimitError:
        st.error("OpenAI API rate limit exceeded. Please try again in a few seconds.")
        st.stop()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)

    prompt = ChatPromptTemplate.from_template("""
You are an expert document summarizer.

Use ONLY the context below to generate a concise summary.

Context:
{context}

Question:
{question}
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            result = rag_chain.invoke(
                "Provide a structured summary including main topic, key arguments, findings, and conclusions."
            )
            st.subheader("Summary")
            st.write(result)

    os.remove(tmp_path)
