import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

st.set_page_config(page_title="RAG Document Summarizer", layout="wide")
st.title("RAG-Based Document Summarizer")
st.write("Upload a PDF and generate a summary using open-source models.")

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
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_documents(docs, embeddings)

    vectorstore = get_vectorstore(docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline=summarizer)

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
