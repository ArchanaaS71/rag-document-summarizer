import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import pipeline


st.set_page_config(
    page_title="RAG PDF Summarizer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 RAG Document Summarizer")
st.write("Upload a PDF and get summary + answers")

@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"   # or your model
    )

summarizer = load_model()


uploaded_file = st.file_uploader(
    "Drag & Drop PDF here",
    type="pdf"
)

if uploaded_file:

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.success("✅ PDF uploaded")

    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    st.divider()

    if st.button("Generate Summary"):

        with st.spinner("Generating summary..."):

            relevant_docs = retriever.get_relevant_documents(
                "Summarize this document"
            )

            context = " ".join(
                [doc.page_content for doc in relevant_docs]
            )

            summary = summarizer(
                context,
                max_length=200,
                min_length=60,
                do_sample=False
            )

            st.subheader("📌 Summary")
            st.write(summary[0]["summary_text"])

    st.divider()

    st.subheader("💬 Ask questions from the document")

    user_question = st.text_input("Enter your question")

    if user_question:

        with st.spinner("Searching answer..."):

            relevant_docs = retriever.get_relevant_documents(
                user_question
            )

            context = " ".join(
                [doc.page_content for doc in relevant_docs]
            )

            answer = summarizer(
                context,
                max_length=150,
                min_length=40,
                do_sample=False
            )

            st.success(answer[0]["summary_text"])
