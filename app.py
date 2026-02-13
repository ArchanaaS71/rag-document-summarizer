import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import base64
import os

# --- Model & Embedding Setup ---
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_MODEL = "LaMini-Flan-T5"

tokenizer = T5Tokenizer.from_pretrained(TEXT_MODEL)
text_model = T5ForConditionalGeneration.from_pretrained(TEXT_MODEL, device_map="auto", torch_dtype="auto")
summarizer = pipeline("text2text-generation", model=text_model, tokenizer=tokenizer, max_length=500, min_length=50)

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# --- Preprocessing & Indexing ---
def build_or_load_index(file_path: str, index_path="faiss_index"):
    ext = os.path.splitext(file_path)[-1].lower()

    # Select appropriate loader
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Please upload PDF, TXT, or Markdown files.")

    # Load and split
    pages = loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    # Use separate index paths for each file type to avoid conflicts
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    vector_index = FAISS.from_documents(docs, embeddings)
    vector_index.save_local(index_path)
    return vector_index


# --- RAG Summarization ---
def rag_summarize(pdf_path: str, num_chunks=5):
    index = build_or_load_index(pdf_path)
    retriever = index.as_retriever(search_kwargs={"k": num_chunks})

    # Wrap the summarizer into LangChain's LLM format
    hf_pipe = pipeline("text2text-generation", model=text_model, tokenizer=tokenizer, max_length=512)
    llm = HuggingFacePipeline(pipeline=hf_pipe)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    query = "Summarize the main points of this document."
    output = qa_chain({"query": query})

    summary = output["result"]
    sources = output["source_documents"]
    return summary, sources

# --- Streamlit UI ---
@st.cache_data
def load_pdf_as_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def display_pdf(b64):
    st.markdown(f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="600"></iframe>', unsafe_allow_html=True)

st.set_page_config(layout="wide", page_title="Document Summarizer")

def main():
    st.title("Document Summarization using Retrieval-Augmented Generation (RAG)")
    uploaded = st.file_uploader("Upload PDF, TXT, or Markdown", type=["pdf", "txt", "md"])

    if uploaded:
        ext = os.path.splitext(uploaded.name)[-1].lower()
        temp_path = uploaded.name
        with open(temp_path, "wb") as f:
            f.write(uploaded.read())

        if st.button("Summarize"):
            col1, col2 = st.columns(2)

            # Show Document
            if ext == ".pdf":
                b64 = load_pdf_as_base64(temp_path)
                with col1:
                    st.subheader("Document")
                    display_pdf(b64)
            else:
                with col1:
                    st.subheader("Document Content")
                    with open(temp_path, "r", encoding="utf-8") as f:
                        st.text(f.read()[:2000])

            # RAG Summary
            with col2:
                st.subheader("Summary & Source Context")
                summary, src_docs = rag_summarize(temp_path, num_chunks=5)
                st.success(summary)

                st.markdown("##### Source Chunks Used:")
                for i, doc in enumerate(src_docs):
                    st.write(f"**Chunk {i+1}:** {doc.page_content[:200].strip()}…")

if __name__ == "__main__":
    main()
