import streamlit as st
import tempfile
import os
from typing import TypedDict

# LangChain modules
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

# LangGraph
from langgraph.graph import StateGraph, END

from transformers import pipeline

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="LangGraph RAG Summariser", layout="wide")
st.title("📄 LangGraph RAG Document Summariser")

uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file:

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # Load document
    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    st.success(f"Document split into {len(chunks)} chunks.")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # Load lightweight LLM
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=512,
        temperature=0.3
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # ---------------------------
    # LangGraph State Definition
    # ---------------------------

    class GraphState(TypedDict):
        question: str
        context: str
        answer: str

    # ---------------------------
    # Nodes
    # ---------------------------

    def retrieve(state: GraphState):
        docs = retriever.get_relevant_documents(state["question"])
        combined = "\n\n".join([doc.page_content for doc in docs])
        return {"context": combined}

    def generate(state: GraphState):
        prompt = f"""
        You are an expert summariser.

        Context:
        {state['context']}

        Provide a concise summary in bullet points.
        """

        response = llm.invoke(prompt)
        return {"answer": response}

    # ---------------------------
    # Build LangGraph
    # ---------------------------

    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    app_graph = graph.compile()

    # ---------------------------
    # Run Graph
    # ---------------------------

    if st.button("Generate Summary"):
        with st.spinner("Running LangGraph workflow..."):

            result = app_graph.invoke({
                "question": "Summarise the document."
            })

        st.subheader("📌 Summary")
        st.write(result["answer"])

    os.remove(file_path)
