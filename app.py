import streamlit as st
import tempfile
import os
from typing import TypedDict

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langgraph.graph import StateGraph, END

# --------------------
# Streamlit UI
# --------------------

st.set_page_config(page_title="LangGraph RAG", layout="wide")
st.title("📄 LangGraph RAG Summariser")

openai_key = st.text_input("Enter OpenAI API Key", type="password")

uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file and openai_key:

    os.environ["OPENAI_API_KEY"] = openai_key

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # Load file
    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # Embeddings (API-based, lightweight)
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # --------------------
    # LangGraph State
    # --------------------

    class GraphState(TypedDict):
        question: str
        context: str
        answer: str

    def retrieve(state: GraphState):
        docs = retriever.get_relevant_documents(state["question"])
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"context": context}

    def generate(state: GraphState):
        prompt = f"""
        Summarise the following document context clearly and concisely:

        {state['context']}
        """
        response = llm.invoke(prompt)
        return {"answer": response.content}

    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    app_graph = graph.compile()

    if st.button("Generate Summary"):
        result = app_graph.invoke({
            "question": "Summarise the document."
        })

        st.subheader("📌 Summary")
        st.write(result["answer"])

    os.remove(file_path)
