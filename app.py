import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.title("🏢 Zyro Dynamics HR Assistant")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
st.write("Key loaded:", bool(GROQ_API_KEY))
st.write("Starts with:", GROQ_API_KEY[:4])

@st.cache_resource
def build_rag():

    pdf_dir = "pdfs"

    documents = []

    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(
                os.path.join(pdf_dir, file)
            )
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k":4}
    )

    llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

    prompt = ChatPromptTemplate.from_template(
    """
You are an HR assistant for Zyro Dynamics.

Answer the question using the provided context.

If the answer is found in the context, provide a clear and complete answer.

Only if the context contains no relevant information for the question, respond exactly:

I can only answer questions based on Zyro Dynamics HR policy documents.

Context:
{context}

Question:
{question}

Answer:
"""
)

    def format_docs(docs):
        return "\n\n".join(
            d.page_content for d in docs
        )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

chain, retriever = build_rag()

question = st.chat_input(
    "Ask an HR question..."
)

if question:

    answer = chain.invoke(question)

    st.write(answer)

    docs = retriever.invoke(question)

    with st.expander("Sources"):
        for doc in docs:
            st.write(
                doc.metadata.get("source")
            )
