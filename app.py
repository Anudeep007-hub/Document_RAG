import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(api_key=groq_api_key, model_name='gemma2-9b-it')

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {content}
    <context>
    Question: {input}
    """
)

def create_vectors_embeddings():
    if "vectors" not in st.session_state:
        try:
            st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
            st.session_state.loader = PyPDFDirectoryLoader("/research_paper")
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.docs[:50]
            )
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents, st.session_state.embeddings
            )
        except Exception as e:
            st.error(f"Error creating embeddings: {e}")

user_prompt = st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vectors_embeddings()
    st.write('Vector Data is ready.')

if user_prompt:
    if "vectors" not in st.session_state:
        st.error("Please click 'Document Embedding' first to create vectors.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        elapsed = time.process_time() - start
        st.write(f"Response time: {elapsed:.2f} sec")

        st.write(response.get('answer', 'No answer found.'))

        with st.expander("Document similarity search"):
            for i, doc in enumerate(response.get('source_documents', [])):
                st.write(doc.page_content)
                st.write('-------------------------')
