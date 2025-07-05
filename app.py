import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# ENV
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# FORCE CPU so no CUDA errors
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# LLM
llm = ChatGroq(api_key=groq_api_key, model_name='llama3-8b-8192')


# Prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response to the question.
    Context:
    {context}

    Question: {input}
    """
)


def create_vectors_embeddings():
    if "vectors" not in st.session_state:
        try:
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            st.session_state.loader = PyPDFDirectoryLoader("researchPapers")
            st.session_state.docs = st.session_state.loader.load()
            if not st.session_state.docs:
                st.error("No PDFs found in the 'research_paper' folder. Please add documents.")
                return
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.docs
            )
            if not st.session_state.final_documents:
                st.error("Document splitting resulted in zero chunks.")
                return
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents, st.session_state.embeddings
            )
            st.success("Vector Data is ready.")
        except Exception as e:
            st.error(f"Error creating embeddings: {e}")

# Streamlit UI
st.title("Document RAG Question Answering")
user_prompt = st.text_input("Enter your query from the research papers")

if st.button("Generate Document Embeddings"):
    create_vectors_embeddings()
    st.success("Vector data is ready.")

if user_prompt:
    if "vectors" not in st.session_state:
        st.error("Please click 'Generate Document Embeddings' first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        elapsed = time.process_time() - start
        st.write(f"Response time: {elapsed:.2f} sec")

        st.write(response.get('answer', 'No answer found.'))

        with st.expander("Show Relevant Documents"):
            for i, doc in enumerate(response.get('source_documents', [])):
                st.write(doc.page_content)
                st.write('-------------------------')
