"""
Main chatbot application using Streamlit.
Users can ask questions and get answers from their documents.
"""

import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the page
st.set_page_config(page_title="Document Chatbot", page_icon="🤖")
st.title("🤖 Document Chatbot")
st.markdown("Ask questions about your documents and get answers.")

# Get API key
api_key = os.getenv("HUGGINGFACE_API_KEY")
if not api_key:
    st.error("API key not found. Please add HUGGINGFACE_API_KEY to your .env file.")
    st.stop()

# Load vector store with caching
@st.cache_resource
def load_vectorstore():
    """Load the saved vector store from disk."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

# Load QA chain with caching
@st.cache_resource
def load_qa_chain():
    """Create and return the question-answering chain."""
    vectorstore = load_vectorstore()
    
    # Initialize the language model
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        huggingfacehub_api_token=api_key,
        model_kwargs={"temperature": 0.1, "max_length": 512}
    )
    
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )
    return qa_chain

# Load the QA chain
try:
    qa_chain = load_qa_chain()
except Exception as e:
    st.error(f"Error loading the system: {e}")
    st.info("Please run 'python ingest.py' first to build the search index.")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your document assistant. How can I help you?"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = qa_chain.invoke(user_input)
                answer = response["result"]
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"An error occurred: {e}")
