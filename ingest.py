
"""
This script processes documents and creates a search index.
Run this once before starting the chatbot.
"""

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from utils.helpers import load_documents
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Starting document processing...")

# Step 1: Load documents
print("Loading documents...")
documents = load_documents("data")

if not documents:
    print("No documents found. Please add PDF or TXT files to the 'data' folder.")
    exit()

print(f"Loaded {len(documents)} document(s)")

# Step 2: Split documents into chunks
print("Splitting documents into smaller pieces...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Step 3: Convert text to embeddings
print("Converting text to numerical vectors...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 4: Create vector store
print("Building search index...")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Step 5: Save to disk
print("Saving search index...")
os.makedirs("faiss_index", exist_ok=True)
vectorstore.save_local("faiss_index")
print("Processing complete. You can now run the chatbot with: streamlit run app.py")
