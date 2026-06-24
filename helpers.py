"""
Helper functions for loading documents from the data directory.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader


def load_documents(data_dir="data"):
    """
    Load all PDF and text files from the specified directory.
    
    Args:
        data_dir (str): Path to the directory containing documents
        
    Returns:
        list: List of loaded document objects
    """
    documents = []
    
    # Create directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created '{data_dir}' folder. Please add your PDF files there.")
        return []
    
    # Process each file in the directory
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        
        # Handle PDF files
        if file_name.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded PDF: {file_name}")
            
        # Handle text files
        elif file_name.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded text file: {file_name}")
    
    return documents
