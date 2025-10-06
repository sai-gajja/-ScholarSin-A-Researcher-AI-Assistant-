from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def text_from_pdf(pdf_path, chunk_size_value=1000,chunk_overlap_value=50):
    # Load PDF and extract raw text
    loader = PyPDFLoader(pdf_path)  
    pages = loader.load()

    # Combine all pages into a single text
    full_text = '\n\n'.join([page.page_content for page in pages])

    # Split text into chunks (optional, for processing large documents)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size= chunk_size_value,  # Adjust based on needs
        chunk_overlap=chunk_overlap_value  # Helps maintain context
    )

    chunks = splitter.split_text(full_text)

    for chunk in chunks:
        print(chunk)
        print('\n')
    # return chunks

texts = text_from_pdf(r"C:\Users\G.SAI\Desktop\Principles of neural ensemble .pdf")


