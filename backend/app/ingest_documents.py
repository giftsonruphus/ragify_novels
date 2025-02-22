import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
import os

# Convert PDF to TXT with page range
def convert_pdf_to_text(pdf_path, start_page, end_page):
    txt_path = pdf_path.replace(".pdf", ".txt")
    doc = fitz.open(pdf_path)
    with open(txt_path, "w", encoding="utf-8") as txt_file:
        for page_num in range(start_page - 1, end_page):
            page = doc.load_page(page_num)
            txt_file.write(page.get_text())
            txt_file.write("\n\n")
    return txt_path

# Ingest text into FAISS
def ingest_to_faiss(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Simple chunking
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode([doc.page_content for doc in documents])

    # Store in FAISS
    vector_db = FAISS.from_embeddings(embeddings, documents)
    vector_db.save_local("./faiss_index")
