import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import BertTokenizer
import numpy as np
import nltk
import os
import faiss

# Download NLTK's sentence tokenizer
nltk.download('punkt_tab')

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
def ingest_to_faiss(txt_path, faiss_index_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split text into sentences
    sentences = nltk.sent_tokenize(text)
    documents = [Document(page_content=sentence) for sentence in sentences]

    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Initialize embedding model
    # embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Tokenize sentences and generate embeddings
    # embeddings = []
    # for sentence in sentences:
    #     inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    #     embedding = embedding_model.encode(sentence)
    #     embeddings.append(embedding)

    # embeddings = np.array(embeddings)

    # Check if FAISS index exists
    if os.path.exists(faiss_index_path):
        vector_db = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
        vector_db.add_documents(documents)
    else:
        # Create a new FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        vector_db = FAISS(embedding_model.encode, index)
        vector_db.add_documents(documents)

    # Save the FAISS index
    vector_db.save_local(faiss_index_path)
