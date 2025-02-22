from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from ingest_documents import convert_pdf_to_text, ingest_to_faiss
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from huggingface_hub import snapshot_download
import os

app = FastAPI()

# Load FAISS index
FAISS_INDEX_PATH = "./faiss_index"
if not os.path.exists(FAISS_INDEX_PATH):
    raise RuntimeError("FAISS index not found. Please upload and process a PDF first.")
vector_db = FAISS.load_local(FAISS_INDEX_PATH)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Set custom cache directory for Hugging Face models
MODEL_CACHE_DIR = "/home/decoder/ai/models"
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_DIR

# Download and load DeepSeek R1
model_path = snapshot_download(repo_id="deepseek-ai/deepseek-llm-7b-instruct", cache_dir=MODEL_CACHE_DIR)
llm = HuggingFacePipeline.from_pretrained(model_path)

@app.post("/upload-pdf/")
async def upload_pdf(
    file: UploadFile = File(...),
    start_page: int = Form(...),
    end_page: int = Form(...)
):
    contents = await file.read()
    pdf_path = f"/tmp/{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(contents)

    txt_path = convert_pdf_to_text(pdf_path, start_page, end_page)
    ingest_to_faiss(txt_path)

    return {"message": f"Successfully processed pages {start_page}-{end_page} of {file.filename}"}

class QueryRequest(BaseModel):
    query: str

@app.post("/query/")
async def query_rag(request: QueryRequest):
    query = request.query

    # Embed the query
    query_embedding = embedding_model.encode(query)

    # Retrieve top-k documents
    docs = vector_db.similarity_search_by_vector(query_embedding, k=5)
    context = "\n".join([doc.page_content for doc in docs])

    # Create prompt
    prompt_template = PromptTemplate(
        template="Answer the question based on the context:\nContext:\n{context}\n\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"]
    )
    prompt = prompt_template.format(context=context, question=query)

    # Generate answer using DeepSeek R1
    answer = llm(prompt)

    return {"answer": answer}
