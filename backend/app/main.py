from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from ingest_documents import convert_pdf_to_text, ingest_to_faiss
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_community.docstore.in_memory import InMemoryDocstore
from huggingface_hub import snapshot_download
import os
import faiss

app = FastAPI()

# Define paths and model identifiers
FAISS_INDEX_PATH = "./faiss_index"
MODEL_CACHE_DIR = "/home/decoder/ai/models"
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
task = "text-generation"
pipeline_kwargs = {"max_new_tokens": 100}

# Set environment variable for Hugging Face cache
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_DIR

# Load or initialize FAISS index
if os.path.exists(FAISS_INDEX_PATH):
    vector_db = FAISS.load_local(FAISS_INDEX_PATH)
else:
    # Initialize an empty FAISS index
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_function = embedding_model.encode
    dimension = embedding_model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dimension)

    docstore = InMemoryDocstore({})
    index_to_docstore_id = {}

    vector_db = FAISS(embedding_function, index, docstore, index_to_docstore_id)

# Download and load the DeepSeek model
llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task=task,
    pipeline_kwargs=pipeline_kwargs
)

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
    ingest_to_faiss(txt_path, FAISS_INDEX_PATH)

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

    # Generate answer using the DeepSeek model
    answer = llm(prompt)

    return {"answer": answer}
