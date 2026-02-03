from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from rag_engine import RAGEngine

app = FastAPI(title="College Manual RAG API")

# Allow Streamlit (default port 8501) + localhost dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_engine = RAGEngine()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class QuestionRequest(BaseModel):
    question: str
    reset_history: bool = False


# class AnswerResponse(BaseModel):
#     answer: str
from typing import List

class AnswerResponse(BaseModel):
    answer: str
    chunks: List[str]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/upload-manual")
async def upload_manual(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        rag_engine.build_from_pdf(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    rag_engine.reset_history()
    return {"status": "indexed", "filename": file.filename}


# @app.post("/api/ask", response_model=AnswerResponse)
# async def ask_question(req: QuestionRequest):
#     answer = rag_engine.ask(req.question)
#     return AnswerResponse(answer=answer)
@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    result = rag_engine.ask(req.question)
    return AnswerResponse(answer=result["answer"], chunks=result["chunks"])

@app.post("/api/save-index")
async def save_index():
    try:
        idx, meta = rag_engine.save_index()
        return {"status": "saved", "index_file": idx, "metadata_file": meta}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/load-index")
async def load_index():
    try:
        rag_engine.load_index()
        return {"status": "index loaded"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
