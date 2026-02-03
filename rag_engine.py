import os
import pickle
import requests
import numpy as np
from typing import List, Dict, Optional
from pypdf import PdfReader
import faiss
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not set in .env file")

BASE_URL = "https://generativelanguage.googleapis.com/v1"
EMBED_MODEL_ID = "text-embedding-004"
LLM_MODEL_ID = "gemini-2.0-flash"

INDEX_FILE = "saved_index.faiss"
META_FILE = "saved_metadata.pkl"


# ---------------- Gemini API wrappers ---------------- #

def gemini_embed(text: str) -> np.ndarray:
    url = f"{BASE_URL}/models/{EMBED_MODEL_ID}:embedContent?key={API_KEY}"
    payload = {
        "model": f"models/{EMBED_MODEL_ID}",
        "content": {"parts": [{"text": text}]}
    }
    r = requests.post(url, json=payload)
    r.raise_for_status()
    data = r.json()
    return np.array(data["embedding"]["values"], dtype="float32")


def gemini_generate(prompt: str) -> str:
    url = f"{BASE_URL}/models/{LLM_MODEL_ID}:generateContent?key={API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            # "top_k": ,
            "top_p": 1.0
        }
    }
    r = requests.post(url, json=payload)
    r.raise_for_status()
    data = r.json()

    parts = []
    for c in data.get("candidates", []):
        for p in c.get("content", {}).get("parts", []):
            if "text" in p:
                parts.append(p["text"])

    return "\n".join(parts).strip()


# ---------------- PDF + Chunking ---------------- #

def load_pdf_texts(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    return [p.extract_text() for p in reader.pages if p.extract_text()]


def split_into_chunks(pages: List[str], chunk_size=800, overlap=200) -> List[str]:
    chunks = []
    for doc in pages:
        i = 0
        while i < len(doc):
            chunks.append(doc[i:i + chunk_size])
            i += max(1, chunk_size - overlap)
    return chunks


# ---------------- Vector + BM25 Store ---------------- #

class FaissStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.texts: List[str] = []
        self.embs: List[np.ndarray] = []
        self.bm25 = None

    def add_texts(self, new_texts: List[str]):
        for text in new_texts:
            if text.strip():
                emb = gemini_embed(text)
                self.texts.append(text)
                self.embs.append(emb)

        if self.embs:
            mat = np.vstack(self.embs)
            faiss.normalize_L2(mat)
            self.index.add(mat)

            tokenized = [t.lower().split() for t in self.texts]
            self.bm25 = BM25Okapi(tokenized)

    def embedding_search(self, query: str, k=5):
        if not self.texts:
            return []
        emb = gemini_embed(query)
        emb = emb / np.linalg.norm(emb)
        emb = emb.reshape(1, -1)
        _, idxs = self.index.search(emb, k)
        return [self.texts[i] for i in idxs[0] if i != -1]

    def hybrid_search(self, query: str, k=7):
        emb_hits = self.embedding_search(query, k)

        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(zip(self.texts, bm25_scores), key=lambda x: x[1], reverse=True)
        bm25_hits = [t for t, _ in ranked[:k]]

        combined = []
        for r in emb_hits + bm25_hits:
            if r not in combined:
                combined.append(r)
            if len(combined) >= k:
                break

        return combined


# ---------------- Query Expansion ---------------- #

def generate_search_queries(question: str, n=5) -> List[str]:
    prompt = f"""
Rewrite the query into {n} search variations including:
- title-only keywords
- short keyword queries
- abbreviation and role-based versions

Return one per line. No numbering.

Query: {question}
"""
    text = gemini_generate(prompt)
    lines = [l.strip().lower() for l in text.split("\n") if l.strip()]
    return list(dict.fromkeys(lines))[:n] or [question]


# ---------------- RAG Answering ---------------- #

def answer_with_rag(question: str, store: FaissStore, history=None) -> Dict[str, any]:
    if store is None or not store.texts:
        return {"answer": "No manual loaded.", "chunks": []}

    history = history or []

    queries = generate_search_queries(question)
    retrieved = []
    for q in queries:
        retrieved.extend(store.hybrid_search(q, k=5))

    chunks = []
    seen = set()
    for c in retrieved:
        if c not in seen:
            seen.add(c)
            chunks.append(c)
        if len(chunks) >= 5:
            break

    context = "\n\n---\n\n".join(chunks)

    prompt = f"""
Answer using ONLY the text below. If not found, reply:
"I do not find that information in the manual."

TEXT:
{context}

QUESTION: {question}

ANSWER:
"""

    answer = gemini_generate(prompt)

    return {"answer": answer, "chunks": chunks}


# ---------------- High-Level Manager ---------------- #

class RAGEngine:
    def __init__(self):
        self.store = None
        self.chat_history = []

    def build_from_pdf(self, pdf_path: str):
        pages = load_pdf_texts(pdf_path)
        chunks = split_into_chunks(pages)
        emb_dim = len(gemini_embed(chunks[0]))

        store = FaissStore(emb_dim)
        store.add_texts(chunks)
        self.store = store

    def save_index(self):
        if not self.store:
            raise ValueError("No index to save.")
        faiss.write_index(self.store.index, INDEX_FILE)
        with open(META_FILE, "wb") as f:
            pickle.dump(self.store.texts, f)
        return True

    def load_index(self):
        if not (os.path.exists(INDEX_FILE) and os.path.exists(META_FILE)):
            raise FileNotFoundError("No saved index found.")
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "rb") as f:
            texts = pickle.load(f)

        store = FaissStore(index.d)
        store.index = index
        store.texts = texts
        tokenized = [t.split() for t in texts]
        store.bm25 = BM25Okapi(tokenized)

        self.store = store
        return True

    def ask(self, question: str):
        result = answer_with_rag(question, self.store, self.chat_history)
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": result["answer"]})
        return result

    def reset_history(self):
        self.chat_history = []
