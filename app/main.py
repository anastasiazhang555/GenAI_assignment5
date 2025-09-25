from typing import List
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from bigram_model import BigramModel

import spacy
import numpy as np

import spacy
import numpy as np

app = FastAPI(title="Text Gen + Embeddings API")


# ---------- Bigram Model ----------
corpus = [
	"The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
	"this is another example sentence",
	"we are generating text based on bigram probabilities",
	"bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)

class TextGenerationRequest(BaseModel):
	start_word: str
	length: int

@app.get("/")
def read_root():
	return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
	generated_text = bigram_model.generate_text(request.start_word, request.length)
	return {"generated_text": generated_text}

# ---------- spaCy Word Embeddings ----------
# Load the medium English model (make sure it's installed: en_core_web_md)
try:
    nlp = spacy.load("en_core_web_md")
except OSError as e:
    raise RuntimeError(
        "spaCy model 'en_core_web_md' not found. "
        "Run: uv run python -m spacy download en_core_web_md"
    ) from e

class EmbeddingRequest(BaseModel):
    word: str = Field(..., min_length=1, description="The word to embed")
    use_doc_vector: bool = Field(default=False, description="If true, use doc vector (nlp(word)).vector")

class EmbeddingResponse(BaseModel):
    word: str
    dim: int
    has_vector: bool
    vector: List[float]

def _get_word_vector(token_text: str, use_doc_vector: bool = False) -> tuple[np.ndarray, bool]:
    """
    Returns a vector and has_vector flag.
    - use_doc_vector=False: directly use vocab word vector
    - use_doc_vector=True: use doc.vector (works for short phrases)
    """
    if use_doc_vector:
        doc = nlp(token_text)
        vec = doc.vector
        has_vec = bool(np.linalg.norm(vec) > 0)
        return vec, has_vec
    else:
        lex = nlp.vocab[token_text]
        vec = lex.vector
        has_vec = bool(lex.has_vector and np.linalg.norm(vec) > 0)
        return vec, has_vec

@app.get("/embed")
def embed_get(word: str = Query(..., min_length=1), use_doc_vector: bool = False):
    vec, has_vec = _get_word_vector(word, use_doc_vector)
    if not has_vec:
        raise HTTPException(status_code=404, detail=f"No usable vector for '{word}' in current model.")
    return {
        "word": word,
        "dim": int(vec.shape[0]),
        "has_vector": has_vec,
        "vector": vec.tolist()
    }

@app.post("/embed", response_model=EmbeddingResponse)
def embed_post(req: EmbeddingRequest):
    vec, has_vec = _get_word_vector(req.word, req.use_doc_vector)
    if not has_vec:
        raise HTTPException(status_code=404, detail=f"No usable vector for '{req.word}' in current model.")
    return EmbeddingResponse(
        word=req.word,
        dim=int(vec.shape[0]),
        has_vector=has_vec,
        vector=vec.tolist()
    )