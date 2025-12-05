from typing import List

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.rnn_model import LSTMModel
from app.llm_model import generate_with_llm

# ---------- FastAPI init ----------
app = FastAPI(title="Text Generation + Embeddings API (RNN / LLM Version)")


# ---------- Corpus for RNN ----------
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

# Build character-level vocabulary
text = " ".join(corpus).lower()
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# ---------- Initialize LSTM Model ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
rnn_model = LSTMModel(vocab_size).to(device)
rnn_model.eval()
# If you have saved weights, you can load them here:
# rnn_model.load_state_dict(torch.load("lstm_weights.pth", map_location=device))


# ---------- API Schemas ----------
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


@app.get("/")
def read_root():
    return {"message": "Welcome to the Text Generation + Embedding API (RNN + GPT-2 LLM)"}


# ---------- Text Generation with RNN ----------
@app.post("/generate_with_rnn")
def generate_with_rnn(request: TextGenerationRequest):
    """
    Generate text using the character-level LSTM model (Module 7).
    """
    if not request.start_word:
        raise HTTPException(status_code=400, detail="start_word cannot be empty.")

    start_char = request.start_word[0].lower()
    if start_char not in char_to_idx:
        raise HTTPException(status_code=400, detail=f"Character '{start_char}' not in vocabulary.")

    generated = rnn_model.generate(
        start_char=start_char,
        length=request.length,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        device=device,
    )
    return {"generated_text": generated}


# ---------- Text Generation with fine-tuned GPT-2 LLM ----------
@app.post("/generate_with_llm")
def generate_with_llm_endpoint(request: TextGenerationRequest):
    """
    Generate text using the fine-tuned GPT-2 model (Module 9 / Assignment 5).
    """
    if not request.start_word:
        raise HTTPException(status_code=400, detail="start_word cannot be empty.")

    generated_text = generate_with_llm(request.start_word, request.length)
    return {"generated_text": generated_text}


# ---------- Optional: spaCy Word Embeddings ----------
# We try to load spaCy, but we do NOT crash the whole app if the model is missing.
try:
    import spacy  # type: ignore

    nlp = spacy.load("en_core_web_md")
    SPACY_AVAILABLE = True
except Exception:
    nlp = None
    SPACY_AVAILABLE = False


class EmbeddingRequest(BaseModel):
    word: str = Field(..., min_length=1, description="The word to embed")
    use_doc_vector: bool = Field(
        default=False,
        description="If true, use doc vector (nlp(word)).vector instead of lexical vector.",
    )


class EmbeddingResponse(BaseModel):
    word: str
    dim: int
    has_vector: bool
    vector: List[float]


def _get_word_vector(token_text: str, use_doc_vector: bool = False) -> tuple[np.ndarray, bool]:
    """
    Helper to obtain a vector for a given token using spaCy, if available.
    Returns (vector, has_vector).
    """
    if not SPACY_AVAILABLE or nlp is None:
        # Return a zero vector with flag False
        return np.zeros(1, dtype="float32"), False

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


@app.get("/embed", response_model=EmbeddingResponse)
def embed_get(
    word: str = Query(..., min_length=1),
    use_doc_vector: bool = False,
):
    """
    Get a single word embedding using spaCy, if the 'en_core_web_md' model is installed.
    """
    if not SPACY_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="spaCy model 'en_core_web_md' is not installed. "
            "Run: uv run python -m spacy download en_core_web_md",
        )

    vec, has_vec = _get_word_vector(word, use_doc_vector)
    if not has_vec:
        raise HTTPException(status_code=404, detail=f"No usable vector for '{word}' in current model.")

    return EmbeddingResponse(
        word=word,
        dim=int(vec.shape[0]),
        has_vector=has_vec,
        vector=vec.tolist(),
    )


@app.post("/embed", response_model=EmbeddingResponse)
def embed_post(req: EmbeddingRequest):
    """
    Same as GET /embed but accepts a JSON body instead of query parameters.
    """
    if not SPACY_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="spaCy model 'en_core_web_md' is not installed. "
            "Run: uv run python -m spacy download en_core_web_md",
        )

    vec, has_vec = _get_word_vector(req.word, req.use_doc_vector)
    if not has_vec:
        raise HTTPException(status_code=404, detail=f"No usable vector for '{req.word}' in current model.")

    return EmbeddingResponse(
        word=req.word,
        dim=int(vec.shape[0]),
        has_vector=has_vec,
        vector=vec.tolist(),
    )