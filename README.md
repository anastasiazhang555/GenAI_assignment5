# 📝 README — Text Generation API (RNN + Fine-Tuned GPT-2)

This project extends the text-generation API from Modules 3 and 7 by adding a fine-tuned GPT-2 model (Module 9 / Assignment 5).
It now supports:
	•	Character-level RNN text generation
	•	Fine-tuned GPT-2 LLM generation
	•	Optional spaCy word embeddings

All code is written in FastAPI, and Python dependencies are managed using uv.

## 📂 Project Structure
```
sps_genai/
│
├── app/
│   ├── main.py                # FastAPI app (RNN + LLM endpoints)
│   ├── rnn_model.py           # Character-level LSTM
│   ├── llm_model.py           # GPT-2 loading + generation
│
├── finetune_gpt2.py           # Training script
├── pyproject.toml             # Project dependencies (uv)
├── .gitignore
├── README.md
│
└── models/
    └── .gitkeep               # Fine-tuned GPT-2 model will be created here
```
---

## 🚀 1. Installation

Clone the repository:
```
git clone https://github.com/anastasiazhang555/genAI_assigenment5
cd sps_genai
```
Install dependencies:
```
uv sync
```

## ⚙️ 2. (Optional) Fine-tune GPT-2
## ▶️ 3. Run the API
```
uv run uvicorn app.main:app --reload
```
👉 http://127.0.0.1:8000/docs

## 📌 4. Endpoints
Generate text using the fine-tuned GPT-2 model.
```
{
  "start_word": "Human: What is AI?\nAssistant:",   #Example
  "length": 80
}
```
