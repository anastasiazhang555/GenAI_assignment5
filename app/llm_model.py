import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Directory of the fine-tuned GPT-2 model
MODEL_DIR = "models/gpt2_finetuned"

# Load tokenizer and model once at import time
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device)


def generate_with_llm(start_word: str, length: int) -> str:
    """
    Generate text using the fine-tuned GPT-2 model.
    start_word: Initial prompt or seed text.
    length: Number of additional tokens to generate.
    """
    input_ids = tokenizer.encode(start_word, return_tensors="pt").to(device)

    output_ids = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + length,
        do_sample=True,     # Enable sampling for creative output
        top_k=50,           # Limit sampling to top-k tokens
        top_p=0.95,         # Nucleus sampling
        num_return_sequences=1,
    )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text