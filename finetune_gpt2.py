# finetune_gpt2.py
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    # 1. Select device (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 2. Load the Nectar dataset (only has a 'train' split)
    dataset = load_dataset("berkeley-nest/Nectar")
    train_data = dataset["train"]

    # Use a small subset for faster training (you can change this if you want)
    max_train_samples = 1000
    if len(train_data) > max_train_samples:
        train_data = train_data.select(range(max_train_samples))
    print(f"Training on {len(train_data)} examples.")

    # 3. Format each sample into a single text string
    def format_example(example):
        prompt = example["prompt"]
        answers = example["answers"]

        # Choose the best answer (smallest rank)
        chosen_answer = ""
        if isinstance(answers, list) and len(answers) > 0:
            best = sorted(
                answers,
                key=lambda x: x.get("rank", 999)
            )[0]
            chosen_answer = best.get("answer", "")
        text = f"Human: {prompt}\nAssistant: {chosen_answer}\n"
        return {"text": text}

    print("Formatting examples...")
    train_data = train_data.map(format_example)

    # 4. Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
        )

    print("Tokenizing examples...")
    tokenized_train = train_data.map(tokenize_function, batched=True)
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask"])

    train_loader = DataLoader(tokenized_train, batch_size=4, shuffle=True)
    num_batches = len(train_loader)
    print(f"Number of training batches per epoch: {num_batches}")

    # 5. Load pretrained GPT-2 base model
    print("Loading GPT-2 model...")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 1  # one epoch for the assignment

    # 6. Training loop (early stop after 50 batches for speed)
    max_batches_per_epoch = 50  # <- IMPORTANT: keep it small on CPU

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1} ...")
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["input_ids"],
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            # Print very frequently so you can see it is doing something
            print(f"Epoch 1 | Batch {step} | Loss {loss.item():.4f}")

            if step >= max_batches_per_epoch:
                print(f"Reached {max_batches_per_epoch} batches, stopping early.")
                break

        avg_loss = total_loss / min(num_batches, max_batches_per_epoch)
        print(f"Epoch {epoch + 1} finished. Avg training loss: {avg_loss:.4f}")

    # 7. Save fine-tuned model
    save_dir = "models/gpt2_finetuned"
    print(f"Saving model to {save_dir} ...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Done. Fine-tuned model saved.")


if __name__ == "__main__":
    main()