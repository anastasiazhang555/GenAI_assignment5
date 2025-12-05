import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden

    def generate(self, start_char: str, length: int, char_to_idx, idx_to_char, device="cpu"):
        self.eval()
        input_eval = torch.tensor([[char_to_idx[start_char]]], device=device)
        hidden = None
        generated_text = start_char

        for _ in range(length):
            logits, hidden = self.forward(input_eval, hidden)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = idx_to_char[next_idx]
            generated_text += next_char
            input_eval = torch.tensor([[next_idx]], device=device)
        return generated_text