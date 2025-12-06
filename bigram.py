import torch 
from torch import nn
from torch.nn import functional as F
import numpy as np

#hyperparameters
batch_size = 4
block_size = 8
max_iters = 10000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'


with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

print("length of dataset in characters: ", len(text))


chars = sorted(list(set(text)))
chars


stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode (s):
    return [stoi[c] for c in s]
def decode (l):
    return ''.join([itos[i] for i in l])
print(encode("hello world"))
print(decode(encode("hello world")))


data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)


n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

x=train_data[:block_size]
y=train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = m(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

            return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]  # (C,)
            probs = F.softmax(logits, dim=-1)  # (C,)
            next_idx = torch.multinomial(probs, num_samples=1)  # (1,)
            idx = torch.cat((idx, next_idx), dim=1)  # (T+1,)
        return idx

m = BigramLanguageModel(vocab_size=len(chars))
m = m.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

print(decode(m.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

for steps in range(max_iters):
    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

print(decode(m.generate(torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=500)[0].tolist()))


