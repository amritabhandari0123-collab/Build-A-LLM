import torch
from train_model import SmallTransformer, enc

checkpoint = torch.load(
    "model_checkpoint.pt", map_location="cpu"
)

vocab = checkpoint["vocab"]
block_size = checkpoint["block_size"]
token_to_id = {t: i for i, t in enumerate(vocab)}

model = SmallTransformer()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

prompt = "The sky is"
tokens = enc.encode_ordinary(prompt)

if any(t not in token_to_id for t in tokens):
    print("Some prompt tokens are missing from the vocabulary.")
    raise SystemExit

with torch.no_grad():
    with open("sky_completions.txt", "w", encoding="utf-8") as f:
        for i in range(100):
            ids = [token_to_id[t] for t in tokens]

            for _ in range(20):
                x = torch.tensor([ids[-block_size:]])
                logits = model(x)[0, -1]
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                ids.append(next_id)

            result = enc.decode([vocab[j] for j in ids])
            f.write(f"{i + 1}. {result}\n")
            print(i + 1, result)

print("Results saved to sky_completions.txt")