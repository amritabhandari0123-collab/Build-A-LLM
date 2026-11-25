import re
import torch
from pathlib import Path
from train_model import SmallTransformer, enc

checkpoint = torch.load(
    "model_checkpoint.pt",
    map_location="cpu"
)

vocab = checkpoint["vocab"]
block_size = checkpoint["block_size"]
token_to_id = {t: i for i, t in enumerate(vocab)}

model = SmallTransformer()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

text = Path("fineweb_corpus.txt").read_text(
    encoding="utf-8"
)

sentences = re.split(r"(?<=[.!?])\s+", text)

for sentence in sentences:
    words = sentence.split()

    if not 6 <= len(words) <= 20:
        continue

    beginning = " ".join(words[:4])
    tokens = enc.encode_ordinary(beginning)

    if all(t in token_to_id for t in tokens):
        break
else:
    raise ValueError("No suitable sentence found.")

prompt_ids = [token_to_id[t] for t in tokens]

torch.manual_seed(42)

with open("corpus_completions.txt", "w",
          encoding="utf-8") as f:

    f.write("Chosen sentence: " + sentence + "\n")
    f.write("Sentence beginning: " + beginning + "\n\n")

    with torch.no_grad():
        for i in range(100):
            ids = prompt_ids.copy()

            for _ in range(40):
                x = torch.tensor(
                    [ids[-block_size:]],
                    dtype=torch.long
                )

                logits = model(x)[0, -1]
                probs = torch.softmax(logits, dim=-1)

                next_id = torch.multinomial(
                    probs, 1
                ).item()

                ids.append(next_id)

            result = enc.decode([vocab[j] for j in ids])

            f.write(f"{i + 1}. {result}\n\n")
            print(i + 1, result)

print("Results saved to corpus_completions.txt")