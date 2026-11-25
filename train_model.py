import time
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Load corpus and validate tokens
torch.manual_seed(42)
torch.set_num_threads(4)

enc = tiktoken.get_encoding("o200k_base")

text = Path("fineweb_corpus.txt").read_text(encoding="utf-8")
tokens = enc.encode_ordinary(text)

saved = Path("fineweb_tokens.txt").read_text()
saved_tokens = [int(x) for x in re.findall(r"\d+", saved)]

if saved_tokens != tokens:
    print("Token mismatch. Regenerating tokens.")
    Path("fineweb_tokens.txt").write_text(
        " ".join(map(str, tokens))
    )
else:
    print("Token validation successful.")

# 2. Create compact vocabulary
vocab = sorted(set(tokens))
token_to_id = {t: i for i, t in enumerate(vocab)}

data = torch.tensor(
    [token_to_id[t] for t in tokens],
    dtype=torch.long
)

vocab_size = len(vocab)
block_size = 32
batch_size = 8

print("Total tokens:", len(tokens))
print("Vocabulary size:", vocab_size)

# 3. Define a small causal transformer
class SmallTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 64)
        self.position = nn.Embedding(block_size, 64)

        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            layer, num_layers=2
        )

        self.output = nn.Linear(64, vocab_size)

    def forward(self, x):
        length = x.size(1)

        positions = torch.arange(length, device=x.device)
        h = self.embedding(x) + self.position(positions)

        mask = torch.triu(
            torch.ones(
                length, length,
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )

        h = self.transformer(h, mask=mask)

        return self.output(h)


# 4. Generate sample text
@torch.no_grad()
def generate(model, prompt_ids, count=40):
    model.eval()

    generator = torch.Generator().manual_seed(42)

    ids = list(prompt_ids)
    original_ids = [vocab[i] for i in ids]

    print(enc.decode(original_ids), end="", flush=True)

    for _ in range(count):
        x = torch.tensor(
            [ids[-block_size:]],
            dtype=torch.long
        )

        logits = model(x)[0, -1]
        probs = F.softmax(logits, dim=-1)

        next_id = torch.multinomial(
            probs, 1, generator=generator
        ).item()

        ids.append(next_id)

        print(
            enc.decode([vocab[next_id]]),
            end="",
            flush=True
        )

    print()

    return enc.decode([vocab[i] for i in ids])


# 5. Train the model
def main():
    model = SmallTransformer().to("cpu")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001
    )

    prompt_ids = data[:8].tolist()

    losses = []
    steps = []

    with open("generated_samples.txt", "w",
              encoding="utf-8") as samples:

        print("\nBefore training:")
        sample = generate(model, prompt_ids)
        samples.write("Before training:\n" + sample + "\n\n")

        start = time.time()
        step = 0

        with open("training_log.txt", "w") as log:

            header = "step | loss | elapsed time | tokens/second"
            print(header)
            log.write(header + "\n")

            while time.time() - start < 60:
                model.train()

                indexes = torch.randint(
                    0,
                    len(data) - block_size,
                    (batch_size,)
                )

                x = torch.stack([
                    data[i:i + block_size]
                    for i in indexes
                ])

                y = torch.stack([
                    data[i + 1:i + block_size + 1]
                    for i in indexes
                ])

                logits = model(x)

                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    y.reshape(-1)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step += 1
                elapsed = time.time() - start

                speed = (
                    step * batch_size * block_size
                ) / elapsed

                losses.append(loss.item())
                steps.append(step)

                line = (
                    f"{step} | {loss.item():.4f} | "
                    f"{elapsed:.2f}s | {speed:.2f}"
                )

                print(line, flush=True)
                log.write(line + "\n")
                log.flush()

                if step % 20 == 0:
                    print(f"\nSample at step {step}:")
                    sample = generate(model, prompt_ids)

                    samples.write(
                        f"Step {step}:\n{sample}\n\n"
                    )
                    samples.flush()

        print("\nAfter training:")
        sample = generate(model, prompt_ids)

        samples.write(
            "After training:\n" + sample + "\n"
        )

    # 6. Save training loss graph
    plt.plot(steps, losses)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig("training_loss.png")
    plt.close()

    # 7. Save model checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "block_size": block_size
    }, "model_checkpoint.pt")

    print("\nTraining completed.")
    print("All results saved.")


if __name__ == "__main__":
    main()