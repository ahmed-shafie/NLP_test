"""Fine-tune an Arabic BERT for beneficiary-name (PER) token classification, on CPU.

Plain torch loop on purpose: `Trainer` pulls in `accelerate`, and this PoC should
not add a dependency to the project venv before the numbers justify it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

HERE = Path(__file__).resolve().parent
BASE = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
OUT = HERE / "model-per"

LABELS = ["O", "B-PER", "I-PER"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
EPOCHS = 3
BATCH = 32


class SpanDataset(Dataset):
    """Character spans -> per-token BIO labels via the tokenizer's offsets."""

    def __init__(self, path: Path, tokenizer, max_length: int = 48) -> None:
        self.rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self.rows[index]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        labels: list[int] = []
        started = False
        for start, end in encoded["offset_mapping"]:
            if start == end:  # special token
                labels.append(-100)
                continue
            # Overlap, not containment: Arabic writes the preposition attached
            # ("لسارة"), so the wordpiece covering the name also covers the "ل".
            # Requiring containment would label that piece O and teach the model
            # to drop the name's first letter.
            inside = any(
                start < span_end and end > span_start
                for span_start, span_end in row["spans"]
            )
            if inside:
                labels.append(LABEL2ID["I-PER" if started else "B-PER"])
                started = True
            else:
                labels.append(LABEL2ID["O"])
                started = False
        encoded.pop("offset_mapping")
        encoded["labels"] = labels
        return dict(encoded)


@torch.no_grad()
def sequence_accuracy(model, loader: DataLoader) -> float:
    """Share of utterances whose whole tag sequence is exactly right."""

    model.eval()
    exact = 0
    total = 0
    for batch in loader:
        labels = batch.pop("labels")
        logits = model(**batch).logits
        predictions = logits.argmax(dim=-1)
        for pred_row, label_row in zip(predictions, labels, strict=True):
            mask = label_row != -100
            total += 1
            exact += int(torch.equal(pred_row[mask], label_row[mask]))
    model.train()
    return exact / max(total, 1)


def main() -> None:
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE,
        num_labels=len(LABELS),
        id2label={i: label for label, i in LABEL2ID.items()},
        label2id=LABEL2ID,
    )
    collator = DataCollatorForTokenClassification(tokenizer, return_tensors="pt")
    train_loader = DataLoader(
        SpanDataset(HERE / "train.jsonl", tokenizer),
        batch_size=BATCH,
        shuffle=True,
        collate_fn=collator,
    )
    holdout_loader = DataLoader(
        SpanDataset(HERE / "holdout.jsonl", tokenizer),
        batch_size=64,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    steps = EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-5, total_steps=steps, pct_start=0.1
    )

    model.train()
    step = 0
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        for batch in train_loader:
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 25 == 0:
                rate = step / (time.time() - started)
                print(
                    f"epoch {epoch} step {step}/{steps} loss {loss.item():.4f} "
                    f"({rate:.2f} steps/s)",
                    flush=True,
                )
        accuracy = sequence_accuracy(model, holdout_loader)
        print(
            f"== epoch {epoch}: holdout exact tag sequence = {accuracy:.4f}", flush=True
        )

    model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)
    print(f"saved -> {OUT} in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
