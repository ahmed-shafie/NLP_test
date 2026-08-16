"""Train the topic head that :mod:`app.nlu.topic_head` serves at runtime.

Labels are the *answer* each corpus row deserves — a reviewed family reply, or a
topic's specific reply where it has one — plus an explicit "no answer" class
trained on every executable row (transfer, bill, balance, ...). Predicting the
answer rather than the raw topic matches what the customer reads and what the
error metric counts, and the "no answer" class is what stops a more talkative
gate from answering a transfer request.

Features are the embeddings the index already computes, so the head adds no
model and no second encode at runtime. The trained weights are written as plain
arrays (no pickle) tagged with the embedding model they belong to; a head trained
for a different embedder is refused at load time.

Requires scikit-learn, a development-only dependency — the runtime forward pass
is numpy. Run::

    python -m scripts.train_topic_classifier
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from app.config import settings
from app.conversation.topic_replies import answer_key
from app.embeddings import get_embedder
from app.nlu.corpus import load_corpus_examples
from app.nlu.topic_head import WEIGHTS_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    embedder = get_embedder()
    if embedder is None:
        raise SystemExit("embedder unavailable")
    corpus = load_corpus_examples()
    labels = np.array([answer_key(example.topic) for example in corpus])
    print(f"embedding {len(corpus)} corpus rows with {settings.embedding_model} ...")
    vectors = embedder.encode([example.text for example in corpus])

    train_x, dev_x, train_y, dev_y = train_test_split(
        vectors, labels, test_size=0.1, random_state=args.seed, stratify=labels
    )
    model = MLPClassifier(
        hidden_layer_sizes=(args.hidden,),
        max_iter=120,
        random_state=args.seed,
        early_stopping=True,
    )
    print(f"training on {len(train_x)} rows, {len(set(labels))} answers ...")
    model.fit(train_x, train_y)
    print(f"held-out accuracy over the corpus itself: {model.score(dev_x, dev_y):.3f}")

    w1, w2 = (weights.astype("float32") for weights in model.coefs_)
    b1, b2 = (bias.astype("float32") for bias in model.intercepts_)
    np.savez_compressed(
        args.out,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        keys=np.array([str(key) for key in model.classes_]),
        embedding_model=np.array(settings.embedding_model),
    )
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
