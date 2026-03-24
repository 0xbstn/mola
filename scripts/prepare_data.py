"""Download and prepare training data for Rust and SQL adapters.

Downloads from HuggingFace, converts to mlx-lm chat format (JSONL with messages),
and creates train/valid splits.

Output:
    data/rust/train.jsonl   (1000 examples)
    data/rust/valid.jsonl   (200 examples)
    data/sql/train.jsonl    (2000 examples)
    data/sql/valid.jsonl    (400 examples)
"""

import json
import random
from pathlib import Path

from datasets import load_dataset


def to_chat_jsonl(messages: list[dict], path: Path):
    """Write a list of chat examples to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(messages)} examples to {path}")


def prepare_rust():
    """Neloy262/rust_instruction_dataset → chat format, 1000 train + 200 valid."""
    print("Downloading Rust dataset...")
    ds = load_dataset("Neloy262/rust_instruction_dataset", split="train")

    examples = []
    for row in ds:
        instruction = row.get("instruction", "").strip()
        output = row.get("output", "").strip()
        if not instruction or not output:
            continue
        examples.append({
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output},
            ]
        })

    random.seed(42)
    random.shuffle(examples)
    examples = examples[:1200]

    to_chat_jsonl(examples[:1000], Path("data/rust/train.jsonl"))
    to_chat_jsonl(examples[1000:1200], Path("data/rust/valid.jsonl"))


def prepare_sql():
    """b-mc2/sql-create-context → chat format, 2000 train + 400 valid."""
    print("Downloading SQL dataset...")
    ds = load_dataset("b-mc2/sql-create-context", split="train")

    examples = []
    for row in ds:
        question = row.get("question", "").strip()
        context = row.get("context", "").strip()
        answer = row.get("answer", "").strip()
        if not question or not answer:
            continue

        # Combine schema + question into user message
        if context:
            user_msg = f"Given this schema:\n{context}\n\nWrite a SQL query: {question}"
        else:
            user_msg = f"Write a SQL query: {question}"

        examples.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": answer},
            ]
        })

    random.seed(42)
    random.shuffle(examples)
    examples = examples[:2400]

    to_chat_jsonl(examples[:2000], Path("data/sql/train.jsonl"))
    to_chat_jsonl(examples[2000:2400], Path("data/sql/valid.jsonl"))


if __name__ == "__main__":
    prepare_rust()
    print()
    prepare_sql()
    print("\nDone! Ready for mlx-lm lora training.")
