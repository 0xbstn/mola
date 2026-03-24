"""End-to-end test: load base model + 2 adapters, compare outputs.

Tests MOLA's core value proposition:
  - Base model loaded once
  - Two adapters loaded on top (rust, sql)
  - Same prompt, different adapter → different output
  - Switch is instant (no reload)
"""

import time

import mlx_lm

# -- Config --
MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
RUST_ADAPTER = "adapters/rust-lora"
SQL_ADAPTER = "adapters/sql-lora"
MAX_TOKENS = 200

RUST_PROMPTS = [
    "Write a Rust function that reverses a string",
    "Implement a basic HTTP server in Rust",
    "How do you handle errors in Rust?",
    "Write a Rust struct for a linked list",
    "Explain ownership and borrowing in Rust",
]

SQL_PROMPTS = [
    "Given this schema:\nCREATE TABLE employees (id INT, name VARCHAR, salary INT, dept VARCHAR)\n\nWrite a SQL query: Find the average salary by department",
    "Write a SQL query to find duplicate emails in a users table",
    "Given this schema:\nCREATE TABLE orders (id INT, customer_id INT, amount DECIMAL, created_at DATE)\n\nWrite a SQL query: Top 5 customers by total spend",
    "Write a SQL query to find employees who earn more than their manager",
    "How do you optimize a slow JOIN query?",
]


def generate(model, tokenizer, prompt, adapter_path=None, max_tokens=MAX_TOKENS):
    """Generate with optional adapter fused."""
    if adapter_path:
        fused = mlx_lm.load(MODEL, adapter_path=adapter_path)
        model, tokenizer = fused

    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return mlx_lm.generate(
        model, tokenizer, prompt=prompt_text, max_tokens=max_tokens, verbose=False
    )


def main():
    print("=" * 70)
    print("MOLA End-to-End Validation")
    print("=" * 70)

    # Load base model once
    print(f"\nLoading base model: {MODEL}")
    t0 = time.time()
    base_model, tokenizer = mlx_lm.load(MODEL)
    print(f"  Base model loaded in {time.time() - t0:.1f}s")

    # Load with adapters
    print(f"\nLoading rust adapter: {RUST_ADAPTER}")
    t0 = time.time()
    rust_model, rust_tok = mlx_lm.load(MODEL, adapter_path=RUST_ADAPTER)
    print(f"  Rust adapter loaded in {time.time() - t0:.1f}s")

    print(f"\nLoading sql adapter: {SQL_ADAPTER}")
    t0 = time.time()
    sql_model, sql_tok = mlx_lm.load(MODEL, adapter_path=SQL_ADAPTER)
    print(f"  SQL adapter loaded in {time.time() - t0:.1f}s")

    def gen(model, tok, prompt):
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return mlx_lm.generate(
            model, tok, prompt=prompt_text, max_tokens=MAX_TOKENS, verbose=False
        )

    # -- Rust prompts --
    print("\n" + "=" * 70)
    print("RUST PROMPTS")
    print("=" * 70)
    for prompt in RUST_PROMPTS:
        print(f"\n>>> {prompt}")
        print(f"\n[BASE]")
        print(gen(base_model, tokenizer, prompt)[:300])
        print(f"\n[RUST ADAPTER]")
        print(gen(rust_model, rust_tok, prompt)[:300])
        print("-" * 40)

    # -- SQL prompts --
    print("\n" + "=" * 70)
    print("SQL PROMPTS")
    print("=" * 70)
    for prompt in SQL_PROMPTS:
        print(f"\n>>> {prompt}")
        print(f"\n[BASE]")
        print(gen(base_model, tokenizer, prompt)[:300])
        print(f"\n[SQL ADAPTER]")
        print(gen(sql_model, sql_tok, prompt)[:300])
        print("-" * 40)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
