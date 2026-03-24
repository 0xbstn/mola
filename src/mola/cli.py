"""CLI entry point.

    mola serve --model Qwen3.5-35B-A3B-4bit --adapter solana ./adapters/solana
    mola adapters list
    mola generate --adapter solana "Write a Solana transfer"
"""

from __future__ import annotations

import logging

import click

logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def main(verbose: bool):
    """MOLA — Multi-adapter Orchestration LoRA Apple."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(name)s | %(message)s")


@main.command()
@click.option("--model", required=True, help="Base model path or HuggingFace ID")
@click.option(
    "--adapter",
    multiple=True,
    nargs=2,
    metavar="NAME PATH",
    help="Adapter name and path (can be repeated)",
)
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8000, type=int)
def serve(model: str, adapter: tuple, host: str, port: int):
    """Start the MOLA inference server."""
    import uvicorn

    from mola.model import MOLAModel
    from mola.server import create_app

    mola_model = MOLAModel(model)

    for name, path in adapter:
        mola_model.load_adapter(name, path)

    app = create_app(mola_model)

    click.echo(f"MOLA serving on http://{host}:{port}")
    click.echo(f"  Base model: {model}")
    click.echo(f"  Adapters: {[name for name, _ in adapter] or ['none']}")
    click.echo()
    click.echo("Endpoints:")
    click.echo(f"  POST   http://{host}:{port}/v1/chat/completions")
    click.echo(f"  GET    http://{host}:{port}/v1/adapters")
    click.echo(f"  POST   http://{host}:{port}/v1/adapters")
    click.echo(f"  DELETE http://{host}:{port}/v1/adapters/{{name}}")

    uvicorn.run(app, host=host, port=port, log_level="info")


@main.command()
@click.option("--model", required=True, help="Base model path or HuggingFace ID")
@click.option("--adapter-name", default=None, help="Adapter to use")
@click.option("--adapter-path", default=None, help="Path to adapter")
@click.option("--max-tokens", default=256, type=int)
@click.option("--temp", default=0.7, type=float)
@click.argument("prompt")
def generate(
    model: str,
    adapter_name: str | None,
    adapter_path: str | None,
    max_tokens: int,
    temp: float,
    prompt: str,
):
    """Generate text with an optional adapter (one-shot, no server)."""
    from mola.model import MOLAModel

    mola_model = MOLAModel(model)

    if adapter_name and adapter_path:
        mola_model.load_adapter(adapter_name, adapter_path)

    result = mola_model.generate(
        prompt, adapter_id=adapter_name, max_tokens=max_tokens, temp=temp
    )
    click.echo(result)


if __name__ == "__main__":
    main()
