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
@click.option(
    "--max-inflight-tokens",
    default=32768,
    show_default=True,
    type=int,
    help="Global in-flight token budget for admission control",
)
@click.option(
    "--enable-routed-decode-reference",
    is_flag=True,
    help="Enable the experimental homogeneous routed-decode reference path",
)
@click.option(
    "--strict-routed-decode-reference",
    is_flag=True,
    help="Fail closed on routed decode contract mismatches during experimental validation",
)
@click.option(
    "--routed-decode-backend",
    type=click.Choice(["reference", "metal-kernel", "gather-mm"]),
    default="reference",
    show_default=True,
    help="Backend used behind the routed decode session factory when routed decode is enabled",
)
@click.option(
    "--enable-mixed-decode-migration",
    is_flag=True,
    help="Experimentally migrate decode-ready adapted requests into a shared mixed decode generator",
)
def serve(
    model: str,
    adapter: tuple,
    host: str,
    port: int,
    max_inflight_tokens: int,
    enable_routed_decode_reference: bool,
    strict_routed_decode_reference: bool,
    routed_decode_backend: str,
    enable_mixed_decode_migration: bool,
):
    """Start the MOLA inference server."""
    import uvicorn

    from mola.model import MOLAModel
    from mola.engine import EngineConfig
    from mola.server import create_app

    mola_model = MOLAModel(model)

    for name, path in adapter:
        mola_model.load_adapter(name, path)

    app = create_app(
        mola_model,
        EngineConfig(
            max_inflight_tokens=max_inflight_tokens,
            enable_routed_decode_reference=enable_routed_decode_reference,
            strict_routed_decode_reference=strict_routed_decode_reference,
            routed_decode_backend=routed_decode_backend,
            enable_mixed_decode_migration=enable_mixed_decode_migration,
        ),
    )

    click.echo(f"MOLA serving on http://{host}:{port}")
    click.echo(f"  Base model: {model}")
    click.echo(f"  Adapters: {[name for name, _ in adapter] or ['none']}")
    click.echo(f"  Routed decode reference: {'on' if enable_routed_decode_reference else 'off'}")
    click.echo(f"  Routed decode strict: {'on' if strict_routed_decode_reference else 'off'}")
    click.echo(f"  Routed decode backend: {routed_decode_backend}")
    click.echo(f"  Mixed decode migration: {'on' if enable_mixed_decode_migration else 'off'}")
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
