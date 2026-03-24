# MOLA Sources

Date: 2026-03-24

This file is a compact source library for the pre-kernel and kernel work.
It is not a literature review. The goal is to keep the links that are most useful for MOLA in one place, with one short note per source.

## Official MLX

- MLX docs index
  - https://ml-explore.github.io/mlx/build/html/index.html
  - Canonical entry point for MLX docs, including custom kernels, extensions, and Metal debugging.

- Custom Metal Kernels
  - https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
  - Best first path for a Python-level kernel prototype via `mlx.core.fast.metal_kernel`.

- `mlx.core.fast.metal_kernel`
  - https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html
  - API-level reference for the quickest custom-kernel prototype path.

- Custom Extensions in MLX
  - https://ml-explore.github.io/mlx/build/html/dev/extensions.html
  - The path to a real MLX primitive once a Python-level prototype is worth hardening.

## Multi-LoRA Serving Systems

- Punica repo
  - https://github.com/punica-ai/punica
  - First strong reference for multi-tenant LoRA serving and routed multi-adapter execution.

- Punica paper
  - https://arxiv.org/abs/2310.18547
  - Primary reference for the BGMV-style multi-LoRA serving problem.

- S-LoRA repo
  - https://github.com/S-LoRA/S-LoRA
  - Strong scheduler plus memory-management reference for serving many adapters concurrently.

- S-LoRA paper
  - https://arxiv.org/abs/2311.03285
  - Primary reference for large-scale concurrent LoRA serving tradeoffs.

- LoRAX repo
  - https://github.com/predibase/lorax
  - Product-oriented multi-LoRA serving system with explicit API and adapter lifecycle focus.

- vLLM LoRA Punica wrapper docs
  - https://docs.vllm.ai/en/stable/api/vllm/lora/punica_wrapper/punica_gpu/
  - Useful for the shape of a routed LoRA operator and the kind of helper API a serving runtime eventually needs.

## Apple Silicon / MLX-Adjacent Runtime Work

- `vllm-metal`
  - https://github.com/vllm-project/vllm-metal
  - Apple-Silicon serving reference; not a multi-LoRA server, but useful for serving/runtime separation and memory controls.

- `pmetal`
  - https://github.com/Epistates/pmetal
  - Public Apple-Silicon project with explicit fused-kernel work, LoRA-related kernels, and device-aware tuning ideas.

- MetalRT blog
  - https://huggingface.co/blog/runanywhere/metalrt-fastest-inference-apple-silicon
  - Evidence that direct Metal hot paths still beat stock MLX baselines in some decode-heavy settings.

- ZMLX
  - https://github.com/Hmbown/ZMLX
  - Experimental MLX-adjacent kernel toolkit and fused Apple-Silicon inference work; useful inspiration, not a serving-equivalent.

## AI-Assisted Kernel Discovery

- OpenEvolve GPU kernel discovery
  - https://huggingface.co/blog/codelion/openevolve-gpu-kernel-discovery
  - Best public example of an iterative propose-build-benchmark loop for GPU kernels.

- Gimlet Labs AI-generated Metal kernels
  - https://gimletlabs.ai/blog/ai-generated-metal-kernels
  - Good benchmark-oriented reference for AI-generated Metal kernels versus baseline implementations.

## How These Map To MOLA

- First kernel target:
  - MLX custom kernels docs
  - Punica paper
  - vLLM Punica wrapper docs

- Pre-kernel runtime design:
  - S-LoRA paper/repo
  - LoRAX repo
  - `vllm-metal`

- Apple-Silicon low-level inspiration:
  - `pmetal`
  - MetalRT
  - ZMLX

- Iteration methodology:
  - OpenEvolve
  - Gimlet Labs
