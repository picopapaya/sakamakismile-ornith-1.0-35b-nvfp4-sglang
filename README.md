# sakamakismile-ornith-1.0-35b-nvfp4-sglang

Docker image that runs a **community NVFP4 requantization of Ornith-1.0-35B-MoE** as an OpenAI-compatible API server, built for the **NVIDIA GB10 (DGX Spark)**.

The base model, [`deepreinforce-ai/Ornith-1.0-35B`](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B), is a coding-agent reasoning model published only in BF16 and GGUF — there is no official NVIDIA or deepreinforce-ai NVFP4 build. This image instead runs [`sakamakismile/Ornith-1.0-35B-NVFP4`](https://huggingface.co/sakamakismile/Ornith-1.0-35B-NVFP4), a third-party quantization, served with [SGLang](https://github.com/sgl-project/sglang).

## What this image is

Ornith-1.0-35B-MoE shares its architecture with the Qwen3.6-35B-A3B images in `../qwen-qwen3.6-35b-a3b-fp8-sglang` and `../nvidia-qwen3.6-35b-a3b-nvfp4-sglang` — same `qwen3_5_moe` model type, confirmed from the official model's `config.json`:

- 35 billion total parameters, but for any given word it only actually uses about 3 billion of them: 256 experts per layer, and a router picks 8 of them (plus one always-on "shared" expert) for each token. The rest sit in memory ready to be picked, but don't add to the compute cost.
- Most of its layers (3 out of every 4, across 40 layers total) use a cheaper, memory-light form of attention; only 1 in 4 uses the full, more expensive kind — this is what lets it handle very long conversations without needing huge amounts of extra memory for each one.
- Ships with a multimodal rope configuration, matching the vision-capable Qwen3.5-MoE line — treat it as text+image capable unless testing on this checkpoint shows otherwise.
- Supports up to 262,144 tokens (roughly 200,000 words) of context.
- It's a reasoning model: every reply opens with a `<think>` block before the final answer. The official serving recipe calls for the `qwen3` reasoning parser and `qwen3_coder` tool-call parser (same as the Qwen3.6 siblings), and recommends giving it room to think (`max_tokens` well above 6,000) for one-shot tasks.

## Unofficial weights — read before trusting this in production

Every NVFP4 build of Ornith 35B-MoE on Hugging Face is a community re-quantization — there is no official one. Several carry names signaling they've had safety training deliberately removed or otherwise modified (`AEON-Ultimate-Uncensored`, `abliterated`, `PABLOG-OPTIMIZED_UNCENSORED`, etc.). This image uses `sakamakismile/Ornith-1.0-35B-NVFP4` specifically because it carries no such branding — but it is still an **unverified third-party upload** from an uploader with no relationship to deepreinforce-ai. Nobody has audited its weights, tokenizer, or any bundled code.

If you need weights you can actually trust to be unmodified, the safer options are:
- Run the official BF16 checkpoint (`deepreinforce-ai/Ornith-1.0-35B`, MIT-licensed) and let SGLang quantize on load, or
- Wait for/request an official NVIDIA ModelOpt NVFP4 build, the way `nvidia/Qwen3.6-35B-A3B-NVFP4` exists for the Qwen sibling.

Per that repo's model card: all linear layers are quantized to NVFP4 (W4A4, group size 16), bringing the download from ~70.3 GB (BF16) down to ~21.9 GB.

## Configuration

### Tunable via `.env`

These have a default baked into the image, but you can override them per-deployment by setting them in a `.env` file next to `docker-compose.yml`. Docker Compose reads that file automatically and passes the values into the container when it starts — no image rebuild needed, just edit `.env` and restart.

| Variable | Default | What it does |
|---|---|---|
| `HF_TOKEN` | *(empty)* | Optional Hugging Face token — avoids download rate limits |
| `CONTEXT_LEN` | `262144` | The longest conversation/prompt (in tokens) the server will accept |
| `MEM_FRACTION` | `0.85` | How much of the GPU's memory this server is allowed to claim |
| `ATTENTION_BACKEND` | `triton` | Which kernel library handles the attention math (`flashinfer` measured identical on this box, 2026-07-09) |
| `EXTRA_ARGS` | *(empty)* | Passed straight through to `sglang.launch_server` — escape hatch for flags not covered above |

This deployment's `.env` sets `CONTEXT_LEN=131072` (128K) and `MEM_FRACTION=0.5` — see "Box-specific findings" below.

### Fixed — not overridable via `.env`

These define what this image *is*, not how it's tuned. Changing them means you're describing a different image, not adjusting this one.

| Variable | Value | Why it's fixed |
|---|---|---|
| `MODEL_ID` | `sakamakismile/Ornith-1.0-35B-NVFP4` | This is which model the image downloads and runs — that's the image's whole identity |
| `QUANTIZATION` | `auto` | This is a community requant, not NVIDIA ModelOpt's mixed-precision format — SGLang reads the quant scheme from the checkpoint's own config instead of a forced flag, since forcing the wrong one aborts startup |
| `KV_CACHE_DTYPE` | `auto` | Left to SGLang to pick automatically |
| `MAX_RUNNING_REQUESTS` | `4` | Not currently wired up as a `.env` override — could be added if a need for it comes up |
| `REASONING_PARSER` | `qwen3` | Needed so SGLang understands this model's "thinking" output format |
| `TOOL_CALL_PARSER` | `qwen3_coder` | Needed so SGLang understands this model's function-calling output format |

## Requirements

- NVIDIA GB10 / DGX Spark (SM_121a)
- Docker with NVIDIA Container Toolkit
- The `llm-net` Docker network: `docker network create llm-net`
- A Hugging Face token is optional (model is not expected to be gated)

## Usage

```bash
# Prod — pull image from Docker Hub
docker compose up

# Dev — build image locally
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

The server starts on port **30000** (inside `llm-net`, not published to the host) and exposes an OpenAI-compatible API once the health check passes — allow significant time for the first run while the ~22 GB of weights download.

## License

MIT
