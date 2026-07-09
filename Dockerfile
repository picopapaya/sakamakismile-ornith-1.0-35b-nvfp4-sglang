# Ornith-1.0-35B-MoE, community NVFP4 re-quantization, served by SGLang, on the NVIDIA GB10 (DGX Spark).
# Base model: deepreinforce-ai/Ornith-1.0-35B (MIT). This image runs the
# NVFP4 quant published by a third party: sakamakismile/Ornith-1.0-35B-NVFP4
# (see "Unofficial weights" note below — chosen after checking, no official
# NVIDIA/deepreinforce-ai NVFP4 release exists for this model as of 2026-07-09).
#
# Model architecture (from the official repo's config.json) —
# Ornith-1.0-35B-MoE is a Qwen3.5-MoE ("qwen3_5_moe") hybrid-attention model:
#   - 35B total parameters; 256 experts per MoE layer, router activates 8 per
#     token (+ a shared expert) — the same "sparse MoE" shape as the
#     Qwen3.6-35B-A3B images in ../qwen-qwen3.6-35b-a3b-fp8-sglang and
#     ../nvidia-qwen3.6-35b-a3b-nvfp4-sglang (identical model_type in config.json).
#   - Hybrid: 3 of every 4 layers use linear attention, 1 uses full attention
#     (40 layers total).
#   - Multimodal rope (mrope) present, matching the Qwen3.5-MoE vision-capable
#     line — treat as text+image capable unless the loaded checkpoint proves
#     otherwise.
#   - 262144-token native context (rope_theta 1e7).
#   - Reasoning model: opens each reply with a <think> block. Official serving
#     recipe (deepreinforce-ai model card): SGLang >= 0.5.9, reasoning parser
#     "qwen3", tool-call parser "qwen3_coder".
#
# Quantization — this is a COMMUNITY NVFP4 requant, not an official release:
#   - deepreinforce-ai ships this model only in BF16 (~70 GB) + GGUF. No
#     NVIDIA ModelOpt or deepreinforce-ai NVFP4 build exists. Several
#     community NVFP4 uploads exist under names signaling modified/stripped
#     safety training (e.g. "abliterated", "Uncensored") — this image
#     deliberately uses sakamakismile/Ornith-1.0-35B-NVFP4, which carries no
#     such branding, but it is still an unverified third-party upload; treat
#     it with the same scrutiny as any unvetted model weights.
#   - Per that repo's card: all linear layers quantized to NVFP4 (W4A4, group
#     size 16), ~21.9 GB vs ~70.3 GB BF16.
#   - VERIFIED on this box (2026-07-09): this checkpoint is in the
#     compressed-tensors NVFP4 format, which the stock SGLang release below
#     loads natively (CompressedTensorsW4A4Nvfp4MoE) — no patched nightly
#     needed, unlike ../nvidia-qwen3.6-35b-a3b-nvfp4-sglang whose ModelOpt
#     mixed-precision checkpoint did. Loaded in 71 s, 21.96 GB on GPU.
#     Single-request decode measured ~59 tok/s on an idle GPU (default
#     triton/auto config) — on par with the Qwen3.6 siblings. flashinfer
#     attention and marlin/flashinfer_cutlass MoE runners benchmarked
#     identical; MTP speculative decoding is impossible (the quantizer
#     dropped the draft-head weights). See README "Box-specific findings".
#
# Base image: CUDA 13.x is required for sm_121a; qwen3_5_moe architecture
# support requires SGLang >= v0.5.13. Using the same release tag already
# proven to load this architecture for the FP8 Qwen3.6 sibling.
ARG SGLANG_IMAGE=lmsysorg/sglang:v0.5.14-cu130
FROM --platform=linux/arm64 ${SGLANG_IMAGE}

ENV MODEL_ID="sakamakismile/Ornith-1.0-35B-NVFP4" \
    HOST="0.0.0.0" \
    PORT="30000" \
    QUANTIZATION="auto" \
    KV_CACHE_DTYPE="auto" \
    CONTEXT_LEN="262144" \
    MEM_FRACTION="0.85" \
    MAX_RUNNING_REQUESTS="4" \
    REASONING_PARSER="qwen3" \
    TOOL_CALL_PARSER="qwen3_coder" \
    ATTENTION_BACKEND="triton" \
    EXTRA_ARGS="" \
    HF_HOME="/root/.cache/huggingface" \
    # Point Triton at CUDA 13.0's ptxas instead of PyTorch's bundled one.
    # The bundled ptxas predates SM_121 and rejects --gpu-name=sm_121a, causing
    # JIT compilation failures for attention and other kernels at runtime.
    # /usr/local/cuda/bin/ptxas (from CUDA 13.0 in this image) knows SM_121a natively.
    TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 30000

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
