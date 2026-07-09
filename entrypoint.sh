#!/usr/bin/env bash
# Launch Ornith-1.0-35B-MoE (community NVFP4 requant) via SGLang on the GB10.
set -euo pipefail

echo "==> Ornith-1.0-35B-MoE (community NVFP4) + SGLang on NVIDIA GB10 (DGX Spark)"
echo "    model=${MODEL_ID}  quant=${QUANTIZATION}  max-concurrent=${MAX_RUNNING_REQUESTS}"

# Unofficial third-party weights — no HF gating expected, but a token still
# avoids anonymous rate limits on the ~22 GB download.
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
else
  echo "    (HF_TOKEN not set — downloading anonymously)"
fi

# Optional prefetch: downloads weights into the mounted HF_HOME volume so the
# download is a visible, cacheable step separate from server startup.
if [[ "${PREFETCH:-1}" == "1" ]]; then
  echo "==> Downloading ${MODEL_ID} into ${HF_HOME} (cached on the mounted volume)"
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}')" || \
    echo "   (prefetch skipped/failed; SGLang will download on startup)"
fi

# This is a community NVFP4 requant, not NVIDIA ModelOpt's mixed-precision
# format — let SGLang read the quant scheme from the checkpoint's own
# quantization_config instead of forcing one, since forcing the wrong value
# aborts startup (same lesson learned in the NVFP4 Qwen3.6 sibling image).
QUANT_ARGS=()
if [[ -n "${QUANTIZATION:-}" && "${QUANTIZATION}" != "auto" ]]; then
  QUANT_ARGS=(--quantization "${QUANTIZATION}")
fi

echo "==> Launching SGLang server on ${HOST}:${PORT}"
exec python3 -m sglang.launch_server \
  --model-path "${MODEL_ID}" \
  --host "${HOST}" \
  --port "${PORT}" \
  "${QUANT_ARGS[@]}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --context-length "${CONTEXT_LEN}" \
  --mem-fraction-static "${MEM_FRACTION}" \
  --tp-size 1 \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --reasoning-parser "${REASONING_PARSER}" \
  --tool-call-parser "${TOOL_CALL_PARSER}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  ${EXTRA_ARGS}
