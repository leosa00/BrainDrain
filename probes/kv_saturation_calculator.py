import math
from typing import Optional


def requests_to_fill_kv(
    gpu_vram_gb, model_weights_gb, gpu_mem_util,
    num_layers, num_kv_heads, head_dim,
    prompt_tokens, max_tokens,
    dtype_bytes=2, activation_overhead_gb=2.0,
    max_model_len: Optional[int] = None,
    verbose=True,
):
    """
    Estimate requests needed to saturate the KV cache on a single GPU.

    gpu_mem_util: fraction of VRAM vLLM is allowed to use (--gpu-memory-utilization)
    model_weights_gb: actual on-device size (depends on quantization — see note below)
    activation_overhead_gb: CUDA workspace + activation buffers beyond model weights;
        vLLM profiles this at startup; typically 1-4 GB for a 32B model
    max_model_len: vLLM's --max-model-len, the hard ceiling on prompt+generated
        tokens for ANY single request. A request can never occupy more KV than this,
        no matter how large max_tokens is — vLLM clamps generation at the limit.
        When set, per-request KV footprint is capped at max_model_len.

    KV memory = (VRAM × gpu_mem_util) - model_weights - activation_overhead
    bytes/token = 2 (K+V) × num_layers × num_kv_heads × head_dim × dtype_bytes
    """
    # KV cache = total reserved memory minus weights and runtime overhead
    kv_mem_gb = gpu_vram_gb * gpu_mem_util - model_weights_gb - activation_overhead_gb
    if kv_mem_gb <= 0:
        raise ValueError(
            f"No memory left for KV cache: {kv_mem_gb:.2f} GB. "
            "Reduce model_weights_gb, lower gpu_mem_util, or quantize further."
        )

    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    total_cache_tokens = (kv_mem_gb * 1024**3) / bytes_per_token

    # A request's KV footprint is prompt + generated tokens, but vLLM never lets a
    # single sequence exceed --max-model-len. If the requested shape would overflow
    # that limit, the server clamps it, so the real per-request footprint is capped.
    requested_tokens = prompt_tokens + max_tokens
    clamped = max_model_len is not None and requested_tokens > max_model_len
    tokens_per_request = min(requested_tokens, max_model_len) if max_model_len else requested_tokens

    if max_model_len is not None and prompt_tokens >= max_model_len:
        raise ValueError(
            f"prompt_tokens ({prompt_tokens}) >= max_model_len ({max_model_len}): "
            "the server would reject every request before generating a single token. "
            "Shorten the prompt or raise --max-model-len."
        )

    if verbose:
        print(f"  Reserved VRAM:       {gpu_vram_gb * gpu_mem_util:.1f} GB")
        print(f"  Model weights:       {model_weights_gb:.1f} GB")
        print(f"  Activation overhead: {activation_overhead_gb:.1f} GB")
        print(f"  KV cache budget:     {kv_mem_gb:.2f} GB")
        print(f"  Bytes/token:         {bytes_per_token / 1024:.1f} KB  "
              f"(2 × {num_layers}L × {num_kv_heads}kv-heads × {head_dim}d × {dtype_bytes}B)")
        print(f"  Cache capacity:      {total_cache_tokens:,.0f} tokens")
        if max_model_len is not None:
            print(f"  max-model-len:       {max_model_len:,} tokens")
        if clamped:
            print(f"  Tokens/request:      {tokens_per_request:,}  "
                  f"(clamped from {requested_tokens:,} = {prompt_tokens} prompt + "
                  f"{max_tokens} max_new → capped at max-model-len)")
        else:
            print(f"  Tokens/request:      {tokens_per_request:,}  "
                  f"({prompt_tokens} prompt + {max_tokens} max_new)")

    return math.ceil(total_cache_tokens / tokens_per_request)


# ── DeepSeek-R1-Distill-Qwen-7B on NVIDIA RTX Pro 6000 Blackwell (96 GB GDDR7) ──
#
# Architecture (Qwen2.5-7B backbone):
#   28 layers, 28 Q-heads, 4 KV-heads (GQA), head_dim = 3584/28 = 128
#   KV footprint: 56 KB/token  (vs 256 KB for the 32B variant)
#
# Quantization note:
#   BF16 full precision ≈ 15 GB — fits easily; leaves ~66 GB for KV cache.
#   4-bit AWQ ≈ 4 GB — leaves even more, but bf16 is the typical vLLM default.
#
if __name__ == "__main__":
    n = requests_to_fill_kv(
        gpu_vram_gb=96,             # RTX Pro 6000 Blackwell
        model_weights_gb=15,        # BF16 7B model
        gpu_mem_util=0.85,
        num_layers=28,              # Qwen2.5-7B backbone
        num_kv_heads=4,             # GQA
        head_dim=128,               # 3584 hidden / 28 Q-heads
        prompt_tokens=250,
        max_tokens=15000,
        dtype_bytes=2,              # bf16 KV cache (vLLM default)
        activation_overhead_gb=2.0,
        max_model_len=32768,        # vLLM --max-model-len for this deployment
    )
    print(f"\nRequests to fill KV cache: {n}")
