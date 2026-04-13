from huggingface_hub import login
import logging
import asyncio
from core.base_attack import APIFormat, TargetConfig
from attacks.think_trap import (
    ThinkTrapConfig,
    ThinkTrapAPG,
    load_surrogate_embeddings,
    make_victim_fn,
)
import secret
login(token=secret.huggingface_token)

logging.basicConfig(level=logging.INFO)

runpod_vllm = TargetConfig(
    base_url="https://xx3t94ynacne3q-8000.proxy.runpod.net",
    model="deepseek-r1-7b",           # must match --served-model-name
    api_format=APIFormat.CUSTOM,       # vLLM is OpenAI-compatible
    api_key=secret.runpod_api_key,
    timeout=600.0,
)

target = runpod_vllm

# ── Parameters ───────────────────────────────────────────────────────────────
# Prompt length (tokens).  Choose one of: 20, 40, 60, 80.
# Derived automatically:
#   latent_dim   = p_length              (paper defaults both to 20; 1:1 ratio)
#   query_budget = min(p_length * 10, 600)  → 200 / 400 / 600 / 600
P_LENGTH = 20

# ─────────────────────────────────────────────────────────────────────────────

latent_dim   = P_LENGTH
query_budget = 200
out_path     = f'prompts/thinktrap_prompts_{P_LENGTH}.json'


async def main():
    # Load surrogate embeddings once (slow — downloads weights on first run)
    T, tok = load_surrogate_embeddings("mistralai/Mistral-7B-v0.1", hf_token=secret.huggingface_token)

    # Option B: from a pre-saved .npy file (fast, after first run)
    # np.save("mistral_embeddings.npy", T)   # save once
    # T, tok = load_surrogate_embeddings("", embeddings_path="mistral_embeddings.npy")

    cfg = ThinkTrapConfig(
        prompt_length=P_LENGTH,
        latent_dim=latent_dim,
        cmaes_sigma=1.0,
        query_budget=query_budget,
        top_k_keep=20,
        prompts_file=out_path,
    )

    logging.info(
        "APG: L=%d  m=%d  budget=%d  → %s",
        P_LENGTH, latent_dim, query_budget, out_path,
    )

    victim = make_victim_fn(target, max_tokens=10000, tokenizer=tok)
    apg = ThinkTrapAPG(victim_fn=victim, T_surrogate=T, config=cfg, tokenizer=tok, seed=42)
    await apg.run()
    apg.save(out_path)
    print(f"[L={P_LENGTH}] Saved {len(apg.best_prompts)} prompts → {out_path}")

asyncio.run(main())
