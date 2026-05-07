# BrainDrain

LLM Denial-of-Service Research Framework — an interactive CLI for running sustained prompt-injection DoS attacks against LLM serving infrastructure and measuring server degradation in real time.

> **Research use only.** Only target systems you own or have explicit written authorisation to test.

---

## Overview

BrainDrain orchestrates concurrent attack requests designed to exhaust GPU KV-cache memory and compute on LLM serving backends (vLLM, llama.cpp, Ollama, managed APIs, etc.). It supports two attack techniques from published research, a suite of side-channel probes that measure server degradation while the attack runs, and several tools to help calibrate the right attack parameters for a given target.

---

## Installation

```bash
git clone <repo>
cd BrainDrain
python -m venv venv && source venv/bin/activate
pip install -e .
# Optional: richer terminal output
pip install rich
```

**Requirements:** Python ≥ 3.10, `aiohttp`, `numpy`, `cmaes`.

---

## Running the CLI

```bash
python BrainDrain.py
```

The CLI is fully interactive — it walks you through a wizard with sensible defaults at each step. Press **Enter** to accept the value shown in brackets.

---

## Wizard walkthrough

### 1. Target Configuration

The first section configures the endpoint that will be attacked.

#### API format

```
API format (openai / anthropic / ollama / vertex / custom) [openai]:
```

| Format | Endpoint path used | Notes |
|---|---|---|
| `openai` | `/v1/chat/completions` | Any OpenAI-compatible server (vLLM, llama.cpp, together.ai, …) |
| `anthropic` | `/v1/messages` | Direct Anthropic API or Claude-compatible servers |
| `ollama` | `/api/chat` | Local Ollama instances |
| `vertex` | Full model URL (Vertex style) | Google Vertex AI — see below |
| `custom` | User-defined | Paste a full JSON request descriptor — see below |

#### Target URL / IP

```
Target URL / IP (e.g. http://1.2.3.4 or https://api.example.com):
```

Base URL of the serving endpoint. The scheme defaults to `http://` if omitted. For Vertex AI, paste the complete model path URL (the model name is extracted automatically).

#### Model name

```
Model name (e.g. deepseek-r1-7b, gpt-4o):
```

The model identifier forwarded in every request. For Vertex AI this is parsed from the URL automatically.

#### API key

```
API key (leave blank if not required):
```

Bearer token sent as `Authorization: Bearer <key>`. Leave blank for unauthenticated local servers. For Vertex AI, pass the OAuth2 access token from `gcloud auth print-access-token`.

#### Request timeout

```
Request timeout (seconds, leave blank for no timeout) [N/A]:
```

Per-request HTTP timeout. Enter `N/A` or leave blank for no timeout (recommended for long-running attack requests that are expected to take minutes). Enter a number of seconds to cap individual requests.

#### stream_options support (Ollama only)

```
Server supports stream_options (OpenAI >=1.x)? (y/n) [y]:
```

Only asked for Ollama. Set to `n` if your Ollama version does not support the `stream_options` field in the request body (older versions will reject it).

---

#### Vertex AI format

When `vertex` is selected, enter the full Vertex AI endpoint URL:

```
https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/LOCATION/publishers/google/models/MODEL
```

Example:
```
https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-2.5-pro
```

The model name is extracted from the last path segment automatically. Use `gcloud auth print-access-token` to obtain a short-lived OAuth2 token as the API key.

---

#### Custom format

When `custom` is selected, paste a complete JSON request descriptor. The tool injects the attack prompt at runtime by replacing the sentinel string `"__PROMPT__"` in the body.

**Format:**
```json
{
  "url": "https://...",
  "headers": {
    "Authorization": "Bearer ...",
    "anthropic-beta": "interleaved-thinking-2025-05-14"
  },
  "body": {
    "model": "claude-opus-4-5",
    "max_tokens": 16000,
    "thinking": { "type": "enabled", "budget_tokens": 10000 },
    "messages": [{"role": "user", "content": "__PROMPT__"}]
  }
}
```

- `url` — the exact endpoint URL to POST to.
- `headers` — all HTTP headers except `Content-Type` (set automatically).
- `body` — the full request body. Replace the message content value with the literal string `"__PROMPT__"`.
- `stream` and `max_tokens` are injected automatically at runtime; you can still include them as defaults.

After pasting the JSON, press **Enter** on a blank line to finish. You will then be asked to choose a response stream format (for parsing only) and a display model name.

---

### 2. KV Cache Saturation Calculator (optional)

```
Run KV cache saturation calculator? (y / n) [n]:
```

An interactive calculator that estimates how many concurrent attacker instances are needed to completely fill the KV cache on a single GPU. You provide the GPU and model parameters and it outputs the minimum number of concurrent requests required.

**Inputs:**

| Prompt | Description |
|---|---|
| GPU VRAM (GB) | Total VRAM of the GPU (e.g. `96` for an RTX Pro 6000) |
| Model weights on-device (GB) | Size after quantisation (e.g. `14` for DeepSeek-R1-7B in bf16) |
| GPU memory utilisation fraction | vLLM's `--gpu-memory-utilization` setting, typically `0.85` |
| Activation overhead (GB) | CUDA workspace + activation buffers; typically `2` for a 7B model |
| Number of layers | Transformer depth (e.g. `28` for Qwen2.5-7B) |
| Number of KV heads | GQA KV-head count, **not** Q-head count (e.g. `4` for Qwen2.5-7B) |
| Head dimension | `hidden_size / num_q_heads` (e.g. `128`) |
| KV cache dtype bytes | `2` for fp16/bf16, `1` for fp8 |
| Prompt tokens per request | Approximate token count of the attack prompt |
| Max tokens per request | The `max_tokens` value you plan to use in the attack |

**Output:** the minimum number of concurrent requests to fill the KV cache. Use this as the lower bound for the "Number of concurrent attacker instances" setting in the attack configuration.

---

### 3. Load Balancer Detection (optional)

```
Run load-balancer detection test before attacking? (y / n) [n]:
```

Sends pairs of identical long-prefix requests to the target and measures whether the second request in each pair receives an Automatic Prefix Caching (APC) cache hit (significantly lower TTFT). Consistent cache hits indicate the requests land on the same backend node; absent hits indicate a load balancer is distributing them across multiple backends.

**Inputs:**

| Prompt | Description |
|---|---|
| Number of probe pairs | How many warm/cold request pairs to send (default `6`) |
| Prompt repeat factor | Controls prefix length — higher means a longer shared prefix and a more detectable cache hit (default `12`) |

**Output:** verdict (`SAME_BACKEND` vs `LOAD_BALANCED`), cache hit rate, median warm/cold TTFT ratio, and a per-pair table.

**Why this matters:** If requests always land on the same backend, setting a shared system prompt pins all attack and probe requests to the same KV-cache slot via prefix-aware routing, which concentrates the pressure on a single node. If a load balancer is detected you may need to run multiple independent attack processes or increase the number of instances to cover all backends.

---

### 4. System Prompt and Request Prefix

```
System prompt (leave blank to omit):
Request prefix (leave blank to omit):
```

Both fields are sent with every attack and probe request.

- **System prompt** — set as the `system` field (Anthropic format) or as a system-role message (OpenAI format).
- **Request prefix** — prepended to the user message content of every request (attack and probe).

A shared prefix pins all concurrent requests to the same KV-cache slot via prefix-aware routing on vLLM (Automatic Prefix Caching). This maximises cache pressure on a single memory region, making the attack more effective against single-node deployments.

---

### 5. Attack Configuration

#### Attack type

```
Attack type (reasoning_bomb / think_trap) [reasoning_bomb]:
```

| Type | Description |
|---|---|
| `reasoning_bomb` | Feeds short math/logic puzzles to a Large Reasoning Model. The model's extended chain-of-thought produces a pathologically long reasoning trace, achieving a very high output/input token amplification ratio. Based on [Liu et al., 2026](https://arxiv.org/abs/2602.00154). Requires puzzle JSON file. |
| `think_trap` | Injects pre-optimised adversarial prompts generated offline by a CMA-ES search that maximised output token count against a surrogate model. Based on [Li et al., NDSS 2026](https://arxiv.org/abs/2512.07086). Requires a prompt cache JSON file. |

---

#### Number of concurrent attacker instances

```
Number of concurrent attacker instances [4]:
```

How many independent request loops to run in parallel. Each instance continuously sends a new request as soon as the previous one completes. Higher values fill the KV cache faster. Use the KV saturation calculator output as the target minimum.

---

#### Token budget

```
Token budget (leave blank for unlimited):
```

Optional total output token cap across all instances combined. The attack stops automatically once the cumulative output + reasoning token count reaches this limit. Leave blank to run until you press **Ctrl+C**.

---

#### Max tokens per request

```
Max tokens per request (max_completion_tokens) [<auto-detected>]:
```

The `max_tokens` parameter sent in every request. This controls the upper bound on how many tokens the server will generate per response. The CLI probes the target automatically to detect its context window limit and suggests a value that accounts for the prompt size:

```
max_completion_tokens = context_window - prompt_tokens - 64
```

Set this as high as the model allows to maximise KV-cache occupancy per request.

---

#### Reasoning Bomb specific options

```
Path to puzzle JSON file [prompts/reasoningBomb_puzzles.json]:
Puzzle budget tier (128 / 256 / 512 / mixed) [256]:
```

**Puzzle file:** Path to the `sample_puzzles.json` from the [ReasoningBomb GitHub repo](https://github.com/SaFo-Lab/ReasoningBomb/blob/main/data/sample_puzzles.json). Save it to `prompts/reasoningBomb_puzzles.json` or provide an absolute path.

**Budget tier:** The token-budget tier of puzzles to draw from. Puzzles in each tier are designed to induce a specific reasoning depth in the model.

| Tier | Prompt length | Typical amplification |
|---|---|---|
| `128` | ~128 tokens | Highest (shortest input → most amplification) |
| `256` | ~256 tokens | High |
| `512` | ~512 tokens | Moderate |
| `mixed` | Rotates 128/256/512 | Distributes across tiers; each instance gets a different tier to prevent synchronised completions |

Each concurrent instance is assigned a different puzzle index so requests do not share a prompt (which would allow the server to serve them from prefix cache rather than computing a full response).

---

#### ThinkTrap specific options

```
Path to ThinkTrap prompts JSON file [prompts/thinktrap_prompts.json]:
```

Path to the adversarial prompt cache produced by `ThinkTrapAPG`. This file must be generated offline before running the attack (see [generating ThinkTrap prompts](#generating-thinktrap-prompts-offline)).

---

### 6. Sustained Pressure Options

These settings control how the attack avoids the "wave pattern" — where all instances happen to finish at the same time, briefly releasing the KV cache and letting the scheduler recover.

#### Launch stagger

```
Launch stagger between instances (seconds, 0=off) [0.0]:
```

Delay between starting each successive instance. Staggered launch times cause instances to complete at different wall-clock times, maintaining continuous KV-cache pressure even across request boundaries.

**Rule of thumb:** `estimated_request_duration / n_instances`. For example, if each request takes ~60 seconds and you run 4 instances, a stagger of 15 seconds distributes completion times evenly.

#### Max-token spread

```
Max-token spread (%, 0=off, e.g. 25) [0.0]:
```

Assigns each instance a slightly different `max_tokens` value distributed linearly across the range `[max_tokens × (1 - spread%), max_tokens]`. This ensures that even after the initial stagger window has passed, instances continue to complete at different times because they hit their token limits at different points.

Example: with `max_tokens=16000` and spread `25%`, 4 instances get `[12000, 13333, 14667, 16000]` tokens.

#### Stream read delay

```
Stream read delay per chunk (seconds, 0=off) [0.0]:
```

Inserts a sleep between consuming each streamed chunk from the server. This backs up the server's output queue, extending the effective KV-block occupancy of each request beyond its pure generation time. The connection stays open and the KV blocks remain allocated until all tokens are consumed.

Recommended range: `0.005` – `0.02` seconds (5–20 ms per chunk).

#### ITL probing

```
Enable ITL probing during attack? (y / n) [y]:
```

When enabled, the orchestrator fires lightweight probe requests every 10 seconds and displays real-time KV-cache utilisation estimates and infra state classifications in the live status panel. When disabled, no probe requests are sent (useful if the target rate-limits all traffic and you need to preserve the full quota for attack requests).

---

## Live Status Display

During the attack a live panel shows:

| Field | Description |
|---|---|
| Elapsed | Wall-clock time since launch |
| Instances | Number of active attack loops |
| Requests | Total requests completed so far |
| Output tokens | Cumulative completion + reasoning tokens generated (vs budget if set) |
| Input tokens | Cumulative prompt tokens consumed |
| Infra state | Current classified server state (see below) |
| KV est | Estimated KV-cache utilisation ∈ [0, 1] |
| Mean ITL | Mean inter-token latency across recent probe window (ms) |
| TTFT | Time-to-first-token from most recent probe (s) |
| Errors | Count of failed requests and last error message |

### Infra states

| State | Meaning |
|---|---|
| `unknown` | No baseline calibrated or insufficient probe data |
| `idle` | Near-baseline ITL and TTFT; KV usage low |
| `filling` | KV usage rising; ITL trending up (1.5–3× baseline) |
| `saturated` | KV near capacity; ITL >> baseline (3×+) |
| `hol_blocked` | TTFT has spiked — WAITING queue is frozen (head-of-line blocking) |
| `thrashing` | High ITL variance — preemption/eviction loop detected |
| `degraded` | Significant degradation that does not match other categories |

Press **Ctrl+C** at any time to stop the attack and print the final summary.

---

## Error Recovery

If 10 consecutive requests fail with the same classifiable error, the attack pauses automatically and prompts you to fix the configuration:

| Error category | Recovery prompt |
|---|---|
| `token_limit` | Enter a lower `max_tokens` value |
| `auth` | Enter a new API key |
| `model_not_found` | Enter the correct model name |
| `rate_limit` | Enter an inter-request delay in seconds |

After correcting the setting, choose `y` to restart the instances with the updated configuration, or `n` to stop.

---

## Attack Summary

After stopping, the CLI prints a summary table:

| Metric | Description |
|---|---|
| Wall clock | Total elapsed time |
| Total requests | Requests sent (successful + failed) |
| Successful | Requests that completed with a response |
| Timed out | Requests that hit the timeout limit |
| Cancelled | Requests cancelled on shutdown |
| Failed | Requests that returned an error |
| Mean amp ratio | Mean (output tokens) / (input tokens) amplification |
| P95 amp ratio | 95th-percentile amplification ratio |
| Output tokens | Total completion + reasoning tokens generated |
| Input tokens | Total prompt tokens consumed |
| Total tokens | Sum of output and input |
| Tokens/s | Output token generation rate |
| Mean TTFT | Mean time-to-first-token across all requests |
| Mean duration | Mean total request duration |

---

## Generating ThinkTrap Prompts (Offline)

Before running a `think_trap` attack you must generate the adversarial prompt cache. This is a one-time offline step that queries the target (or a compatible surrogate) to find prompts that maximise output length via CMA-ES optimisation.

```python
import asyncio
from attacks.think_trap import ThinkTrapAPG, ThinkTrapConfig, make_victim_fn, load_surrogate_embeddings
from core.base_attack import TargetConfig, APIFormat

target = TargetConfig(
    base_url="http://your-target:8000",
    model="your-model",
    api_format=APIFormat.OPENAI,
)

cfg = ThinkTrapConfig(
    surrogate_model="meta-llama/Llama-2-7b-hf",  # or use surrogate_embeddings_path
    prompt_length=20,
    latent_dim=20,
    query_budget=200,
    top_k_keep=10,
)

T, tokenizer = load_surrogate_embeddings(cfg.surrogate_model)
victim = make_victim_fn(target, max_tokens=4096, tokenizer=tokenizer)

apg = ThinkTrapAPG(victim_fn=victim, T_surrogate=T, config=cfg, tokenizer=tokenizer)
asyncio.run(apg.run())
apg.save("prompts/thinktrap_prompts.json")
```

To avoid downloading the full model weights every run, save the embedding matrix once:

```python
import numpy as np
T, tokenizer = load_surrogate_embeddings("meta-llama/Llama-2-7b-hf")
np.save("llama2_embeddings.npy", T)
# Next time:
T, _ = load_surrogate_embeddings("", embeddings_path="llama2_embeddings.npy")
```

---

## File layout

```
BrainDrain/
├── BrainDrain.py              # CLI entry point (run this)
├── attacks/
│   ├── reasoning_bomb.py      # ReasoningBomb attack module
│   └── think_trap.py          # ThinkTrap attack module (APG + DSA)
├── core/
│   └── base_attack.py         # Abstract base class, TargetConfig, AttackConfig
├── orchestration/
│   ├── attacker_instance.py   # Single sustained request loop
│   ├── registry.py            # Attack config factory helpers
│   └── result_collector.py    # Result aggregation and summary
├── probes/
│   ├── itl_probes.py          # ITL side-channel probe suite
│   └── kv_saturation_calculator.py
└── prompts/
    ├── reasoningBomb_puzzles.json   # Download from ReasoningBomb repo
    └── thinktrap_prompts.json       # Generated by ThinkTrapAPG
```

---

## Quick example: attacking a local vLLM server

```
python BrainDrain.py

  API format: openai
  Target URL: http://localhost:8000
  Model name: deepseek-r1-7b
  API key:    (blank)
  Timeout:    N/A

  KV calculator: n
  LB detection:  n
  System prompt: (blank)
  Request prefix:(blank)

  Attack type:    reasoning_bomb
  Instances:      8
  Token budget:   (blank — unlimited)
  [auto-detected context limit: 131072 tokens]
  Max tokens:     130944

  Launch stagger: 8.0   (est. 60s request / 8 instances)
  Token spread:   25
  Stream delay:   0.01
  ITL probing:    y
```
