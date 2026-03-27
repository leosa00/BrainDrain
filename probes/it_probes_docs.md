# `itl_probes.py` — Function Reference

> Black-box ITL side-channel probing for LLM serving infrastructure.
> Implements all probing mechanisms from the Fill & Squeeze paper (Wang et al., 2026).
>
> **Core insight (§5.2.2):** ITL of standard requests correlates linearly with
> global GPU KV-cache usage due to physical memory bandwidth contention.
> This makes ITL a black-box side-channel for estimating real-time scheduler
> state without privileged access, enabling adaptive Fill & Squeeze control.

---

## Table of Contents

1. [Data Classes](#data-classes)
   - [ProbeResult](#proberesult)
   - [BurstProbeResult](#burstproberesult)
   - [ProbeConfig](#probeconfig)
2. [KVUsageEstimator](#kvusageestimator)
   - [extract_features()](#extract_featuresitl_ms)
   - [fit()](#fititl_sequences-kv_usages)
   - [set_baseline()](#set_baselinebaseline_itl_ms)
   - [estimate()](#estimateitl_ms)
3. [ITLProbes — Infrastructure Methods](#itlprobes--infrastructure-methods)
   - [_get_session()](#_get_session)
   - [_build_payload()](#_build_payloadprompt-max_tokens-stream)
   - [_extract_token()](#_extract_tokenpayload_str)
   - [_stream_probe()](#_stream_probeprompt-max_tokens-probe_type)
   - [_nonstream_probe()](#_nonstream_probeprompt-max_tokens-probe_type)
4. [ITLProbes — Public Probe Methods](#itlprobes--public-probe-methods)
   - [calibrate_baseline()](#calibrate_baselinen_samples-warm_up-inter_delay_s)
   - [single_probe()](#single_probeprompt-max_tokens)
   - [burst_probe()](#burst_proben-prompt-max_tokens)
   - [ttft_probe()](#ttft_probeprompt-use_streaming)
   - [preemption_probe()](#preemption_probeduration_s-max_tokens-spike_threshold)
   - [hol_probe()](#hol_proben_fill-long_prompt-short_prompt)
   - [memory_pressure_scan()](#memory_pressure_scann_probes-interval_s)
   - [estimate_kv_usage()](#estimate_kv_usage)
   - [tpot_probe()](#tpot_proben_probes)
   - [request_timeout_probe()](#request_timeout_proben_probes-hard_timeout)
   - [adaptive_fill_signal()](#adaptive_fill_signalc_sat)
   - [classify_state()](#classify_stateprobe-burst)
   - [monitor()](#monitorinterval_s-on_probe-on_state_change-stop_event-max_iterations)
   - [rolling_summary()](#rolling_summary)

---
## Derived Latency Metrics

These fields are all `0.0` at construction time and populated by
`compute_derived()` after a streaming probe completes. They are computed
from `itl_values` — the list of raw inter-token gap timings (in seconds)
recorded during the stream, converted to milliseconds.

---

## `mean_itl_ms` — Mean Inter-Token Latency

**Formula:** `mean(itl_values × 1000)`

The average time between consecutive token arrivals across the entire
response. This is the primary KV cache pressure signal — as the KV pool
fills up, attention computation becomes increasingly memory-bound, and
each token takes longer to generate.

**What it tells you:**
- Near `baseline_itl_ms` → system is idle, KV pool has free capacity
- 1.5× baseline → KV pool is filling (`FILLING` state)
- 3× baseline → KV near saturation (`SATURATED` state)

**Limitation on hosted APIs (Gemini, OpenAI):** Because multiple tokens
are batched into a single SSE chunk, `mean_itl_ms` is actually the mean
**inter-chunk** latency, not true per-token latency. The signal still
correlates with load, but the absolute value is not comparable to vLLM
where each chunk = 1 token.

---

## `p95_itl_ms` — 95th Percentile Inter-Token Latency

**Formula:** `sorted(itl_values × 1000)[ceil(0.95 × n) - 1]`

The ITL value below which 95% of all token gaps fall. Captures tail
latency spikes that the mean would smooth over.

**What it tells you:**
- A large gap between `mean_itl_ms` and `p95_itl_ms` indicates
  occasional stalls — for example, the scheduler briefly pausing to
  allocate new KV blocks, or a batch boundary on the GPU
- Under preemption, a single evicted-and-recomputed request will produce
  one enormous ITL value that spikes `p95_itl_ms` far above the mean
  while leaving `mean_itl_ms` relatively stable

---

## `p99_itl_ms` — 99th Percentile Inter-Token Latency

**Formula:** `sorted(itl_values × 1000)[ceil(0.99 × n) - 1]`

The ITL value below which 99% of all token gaps fall. More sensitive than
p95 to extreme outliers.

**What it tells you:**
- `p99_itl_ms` is almost always dominated by the **first** ITL value,
  which equals TTFT (the gap from request start to first token). Since
  TTFT includes prefill time and queue wait, it is always larger than
  the decode-phase gaps that follow
- For a response with 4 tokens, `p99 = ceil(3.96) - 1 = 3` — same index
  as p95. They only diverge meaningfully when `n ≥ 20`
- On vLLM under preemption, a sudden multi-second silence mid-stream
  will show up here as a value orders of magnitude above `p95_itl_ms`

---

## `itl_stddev_ms` — Standard Deviation of ITL

**Formula:** `stdev(itl_values × 1000)` (requires n ≥ 2, else 0.0)

Measures how much individual token gaps vary around the mean.

**What it tells you:**
- **Low stddev** (< 20% of mean) → stable decode throughput, scheduler
  is running smoothly
- **High stddev** → erratic token delivery; possible causes:
  - GPU batch boundaries (tokens arrive in bursts then pause)
  - Scheduler preempting and resuming the request
  - KV block allocation stalls between batches
  - On hosted APIs: natural CDN routing variance

**Role in `classify_state()`:** The rolling CoV (= stddev / mean)
computed across `_itl_history` is the THRASHING detector. If
`CoV > itl_cv_thrash_threshold` over the last N probes, the system is
classified as `THRASHING` regardless of the absolute ITL level.

---

## `tpot_ms` — Time Per Output Token

**Formula:** `mean_itl_ms` (they are set to the same value)

TPOT is an alias for mean ITL that emphasises a different interpretation:
where `mean_itl_ms` describes the side-channel signal, `tpot_ms` frames
the same number as a **throughput metric** — how long the model takes to
produce each output token on average.

**Why the distinction matters:**
- `tpot_ms` is used by `tpot_probe()`, which runs N sequential probes and
  aggregates `tpot_ms` values across requests. This cross-request
  aggregation isolates the decode phase from queuing effects (TTFT)
- A rising `tpot_ms` trend across sequential probes is a direct signal of
  GPU memory bandwidth saturation — attention matmuls are O(sequence_length)
  in memory reads, so as KV tensors grow, each token costs more bandwidth
- `tpot_ratio = mean_tpot / baseline_itl` > `itl_fill_threshold` triggers
  `degraded: True` in the `tpot_probe()` result

---

## `tokens_out` — Output Token Count

**Formula:**
```python
if n > 1 and len(raw_text.split()) > n:
    tokens_out = len(raw_text.split())   # hosted API — chunks batch multiple tokens
else:
    tokens_out = n                        # vLLM / Ollama — one token per chunk
```

The number of tokens (or chunks) received during the stream. The value means
different things depending on the backend:

| Backend | Chunk granularity | `tokens_out` accuracy |
|---|---|---|
| vLLM (local) | 1 token per SSE chunk | Exact token count |
| Ollama | 1 token per NDJSON line | Exact token count |
| Gemini / OpenAI hosted | Many tokens per chunk | Word-split estimate (undercount) |

**Why it matters for ITL interpretation:** On hosted APIs where
`tokens_out = 4` but `raw_text` contains 60+ words, the 4 `itl_values`
entries are **inter-chunk** gaps, not per-token gaps. The absolute ITL
values are larger than true per-token ITL, and the statistical metrics
(`mean_itl_ms`, `p95_itl_ms`) reflect chunk delivery cadence rather than
decode speed. On vLLM, `tokens_out` and `len(itl_values)` are equal and
both reflect true token-level timing.

**Note:** `len(raw_text.split())` is a whitespace word-count estimate, not
a BPE token count. Markdown formatting like `**Mercury**` counts as one
word but may be 3–4 BPE tokens. The true token count is only available if
the endpoint returns a `usage.completion_tokens` field in the final stream
chunk (requires `stream_options: {include_usage: true}`).

---

## `kv_usage_est` — KV Cache Utilisation Estimate

**Type:** `float`, range `[0.0, 1.0]`, sentinel value `-1.0`

An estimate of the current global GPU KV cache occupancy, Û_sys, derived
from the ITL values of this probe via `KVUsageEstimator.estimate()`.

**Sentinel value `-1.0`** means the estimate has not been computed yet.
This happens in two cases:
- The probe failed (`success = False`)
- The probe type deliberately skips estimation (e.g. `ttft_probe()` which
  produces only 1 ITL value — the TTFT itself — which is not a meaningful
  KV pressure signal)

**How it is populated:** `single_probe()` calls
`estimator.estimate(itl_values × 1000)` and writes the result back into
`result.kv_usage_est` after `_stream_probe()` returns. The raw
`_stream_probe()` method itself always leaves `kv_usage_est = -1.0` —
estimation is the responsibility of the caller.

**Interpretation:**

| Value | Meaning |
|---|---|
| `-1.0` | Not estimated (probe type skips it, or probe failed) |
| `0.0` | ITL at or below baseline — KV pool has free capacity |
| `0.0 – 0.5` | Moderate pressure — KV filling but not critical |
| `0.5 – 0.8` | High pressure — Fill phase active |
| `> 0.8` | Near saturation — Squeeze phase should begin |

**Important caveat:** Without supervised training via `estimator.fit()`,
the value comes from the unsupervised soft sigmoid fallback. On idle
systems or hosted APIs with natural latency variation, this can return
`0.0` even under real load if the current probe's ITL happens to be at or
below the calibrated baseline. Always interpret trends across multiple
probes rather than trusting a single reading.

---

## `success` — Probe Success Flag

**Type:** `bool`, default `True`

`True` if the HTTP request completed without error and at least the
response headers were received successfully (status < 400).

**Set to `False` when:**
- HTTP status ≥ 400 (e.g. `429 Too Many Requests`, `503 Service Unavailable`)
- `asyncio.TimeoutError` — the request exceeded `TargetConfig.timeout`
- `aiohttp.ClientError` — network-level failure (connection refused, DNS
  failure, TLS error)
- RPD budget exhausted (set by `RateLimitedITLProbes` before the request fires)

**Note:** `success = True` does not mean the response was useful. A probe
can succeed but return `tokens_out = 0` and `raw_text = ""` if the model
hit `max_tokens` during the reasoning phase (DeepSeek-R1 thinking tokens)
before producing any visible output.

---

## `error` — Error Description

**Type:** `Optional[str]`, default `None`

Human-readable error string when `success = False`, otherwise `None`.

**Common values:**

| Value | Cause |
|---|---|
| `"timeout"` | `asyncio.TimeoutError` — request exceeded `TargetConfig.timeout` |
| `"HTTP 429: ..."` | Rate limit hit — `RateLimitedITLProbes` will retry with back-off |
| `"HTTP 503: ..."` | Server 

## Data Classes

### `ProbeResult`

The universal return type for every individual probe. Stores:

- **Raw timing data:** `ttft_s`, `total_s`, `itl_values` (list of per-token gaps in seconds)
- **Derived metrics** (populated by `compute_derived()`): `mean_itl_ms`, `p95_itl_ms`, `p99_itl_ms`, `itl_stddev_ms`, `tpot_ms`, `tokens_out`
- **KV estimate:** `kv_usage_est` — `-1.0` means not yet estimated
- **Status:** `success`, `error`
- **Raw text:** `raw_text` — accumulated token text from the stream

#### `compute_derived()`

Converts the raw `itl_values` list (seconds) into millisecond statistics. Uses `ceil`-based percentile index clamping:


> **Note:** The first entry in `itl_values` is always the TTFT itself (gap from
> `t_start` to first token), which is why `p99_itl_ms` often shows a large
> value — it is dominated by that first gap.

---

### `BurstProbeResult`

Aggregates N individual `ProbeResult` objects from a concurrent burst.

#### `compute()`

Pools TTFT and ITL values from all successful probes and computes cross-request
means, stddevs, and percentiles. Filters out probes where `mean_itl_ms == 0`
(zero-token probes) from the ITL pool. `kv_usage_est` is the mean across all
individual KV estimates — it represents system-wide pressure rather than a
single request's experience.

---

### `ProbeConfig`

All tunable knobs in one dataclass. Key fields:

| Field | Default | Purpose |
|---|---|---|
| `token_budget` | 512 | `max_tokens` for every probe request |
| `baseline_window` | 5 | Number of samples in `calibrate_baseline()` |
| `burst_concurrency` | 4 | Default N for `burst_probe()` |
| `rolling_window` | 20 | History depth for `classify_state()` and `rolling_summary()` |
| `itl_fill_threshold` | 1.5 | ITL × baseline → FILLING state |
| `itl_saturated_threshold` | 3.0 | ITL × baseline → SATURATED state |
| `ttft_hol_threshold` | 5.0 | TTFT × baseline → HOL_BLOCKED state |
| `itl_cv_thrash_threshold` | 0.5 | Rolling CoV(ITL) → THRASHING state |
| `regressor_backend` | `"linear"` | `"lightgbm"`, `"linear"`, or `"none"` |
| `probe_extra_body` | `{}` | Dict merged into every API payload (e.g. Gemini thinking controls) |

---

## `KVUsageEstimator`

Lightweight regressor: ITL feature vector → Û_sys ∈ [0, 1].

From Fill & Squeeze §5.2.2: ITL correlates strongly with global GPU memory
usage due to physical memory bandwidth contention, enabling a black-box KV
cache utilisation estimate without privileged access.

---

### `extract_features(itl_ms)`

**Static method.** Converts a variable-length ITL sequence into a fixed
**11-dimensional feature vector**:


- **`linear_slope`** — fitted via `np.polyfit`; captures whether ITL is
  trending upward within a single request. A rising slope can indicate the
  scheduler is adding new requests to the KV pool while yours is running.
- **`CoV`** (coefficient of variation = std / mean) — measures relative
  variability; the most reliable thrashing signal, since it normalises variance
  against the mean.

---

### `fit(itl_sequences, kv_usages)`

**Supervised training path.** Takes ground-truth `(ITL_sequence, kv_occupancy)`
pairs collected during a white-box calibration phase — i.e. you run probes
while reading the real `vllm:gpu_cache_usage_perc` metric from vLLM's
`/metrics` endpoint, then call `fit()` with those pairs.

Until `fit()` is called, the estimator falls back to the unsupervised sigmoid.

Supported backends:
- `"lightgbm"` — `LGBMRegressor(n_estimators=100, max_depth=4)`
- `"linear"` — `Ridge(alpha=1.0)` with `StandardScaler`

---

### `set_baseline(baseline_itl_ms)`

Sets the denominator for the unsupervised fallback formula. Called
automatically by `calibrate_baseline()`. If never called, `estimate()` returns
`0.0` for everything.

---

### `estimate(itl_ms)`

Primary inference method. Returns Û_sys ∈ [0.0, 1.0].

- **Supervised path:** Extracts the 11-dim feature vector and runs Ridge or
  LightGBM prediction.
- **Unsupervised fallback (soft sigmoid):**


This maps any ITL below baseline to `0.0`, and asymptotically approaches `1.0`
without ever snapping to it:

| mean_ITL | Û_sys |
|---|---|
| ≤ baseline | 0.00 |
| 1.5× baseline | 0.33 |
| 2× baseline | 0.50 |
| 4× baseline | 0.75 |
| → ∞ | → 1.00 |

---

## `ITLProbes` — Infrastructure Methods

These are private methods that underpin all public probes.

---

### `_get_session()`

Lazy `aiohttp.ClientSession` factory. Creates a new session only if none exists
or the existing one is closed. The session is reused across all probes, avoiding
TCP handshake overhead on every request.

---

### `_build_payload(prompt, max_tokens, stream)`

Constructs the full JSON request body for whichever API format is configured.

| Format | Fields set |
|---|---|
| OpenAI / CUSTOM | `messages`, `max_tokens`, optionally `stream_options` |
| Anthropic | `messages`, `max_tokens` |
| Ollama | `messages`, `options.num_predict`, `options.temperature` |

---

### `_extract_token(payload_str)`

Parses a single SSE data line and returns the content delta string, or `None`
for non-content chunks.

| Format | Reads from |
|---|---|
| OpenAI / CUSTOM | `choices[0].delta.content` or `reasoning_content` |
| Anthropic | `delta.text` on `content_block_delta` events only |
| Ollama | `message.content`; returns `None` on `done=true` |

---

### `_stream_probe(prompt, max_tokens, probe_type)`

**The core HTTP measurement primitive.** All public probe methods ultimately
call this.

**Flow:**
1. Opens a streaming POST to the configured endpoint
2. Reads SSE lines from the response byte-by-byte using a `buf` accumulator
   (handles chunked TCP delivery — a newline may arrive split across multiple
   TCP segments)
3. Timestamps each token arrival with `time.perf_counter()`
4. Records TTFT on the first token: `ttft_s = t_first - t_start`
5. Appends each `t_now - t_last` gap to `itl_values`
6. Calls `compute_derived()` before returning

> **Important:** The **first entry** in `itl_values` equals TTFT, because
> `t_last` starts at `t_start`. This means per-request ITL statistics are
> slightly inflated by the prefill wait time. On vLLM, each SSE chunk = 1
> token, so `tokens_out = len(itl_values)` is accurate. On hosted APIs
> (Gemini, OpenAI), multiple tokens are batched per chunk, so `tokens_out`
> underestimates real token count.

---

### `_nonstream_probe(prompt, max_tokens, probe_type)`

Non-streaming variant. Sends a request with `stream=False` and waits for the
complete JSON response. Returns no `itl_values`.

Used only by `ttft_probe(use_streaming=False)`. `total_s ≈ TTFT` for very
short outputs since there is no decode phase to observe.

---

## `ITLProbes` — Public Probe Methods

---

### `calibrate_baseline(n_samples, warm_up, inter_delay_s)`

**Must be called before any other probe is meaningful.**

Fires `warm_up` discarded probes (to flush prefix cache and cold-start effects),
then collects `n_samples` real measurements and stores their means into:
- `config.baseline_itl_ms`
- `config.baseline_ttft_s`
- `estimator._baseline_itl_ms`

Every subsequent probe's KV estimate and state classification is relative to
these values. A higher `n_samples` (8+) produces a more stable baseline,
especially on hosted APIs with variable routing.

---

### `single_probe(prompt, max_tokens)`

**The most-called function in the codebase.**

One streaming request measured end-to-end. After `_stream_probe` returns:
1. Calls `estimator.estimate()` on the ITL values → sets `kv_usage_est`
2. Appends result to `_itl_history` (the rolling window used by
   `classify_state()` and `rolling_summary()`)

All of `monitor()`, `estimate_kv_usage()`, `memory_pressure_scan()`,
`tpot_probe()`, and `adaptive_fill_signal()` are thin wrappers over this.

---

### `burst_probe(n, prompt, max_tokens)`

Fires N requests simultaneously using `asyncio.gather`.

**Key diagnostic signals:**

| Signal | Interpretation |
|---|---|
| High `ttft_stddev_s` | HOL blocking — some requests got a KV slot, others stalled |
| Mean TTFT >> single_probe TTFT | WAITING queue growing |
| High `itl_stddev_ms` across requests | Preemption — scheduler evicting some runners mid-decode |
| All TTFTs near-identical | Gemini/hosted CDN batching — all N go to same backend in same prefill batch |

`kv_usage_est` on the `BurstProbeResult` is computed from the **pooled** ITL
values of all N requests, giving a broader system-wide pressure reading than
any single probe.

> **Rate limiting note:** In `RateLimitedITLProbes`, `burst_probe` pre-acquires
> all N rate-limit tokens atomically via `acquire_many(n)` before spawning
> tasks. This ensures all N HTTP requests fire concurrently rather than being
> serialized by the per-request `acquire()` call in the overridden
> `_stream_probe`.

---

### `ttft_probe(prompt, use_streaming)`

Deliberately minimal probe — `max_tokens=5` using the `minimal` prompt
("Reply with only the single word: ok").

Since generation is 1–5 tokens, almost all elapsed time is prefill + queue
wait:


A rising `ttft_s` trend across sequential `ttft_probe()` calls indicates the
WAITING queue is growing.

Returns `kv_usage_est = -1.0` — a single ITL value (which equals TTFT itself)
is not a meaningful KV pressure signal since it contains no decode-phase
information.

---

### `preemption_probe(duration_s, max_tokens, spike_threshold)`

Long-running probe that watches for sudden mid-stream ITL spikes.

**Mechanism:** Maintains a rolling window of the last 10 ITL values. If any
new ITL exceeds `spike_threshold × local_mean` (default 5×), it records a
spike event with:
- `token_index` — which token triggered the spike
- `itl_s` — the spike duration
- `local_mean_s` — the preceding 10-token average
- `ratio` — how many times above local mean
- `elapsed_s` — wall clock since probe start

**vLLM preemption signature:** When the scheduler invokes `_preempt()` on a
running request, the victim experiences a multi-second silence between tokens
before resuming (swap I/O in v0, full recompute in v1). This manifests as an
ITL spike detectable against the local mean.

> On hosted APIs (Gemini, OpenAI), this always returns
> `preemption_detected: False` — there is no shared preemptible scheduler
> visible from outside.

---

### `hol_probe(n_fill, long_prompt, short_prompt)`

Head-of-Line blocking detection.

**Method (Fill & Squeeze §5.1):**
1. Send `n_fill` long-output requests concurrently to fill KV pressure
2. After a 50ms stagger, send one minimal short probe
3. HOL detected if `short_ttft / baseline_ttft > ttft_hol_threshold`

**Why it works on vLLM:** Once `free_block_queue` is exhausted and
`can_allocate()` returns `False`, the FCFS scheduler freezes the entire
WAITING queue. Even a trivial 5-token request must wait for a fill request
to complete and release KV blocks, causing `short_ttft` to explode.

> On hosted APIs this test returns negative (`ttft_ratio < 1`) because the
> CDN routes fill and short requests to independent backends with no
> shared KV pool. Document as a **known negative result for hosted APIs**.

---

### `memory_pressure_scan(n_probes, interval_s)`

Sequential time-series of `n_probes` single probes spaced `interval_s` apart.

**Use cases:**
- Map ITL rise during Fill phase to identify saturation point
- Detect recovery (ITL decay post-attack)
- Collect `(ITL_sequence, kv_usage)` pairs for offline `estimator.fit()` training
- Validate that an attack is maintaining memory pressure over time

Does **not** append to `_itl_history` (uses `_stream_probe` directly rather
than `single_probe`), so it does not affect `classify_state()` or
`rolling_summary()`.

---

### `estimate_kv_usage()`

One-liner wrapper: fires `single_probe()` and returns `result.kv_usage_est`.

This is the function the Fill & Squeeze control loop calls on every tick:


---

### `tpot_probe(n_probes)`

Runs `n_probes` sequential probes and aggregates their `tpot_ms`
(= `mean_itl_ms`) values.

**Why TPOT matters separately from ITL:**
TPOT isolates decode-phase throughput from queuing effects. TTFT is dominated
by prefill and waiting time; TPOT measures pure per-token decode speed. Under
GPU memory bandwidth saturation from KV cache pressure, TPOT rises because
attention computation becomes increasingly memory-bound as the KV tensors grow
larger.

`degraded = True` when `tpot_ratio > itl_fill_threshold` (default 1.5×).

---

### `request_timeout_probe(n_probes, hard_timeout)`

Fires N probes concurrently with a hard client-side `asyncio.wait_for` timeout.

Classifies each result into three categories:

| Category | Condition |
|---|---|
| `success` | Completed within `hard_timeout`, `total_s > 0.3` |
| `partial` | Completed but suspiciously fast (`total_s < 0.3`) — server returned empty response |
| `failed` | `asyncio.TimeoutError` or any other exception |

`scheduler_overloaded = True` when > 30% of probes fail — a direct indicator
of scheduler collapse under DoS load.

> **Note:** On `RateLimitedITLProbes`, the N inner probes serialize at the
> rate limiter since each calls the overridden `_stream_probe`. This is
> acceptable for abort/timeout testing, but for true concurrency use
> `burst_probe` instead.

---

### `adaptive_fill_signal(c_sat)`

Combined measure-and-decide primitive for the Fill & Squeeze control loop.

**Steps:**
1. Calls `single_probe()` → gets `kv_usage_est`
2. Computes `delta = c_sat - kv_usage_est`
3. Calls `classify_state()` → gets current `InfraState`
4. Returns one of four actions:

| Action | Condition |
|---|---|
| `"fill"` | `delta` is large and state is not HOL/THRASHING |
| `"squeeze"` | `delta < 0.05` (near saturation) or `HOL_BLOCKED` |
| `"backoff"` | `delta < 0` (over-saturated) or `THRASHING`/`DEGRADED` |
| `"wait"` | Baseline not calibrated or probe failed |

---

### `classify_state(probe, burst)`

Priority-ordered decision tree over the probe result and rolling history.
Returns an `InfraState` enum value.

**Decision order (highest priority first):**


The **CoV check** (step 1) is evaluated over the entire `_itl_history` rolling
window, not just the current probe — a single noisy reading does not trigger
THRASHING. Steps 2–5 use the most recent probe's `mean_itl_ms` and `ttft_s`
ratios against the calibrated baseline.

---

### `monitor(interval_s, on_probe, on_state_change, stop_event, max_iterations)`

Continuous background polling loop for real-time infra state tracking.

**Every `interval_s` seconds:**
1. Fires `single_probe()`
2. Calls `classify_state()`
3. Invokes `on_probe(result, state)` on every cycle
4. Invokes `on_state_change(old_state, new_state)` only on transitions

Both callbacks can be sync or async — the monitor checks with
`asyncio.iscoroutine()` and `await`s if needed.

**Clean shutdown:** The inter-cycle sleep uses `asyncio.wait_for(_stop.wait(), timeout=sleep_s)` so the loop wakes immediately when `stop_event.set()` is called rather than waiting out the full interval.

**Typical usage alongside an attack:**

```python
stop = asyncio.Event()
task = asyncio.create_task(probes.monitor(
    interval_s=2.0,
    on_state_change=lambda old, new: dispatch_fill_squeeze(new),
    stop_event=stop,
))
# ... run attack ...
stop.set()
await task
```

---

### `rolling_summary()`

Aggregates the last `rolling_window` successful probes from `_itl_history`.

**Key output fields:**

| Field | Meaning |
|---|---|
| `mean_itl_ms` | Average decode latency across the window |
| `p95_itl_ms` | 95th percentile ITL — tail latency |
| `itl_slope_ms_per_probe` | Linear trend slope — positive = system degrading over time |
| `mean_kv_est` | Average Û_sys across the window |
| `degradation_ratio` | `mean_itl / baseline_itl` — how far above idle you are |
| `current_state` | Result of `classify_state()` on most recent history |

The `itl_slope_ms_per_probe` is computed with `np.polyfit` — a consistently
positive slope across 20 probes is a more reliable indicator of Fill & Squeeze
effectiveness than any single probe reading, since it smooths over the natural
bimodal variance of hosted API routing.

---

## `InfraState` Enum

| State | Value | Meaning |
|---|---|---|
| `UNKNOWN` | `"unknown"` | No baseline or insufficient data |
| `IDLE` | `"idle"` | Near-baseline ITL/TTFT, KV usage low |
| `FILLING` | `"filling"` | KV usage rising, ITL trending up (1.5–3× baseline) |
| `SATURATED` | `"saturated"` | KV near 100%, ITL >> baseline (3×+) |
| `HOL_BLOCKED` | `"hol_blocked"` | TTFT spiked — WAITING queue frozen |
| `THRASHING` | `"thrashing"` | High ITL variance — preemption loop detected |
| `DEGRADED` | `"degraded"` | Generic significant degradation |