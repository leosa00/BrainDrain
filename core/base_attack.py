"""
base_attack.py — Abstract base class for all LLM DoS attack modules.
"""

from __future__ import annotations

import asyncio
import socket
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Optional

import aiohttp


# ─────────────────────────────────────────────
# Keepalive connector
# ─────────────────────────────────────────────

# Aggressive TCP keepalive: probe an idle connection after 30s, then every 10s,
# and give up after 3 unanswered probes (~60s to detect a dead peer).  This is a
# transport-layer liveness check with no assumptions about the serving stack —
# it surfaces connections whose backend has gone away (process restart, reset,
# network drop) even while the HTTP stream is silent: the socket errors, the
# streaming read raises, and the dispatcher recycles the slot.  A connection
# that is merely idle but alive (request queued or preempted on a healthy
# server) keeps answering probes and is left untouched, so KV-heavy requests are
# never dropped.  How far the probe reaches depends on the deployment (an L4 /
# passthrough LB reaches the backend; an L7 proxy only proves the proxy is up).
_KEEPALIVE_OPTS = {
    "TCP_KEEPIDLE":  30,   # seconds idle before the first probe (Linux)
    "TCP_KEEPINTVL": 10,   # seconds between probes
    "TCP_KEEPCNT":   3,    # failed probes before the socket is declared dead
}


class KeepAliveTCPConnector(aiohttp.TCPConnector):
    """TCPConnector that enables aggressive TCP keepalive on every socket."""

    async def _wrap_create_connection(self, *args, **kwargs):
        transport, protocol = await super()._wrap_create_connection(*args, **kwargs)
        sock = transport.get_extra_info("socket")
        if sock is not None:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                for name, value in _KEEPALIVE_OPTS.items():
                    opt = getattr(socket, name, None)  # platform-dependent
                    if opt is not None:
                        sock.setsockopt(socket.IPPROTO_TCP, opt, value)
            except (OSError, AttributeError):
                pass
        return transport, protocol


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────
CUSTOM_PATH = "/v1/chat/completions"

class AttackStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    SUCCESS   = "success"
    FAILED    = "failed"
    TIMEOUT   = "timeout"
    CANCELLED = "cancelled"


class APIFormat(Enum):
    OPENAI    = "openai"      # /v1/chat/completions
    ANTHROPIC = "anthropic"   # /v1/messages
    OLLAMA    = "ollama"      # /api/chat
    CUSTOM    = "custom"
    TEST      = "test"        # user-defined endpoint
    VERTEX    = "vertex"      # Vertex AI — base_url must include the full model path


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class TargetConfig:
    base_url:             str
    model:                str
    api_format:           APIFormat = APIFormat.OPENAI
    api_key:              Optional[str] = None
    # endpoint_path controls the URL used for requests:
    #   None  → use the format default (e.g. /v1/chat/completions)
    #   ""    → use base_url exactly as-is
    #   other → append to base_url
    endpoint_path:        Optional[str] = None
    timeout:              Optional[float] = None
    verify_ssl:           bool  = True
    extra_headers:        dict[str, str] = field(default_factory=dict)
    supports_stream_options: bool = True
    system_prompt:        Optional[str] = None
    request_prefix:       Optional[str] = None
    # If set, used as the base request body instead of the built-in payload
    # builders. The string "__PROMPT__" anywhere in the template is replaced
    # with the attack prompt at runtime.
    request_template:     Optional[dict] = None

    @property
    def endpoint(self) -> str:
        if self.endpoint_path is not None:
            if self.endpoint_path:
                return f"{self.base_url.rstrip('/')}/{self.endpoint_path.lstrip('/')}"
            return self.base_url  # empty string → use base_url exactly
        if self.api_format == APIFormat.VERTEX:
            return f"{self.base_url.rstrip('/')}:streamGenerateContent?alt=sse"
        defaults = {
            APIFormat.OPENAI:    "/v1/chat/completions",
            APIFormat.ANTHROPIC: "/v1/messages",
            APIFormat.OLLAMA:    "/api/chat",
            APIFormat.CUSTOM:    CUSTOM_PATH,
            APIFormat.TEST:      "/v1/chat/completions",
        }
        return f"{self.base_url.rstrip('/')}{defaults[self.api_format]}"

    @property
    def nonstreaming_endpoint(self) -> str:
        """Endpoint for non-streaming requests (used by ttft_probe)."""
        if self.api_format == APIFormat.VERTEX:
            return f"{self.base_url.rstrip('/')}:generateContent"
        return self.endpoint

    @property
    def auth_headers(self) -> dict[str, str]:
        headers = {}
        if self.api_key:
            if self.api_format == APIFormat.ANTHROPIC:
                headers["x-api-key"] = self.api_key
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"
        return {**headers, **self.extra_headers}


@dataclass
class AttackConfig:
    """Common parameters shared across all attack types."""
    target: TargetConfig
    max_tokens: int     = 8192            # requested max_tokens / max_completion_tokens
    temperature: float  = 0.0             # determinism for reproducibility
    stream: bool        = True            # enables ITL measurement
    # Seconds to sleep after reading each streaming chunk.
    # Slows client-side consumption, keeping the HTTP connection open longer
    # and backing up the server's output queue — extends effective KV block
    # occupancy beyond pure generation time on servers that flow-control sends.
    stream_read_delay_s: float = 0.0
    request_id: str     = field(default_factory=lambda: str(uuid.uuid4()))
    tag: Optional[str]  = None            # label for reporting (e.g. "reasoning_bomb_v1")
    metadata: dict      = field(default_factory=dict)


# ─────────────────────────────────────────────
# Result / Metrics
# ─────────────────────────────────────────────

@dataclass
class TokenMetrics:
    prompt_tokens: int      = 0
    completion_tokens: int  = 0
    reasoning_tokens: int   = 0           # populated when model exposes thinking tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def amplification_ratio(self) -> float:
        """Completion / prompt ratio — key DoS efficiency metric."""
        if self.prompt_tokens == 0:
            return 0.0
        return self.completion_tokens / self.prompt_tokens


@dataclass
class LatencyMetrics:
    ttft_s: float               = 0.0    # Time To First Output Token (seconds)
    ttfrt_s: float              = 0.0    # Time To First Reasoning Token (seconds)
    total_duration_s: float     = 0.0    # Wall-clock time for full response
    inter_token_latencies: list[float] = field(default_factory=list)

    @property
    def mean_itl_ms(self) -> float:
        if not self.inter_token_latencies:
            return 0.0
        return (sum(self.inter_token_latencies) / len(self.inter_token_latencies)) * 1000

    @property
    def tokens_per_second(self) -> float:
        if self.total_duration_s == 0:
            return 0.0
        n = len(self.inter_token_latencies) + 1
        return n / self.total_duration_s


@dataclass
class AttackResult:
    request_id: str
    attack_name: str
    status: AttackStatus
    token_metrics: TokenMetrics         = field(default_factory=TokenMetrics)
    latency_metrics: LatencyMetrics     = field(default_factory=LatencyMetrics)
    raw_response: Optional[str]         = None
    error: Optional[str]                = None
    metadata: dict[str, Any]            = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id":           self.request_id,
            "attack_name":          self.attack_name,
            "status":               self.status.value,
            "prompt_tokens":        self.token_metrics.prompt_tokens,
            "completion_tokens":    self.token_metrics.completion_tokens,
            "reasoning_tokens":     self.token_metrics.reasoning_tokens,
            "amplification_ratio":  round(self.token_metrics.amplification_ratio, 4),
            "ttft_s":               round(self.latency_metrics.ttft_s, 4),
            "ttfrt_s":              round(self.latency_metrics.ttfrt_s, 4),
            "total_duration_s":     round(self.latency_metrics.total_duration_s, 4),
            "mean_itl_ms":          round(self.latency_metrics.mean_itl_ms, 4),
            "tokens_per_second":    round(self.latency_metrics.tokens_per_second, 4),
            "error":                self.error,
            "metadata":             self.metadata,
        }


# ─────────────────────────────────────────────
# Abstract Base Attack
# ─────────────────────────────────────────────

class BaseAttack(ABC):
    """
    Abstract base for all DoS attack modules.

    Subclasses must implement:
      - attack_name (property)
      - build_payload(config) -> dict
      - parse_stream_chunk(chunk, result) -> Optional[str]   [if stream=True]
      - parse_full_response(data, result)                    [if stream=False]
    """

    def __init__(self, config: AttackConfig) -> None:
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._current_result: Optional[AttackResult] = None
        # Lifecycle phase of the in-flight request, for live observability.
        # One of: idle, connecting, awaiting_ttft, streaming, done, failed.
        #   awaiting_ttft → connection accepted by the server but no token has
        #   been generated yet.  A request stuck here is "dispatched but not
        #   actually running" — the server has not admitted it to its running
        #   batch (e.g. KV cache saturated).
        self.phase: str = "idle"
        # perf_counter timestamp of the most recent token (content OR reasoning).
        # Lets an external observer detect a request that started streaming but
        # has since gone silent — i.e. the server preempted/swapped it out
        # (a "stalled" request, counted in neither running nor waiting).
        self._last_token_t: Optional[float] = None

    # ── Identity ─────────────────────────────

    @property
    @abstractmethod
    def attack_name(self) -> str:
        """Short identifier, e.g. 'reasoning_bomb'."""
        ...

    # ── Payload Construction ─────────────────

    @abstractmethod
    def build_payload(self) -> dict[str, Any]:
        """
        Construct the full JSON body for the target API.
        Uses self.config for parameters.
        """
        ...

    # ── Stream / Response Parsing ─────────────

    @abstractmethod
    def parse_stream_chunk(
        self, raw_chunk: str, result: AttackResult
    ) -> Optional[str]:
        """
        Parse one SSE data line from a streaming response.
        Returns the decoded text delta, or None if not a content chunk.
        Subclass is responsible for updating result.token_metrics
        from usage fields when available (final chunk).
        """
        ...

    @abstractmethod
    def parse_full_response(
        self, data: dict[str, Any], result: AttackResult
    ) -> None:
        """
        Parse a non-streaming JSON response body.
        Populate result.token_metrics and result.raw_response.
        """
        ...

    # ── HTTP Session ─────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = KeepAliveTCPConnector(ssl=self.config.target.verify_ssl)
            # sock_connect=10: fail fast if the server won't accept a new
            # connection (saturated / queue full).
            # No read timeout is set on purpose: a long but alive response (e.g.
            # a request queued or preempted on a healthy server) must not be
            # killed.  A genuinely dead connection is detected by TCP keepalive
            # (see KeepAliveTCPConnector), which can't false-positive on a peer
            # that is still answering.
            timeout = aiohttp.ClientTimeout(
                total=self.config.target.timeout,
                sock_connect=10,
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Core Execution ────────────────────────

    async def run(self) -> AttackResult:
        """
        Execute a single attack request, measure all latency metrics,
        and return a populated AttackResult.
        """
        result = AttackResult(
            request_id=self.config.request_id,
            attack_name=self.attack_name,
            status=AttackStatus.RUNNING,
        )
        result.metadata = self.config.metadata
        self._current_result = result

        payload = self.build_payload()
        session = await self._get_session()

        t_start = time.perf_counter()
        self.phase = "connecting"

        try:
            async with session.post(
                self.config.target.endpoint,
                json=payload,
                headers=self.config.target.auth_headers,
            ) as response:

                if response.status >= 400:
                    body = await response.text()
                    result.status = AttackStatus.FAILED
                    result.error  = f"HTTP {response.status}: {body[:512]}"
                    self.phase = "failed"
                    return result

                # Response headers received: the server accepted the request.
                # For a streaming request we are now waiting for the first
                # generated token — if the server's KV cache is saturated the
                # request parks here, admitted on the wire but not generating.
                self.phase = "awaiting_ttft"

                if self.config.stream:
                    result = await self._consume_stream(
                        response, result, t_start
                    )
                else:
                    data = await response.json()
                    t_end = time.perf_counter()
                    self.parse_full_response(data, result)
                    result.latency_metrics.total_duration_s = t_end - t_start
                    result.status = AttackStatus.SUCCESS

            self.phase = "done"

        except asyncio.TimeoutError:
            self.phase = "done"
            result.status = AttackStatus.TIMEOUT
            t = self.config.target.timeout
            result.error  = (
                f"Request timed out after {t}s"
                if t is not None
                else "Request timed out (read/connect timeout)"
            )
            result.latency_metrics.total_duration_s = time.perf_counter() - t_start

        except aiohttp.ClientError as exc:
            self.phase = "failed"
            result.status = AttackStatus.FAILED
            result.error  = str(exc)

        return result

    async def _consume_stream(
        self,
        response: aiohttp.ClientResponse,
        result: AttackResult,
        t_start: float,
    ) -> AttackResult:
        """Read SSE stream, record TTFT and per-token ITL.

        Uses try/finally so partial token counts and latency data are always
        saved — even when the stream is cut short by a timeout or cancellation.
        This means TIMEOUT results carry the tokens received up to the cut-off.
        """
        t_last        = t_start
        t_first_token = None
        full_text     = []

        try:
            async for raw_line in response.content:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue

                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break

                t_now = time.perf_counter()

                result.metadata["_t_now"]   = t_now
                result.metadata["_t_start"] = t_start

                rt_before = result.metadata.get("_streaming_reasoning_tokens", 0)
                delta = self.parse_stream_chunk(data_str, result)
                rt_after = result.metadata.get("_streaming_reasoning_tokens", 0)

                # A token of either kind (content delta OR reasoning token).
                # Reasoning models (DeepSeek R1, QwQ) stream reasoning_content,
                # which parse_stream_chunk counts but returns as None; keying
                # only off `delta` would read 20k-token reasoning as idle.
                if delta is not None or rt_after > rt_before:
                    self._last_token_t = t_now
                    # Flip on first token, and recover from "stalled" if the
                    # server resumed a previously-preempted request.
                    self.phase = "streaming"

                    # Count EVERY generated token — reasoning or content — toward
                    # TTFT and inter-token latency.  A reasoning bomb may emit its
                    # entire output as reasoning_content (delta is None) and hit
                    # max_tokens before any answer text, so gating these on
                    # `delta` would leave ttft/itl/tps at zero for the runs we
                    # care about most.
                    if t_first_token is None:
                        t_first_token = t_now
                    result.latency_metrics.inter_token_latencies.append(t_now - t_last)
                    t_last = t_now

                if delta is not None:
                    # raw_response holds answer text only — reasoning is excluded.
                    full_text.append(delta)

                if self.config.stream_read_delay_s > 0:
                    await asyncio.sleep(self.config.stream_read_delay_s)

        finally:
            # Always save latency + partial text regardless of how the loop ended.
            result.latency_metrics.total_duration_s = time.perf_counter() - t_start
            if t_first_token is not None:
                result.latency_metrics.ttft_s = t_first_token - t_start
            result.raw_response = "".join(full_text)

            # Completion token fallback: if no usage chunk arrived (timeout /
            # truncated stream), use the number of ITL entries we collected.
            if result.token_metrics.completion_tokens == 0:
                result.token_metrics.completion_tokens = len(
                    result.latency_metrics.inter_token_latencies
                )

            # Reasoning token fallback: parse_stream_chunk implementations
            # increment _streaming_reasoning_tokens for each reasoning-content
            # chunk.  Use that count when the final usage chunk never arrived.
            streaming_rt = result.metadata.pop("_streaming_reasoning_tokens", 0)
            if result.token_metrics.reasoning_tokens == 0 and streaming_rt > 0:
                result.token_metrics.reasoning_tokens = streaming_rt

        result.status = AttackStatus.SUCCESS
        return result

    # ── Convenience ───────────────────────────

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"target={self.config.target.base_url} "
            f"model={self.config.target.model}>"
        )
