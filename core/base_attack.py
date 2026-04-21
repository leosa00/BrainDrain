"""
base_attack.py — Abstract base class for all LLM DoS attack modules.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Optional

import aiohttp


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
        return self.prompt_tokens + self.completion_tokens + self.reasoning_tokens

    @property
    def amplification_ratio(self) -> float:
        """Completion / prompt ratio — key DoS efficiency metric."""
        if self.prompt_tokens == 0:
            return 0.0
        return (self.completion_tokens + self.reasoning_tokens) / self.prompt_tokens


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
            connector = aiohttp.TCPConnector(ssl=self.config.target.verify_ssl)
            timeout   = aiohttp.ClientTimeout(total=self.config.target.timeout)  # None = no timeout
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
        t_first_token: Optional[float] = None

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
                    return result

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

        except asyncio.TimeoutError:
            result.status = AttackStatus.TIMEOUT
            result.error  = f"Request timed out after {self.config.target.timeout}s"
            result.latency_metrics.total_duration_s = time.perf_counter() - t_start

        except aiohttp.ClientError as exc:
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

                delta = self.parse_stream_chunk(data_str, result)

                if delta is not None:
                    if t_first_token is None:
                        t_first_token = t_now
                    result.latency_metrics.inter_token_latencies.append(t_now - t_last)
                    full_text.append(delta)
                    t_last = t_now

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
