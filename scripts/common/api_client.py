"""Minimal async OpenAI-compatible chat-completions client."""
from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import subprocess
from dataclasses import dataclass, field
from itertools import cycle
from math import ceil
from typing import Any, Protocol


def _env(prefix: str, name: str, fallback: str | None = None) -> str | None:
    return os.getenv(f"{prefix}_{name}") or (os.getenv(fallback) if fallback else None)


class ChatClient(Protocol):
    """The chat interface shared by a direct client and a keyed client pool."""

    model: str

    async def chat_json(
        self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int,
        seed: int | None = None,
    ) -> str: ...


@dataclass
class ApiClient:
    """A bounded client for a single OpenAI-compatible chat endpoint."""

    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 120.0
    max_concurrent: int = 4
    max_retries: int = 3
    retry_initial_backoff_seconds: float = 1.0
    json_mode: bool = False
    disable_thinking: bool = False
    connect_to: str | None = None
    _semaphore: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if self.base_url.endswith("/chat/completions"):
            self.base_url = self.base_url[: -len("/chat/completions")]
        if not self.base_url or not self.model:
            raise ValueError("base_url and model are required")
        if self.max_concurrent < 1 or self.max_retries < 1 or self.retry_initial_backoff_seconds < 0:
            raise ValueError("max_concurrent and max_retries must be positive; retry backoff cannot be negative")
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

    @classmethod
    def from_env(cls, prefix: str, *, fallback_prefix: str | None = None) -> "ApiClient":
        """Read ``<PREFIX>_{BASE_URL,MODEL,API_KEY}`` without logging secrets."""
        fallback = fallback_prefix or prefix
        base_url = _env(prefix, "BASE_URL", f"{fallback}_BASE_URL")
        model = _env(prefix, "MODEL", f"{fallback}_MODEL")
        api_key = _env(prefix, "API_KEY", f"{fallback}_API_KEY")
        if not base_url or not model:
            raise RuntimeError(
                f"Set {prefix}_BASE_URL and {prefix}_MODEL for the OpenAI-compatible scorer service."
            )
        return cls(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_seconds=float(os.getenv(f"{prefix}_TIMEOUT_SECONDS", "120")),
            max_concurrent=int(os.getenv(f"{prefix}_MAX_CONCURRENT", "4")),
            max_retries=int(os.getenv(f"{prefix}_MAX_RETRIES", "3")),
            retry_initial_backoff_seconds=float(os.getenv(f"{prefix}_RETRY_INITIAL_BACKOFF_SECONDS", "1")),
            json_mode=os.getenv(f"{prefix}_JSON_MODE", "0").lower() in {"1", "true", "yes"},
            disable_thinking=os.getenv(f"{prefix}_DISABLE_THINKING", "0").lower() in {"1", "true", "yes"},
            connect_to=os.getenv(f"{prefix}_CONNECT_TO") or (os.getenv(f"{fallback}_CONNECT_TO") if fallback else None),
        )

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> str:
        """Call chat completions and return the assistant content as text."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.json_mode:
            # Flash models otherwise occasionally return an empty or truncated
            # prose completion even when the prompt asks for JSON.
            payload["response_format"] = {"type": "json_object"}
        if self.disable_thinking:
            # DeepSeek Flash emits hidden reasoning separately.  For compact,
            # deterministic JSON judges it can consume the output budget and
            # leave the final ``content`` field empty.
            payload["thinking"] = {"type": "disabled"}
        if seed is not None:
            payload["seed"] = seed
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                async with self._semaphore:
                    body = await asyncio.to_thread(
                        _post_json, f"{self.base_url}/chat/completions", payload, headers, self.timeout_seconds, self.connect_to
                    )
                data = json.loads(body)
                content = data["choices"][0]["message"].get("content")
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("chat completion returned empty assistant content")
                if self.json_mode:
                    json.loads(content)
                return content
            except Exception as exc:
                error = exc
                if attempt + 1 == self.max_retries:
                    raise
                # Jitter prevents many failed judge calls from retrying in lockstep.
                delay = self.retry_initial_backoff_seconds * (2**attempt)
                await asyncio.sleep(delay * (0.5 + random.random()))
        assert error is not None
        raise error


@dataclass
class ApiClientPool:
    """Bound and balance a collection of credentials as one judge service.

    ``max_concurrent`` is deliberately a *pool-wide* request budget, rather
    than a limit applied once per credential.  Otherwise adding API keys
    silently multiplies load on the provider and turns retries into a burst.
    """

    clients: list[ApiClient]
    max_concurrent: int | None = None
    model: str = field(init=False)
    _clients: Any = field(init=False, repr=False)
    _semaphore: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("at least one API client is required")
        if self.max_concurrent is None:
            # Preserve the direct-construction behaviour: one client has its
            # existing capacity; several clients sum their capacities.
            self.max_concurrent = sum(client.max_concurrent for client in self.clients)
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be at least one")
        self.model = self.clients[0].model
        self._clients = cycle(self.clients)
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

    @classmethod
    def from_env(cls, prefix: str, *, fallback_prefix: str | None = None) -> "ApiClientPool":
        """Build a shared judge client from role settings and their fallback."""
        fallbacks = [fallback_prefix] if fallback_prefix and fallback_prefix != prefix else []
        if "LOYAL_JUDGE" not in fallbacks and prefix != "LOYAL_JUDGE":
            fallbacks.append("LOYAL_JUDGE")
        base_url = _env(prefix, "BASE_URL")
        model = _env(prefix, "MODEL")
        for fallback in fallbacks:
            base_url = base_url or _env(fallback, "BASE_URL")
            model = model or _env(fallback, "MODEL")
        if not base_url or not model:
            raise RuntimeError(
                f"Set {prefix}_BASE_URL and {prefix}_MODEL for the OpenAI-compatible scorer service."
            )

        # A comma-separated pool spreads independent high-volume judge calls
        # over multiple provider credentials.  API_KEY remains supported for
        # single-key deployments and for all existing EIL configurations.
        api_keys_value = _env(prefix, "API_KEYS")
        for fallback in fallbacks:
            api_keys_value = api_keys_value or _env(fallback, "API_KEYS")
        if api_keys_value:
            api_keys = [key.strip() for key in api_keys_value.split(",") if key.strip()]
        else:
            api_key = _env(prefix, "API_KEY")
            for fallback in fallbacks:
                api_key = api_key or _env(fallback, "API_KEY")
            api_keys = [api_key]
        if not api_keys:
            api_keys = [None]
        def setting(name: str, default: str) -> str:
            for candidate in (prefix, *fallbacks):
                value = os.getenv(f"{candidate}_{name}")
                if value:
                    return value
            return default

        total_concurrent = int(setting("MAX_CONCURRENT", "4"))
        if total_concurrent < 1:
            raise ValueError(f"{prefix}_MAX_CONCURRENT must be at least one")
        # The pool semaphore is the authoritative provider-wide cap.  This
        # smaller per-key cap keeps round-robin allocation balanced while the
        # outer cap also covers retries and their backoff period.
        per_key_concurrent = max(1, ceil(total_concurrent / len(api_keys)))
        return cls([
            ApiClient(
                base_url=base_url,
                model=model,
                api_key=api_key,
                timeout_seconds=float(setting("TIMEOUT_SECONDS", "120")),
                max_concurrent=per_key_concurrent,
                max_retries=int(setting("MAX_RETRIES", "3")),
                retry_initial_backoff_seconds=float(setting("RETRY_INITIAL_BACKOFF_SECONDS", "1")),
                json_mode=setting("JSON_MODE", "0").lower() in {"1", "true", "yes"},
                disable_thinking=setting("DISABLE_THINKING", "0").lower() in {"1", "true", "yes"},
                connect_to=setting("CONNECT_TO", "") or None,
            )
            for api_key in api_keys
        ], max_concurrent=total_concurrent)

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> str:
        # Acquire before selecting a credential so that failed logical calls
        # (including their retries) remain within the one service budget.
        async with self._semaphore:
            return await next(self._clients).chat_json(
                messages, temperature=temperature, max_tokens=max_tokens, seed=seed,
            )


def _post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout_seconds: float, connect_to: str | None = None,
) -> str:
    """POST through curl's HTTP/2 client, bypassing any locally configured proxy."""
    if not shutil.which("curl"):
        raise RuntimeError("curl is required for judge API requests")

    # Pass credentials through curl's stdin config rather than the process list.
    config = [
        f"url = {json.dumps(url)}",
        'request = "POST"',
        'noproxy = "*"',
        "http2",
        "silent",
        "show-error",
        *[f"header = {json.dumps(f'{name}: {value}')}" for name, value in headers.items()],
        f"data = {json.dumps(json.dumps(payload, separators=(',', ':')))}",
        'write-out = "\\n__LOYAL_HTTP_STATUS__:%{http_code}"',
    ]
    try:
        command = ["curl", "--config", "-", "--connect-timeout", "15", "--max-time", str(timeout_seconds)]
        if connect_to:
            command.extend(["--connect-to", connect_to])
        result = subprocess.run(
            command,
            input="\n".join(config) + "\n",
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"could not execute curl for chat completion: {exc}") from exc

    body, marker, status = result.stdout.rpartition("\n__LOYAL_HTTP_STATUS__:")
    if result.returncode or not marker or not status.strip().isdigit() or int(status) >= 400:
        detail = body if marker else result.stdout
        detail = detail.strip() or result.stderr.strip()
        http_status = status.strip() if marker else "unknown"
        raise RuntimeError(f"chat completion HTTP {http_status}: {detail[:500]}")
    return body
