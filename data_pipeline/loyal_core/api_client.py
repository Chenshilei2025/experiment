"""Small async chat client used by reward and evaluation scorers."""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any


class ApiClient:
    """OpenAI-compatible client with the narrow interface this project needs."""

    def __init__(self, model: str, max_concurrent: int = 8, api_key_env: str = "OPENAI_API_KEY",
                 request_timeout_seconds: float = 60.0, base_url_env: str = "OPENAI_BASE_URL",
                 max_concurrent_per_key: int | None = None, max_retries: int = 2) -> None:
        if max_concurrent < 1 or (max_concurrent_per_key is not None and max_concurrent_per_key < 1):
            raise ValueError("concurrency limits must be positive")
        self.model = model
        self.max_concurrent = max_concurrent
        self.max_concurrent_per_key = max_concurrent_per_key or max_concurrent
        self.api_key_env = api_key_env
        self.request_timeout_seconds = request_timeout_seconds
        self.base_url_env = base_url_env
        self.max_retries = max_retries
        self._key_envs = tuple(name.strip() for name in api_key_env.split(",") if name.strip())
        if not self._key_envs:
            raise ValueError("api_key_env must name at least one environment variable")
        self._clients: dict[str, Any] = {}
        self._next_key_index = 0
        self._request_semaphore: asyncio.Semaphore | None = None
        self._key_semaphores: dict[str, asyncio.Semaphore] = {}
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None

    async def __aenter__(self) -> "ApiClient":
        self._request_semaphore = asyncio.Semaphore(self.max_concurrent)
        self._key_semaphores = {name: asyncio.Semaphore(self.max_concurrent_per_key) for name in self._key_envs}
        self._semaphore_loop = asyncio.get_running_loop()
        return self

    async def __aexit__(self, *_: object) -> None:
        await asyncio.gather(*(client.close() for client in self._clients.values()))
        self._clients.clear()

    def _next_key_env(self) -> str:
        key = self._key_envs[self._next_key_index % len(self._key_envs)]
        self._next_key_index += 1
        return key

    def _get_client(self, key_env: str):
        if key_env not in self._clients:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Install the openai package to run scorers.") from exc
            self._clients[key_env] = AsyncOpenAI(
                api_key=os.environ.get(key_env), base_url=os.environ.get(self.base_url_env) or None,
                timeout=self.request_timeout_seconds, max_retries=self.max_retries,
            )
        return self._clients[key_env]

    def _get_request_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._request_semaphore is None or self._semaphore_loop is not loop:
            self._request_semaphore = asyncio.Semaphore(self.max_concurrent)
            self._key_semaphores = {name: asyncio.Semaphore(self.max_concurrent_per_key) for name in self._key_envs}
            self._semaphore_loop = loop
        return self._request_semaphore

    def _get_key_semaphore(self, key_env: str) -> asyncio.Semaphore:
        self._get_request_semaphore()
        return self._key_semaphores[key_env]

    async def _create_completion(self, request: dict[str, Any]) -> str:
        key_env = self._next_key_env()
        async with self._get_request_semaphore(), self._get_key_semaphore(key_env):
            response = await asyncio.wait_for(
                self._get_client(key_env).chat.completions.create(**request), timeout=self.request_timeout_seconds,
            )
        return response.choices[0].message.content or ""

    async def chat_text(self, messages: list[dict[str, str]], temperature: float, max_tokens: int,
                        seed: int | None = None, extra_body: dict[str, Any] | None = None) -> str:
        request: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if seed is not None:
            request["seed"] = seed
        if extra_body is not None:
            request["extra_body"] = extra_body
        return await self._create_completion(request)

    async def chat_json(self, messages: list[dict[str, str]], temperature: float, max_tokens: int,
                       seed: int | None = None, extra_body: dict[str, Any] | None = None) -> str:
        request: dict[str, Any] = {
            "model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if seed is not None:
            request["seed"] = seed
        if extra_body is not None:
            request["extra_body"] = extra_body
        text = await self._create_completion(request)
        json.loads(text)
        return text
