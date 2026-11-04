"""OpenAI 兼容的 LLM 实现。

DeepSeek 与阿里云百炼（DashScope）都提供 OpenAI 兼容端点，
所以一个实现就够，切换只改 base_url / model / 环境变量。

    # DeepSeek 官方
    export DEEPSEEK_API_KEY=sk-xxx
    # 阿里云百炼（Qwen 系列）
    export DASHSCOPE_API_KEY=sk-xxx
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from ragkb.generation.base import Message


@dataclass(frozen=True, slots=True)
class ProviderPreset:
    base_url: str
    model: str
    env_key: str
    label: str


PRESETS: dict[str, ProviderPreset] = {
    "deepseek": ProviderPreset(
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        env_key="DEEPSEEK_API_KEY",
        label="DeepSeek",
    ),
    "dashscope": ProviderPreset(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",
        env_key="DASHSCOPE_API_KEY",
        label="阿里云百炼（Qwen）",
    ),
    # 国内可直连 + 9B 以下模型永久免费，适合放公开 demo
    "siliconflow": ProviderPreset(
        base_url="https://api.siliconflow.cn/v1",
        model="Qwen/Qwen3-8B",
        env_key="SILICONFLOW_API_KEY",
        label="硅基流动（SiliconFlow）",
    ),
}


def available_providers() -> list[str]:
    return [name for name, preset in PRESETS.items() if os.environ.get(preset.env_key)]


class OpenAICompatibleLLM:
    def __init__(
        self,
        provider: str = "deepseek",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise RuntimeError("需要 openai SDK：pip install openai") from exc

        preset = PRESETS.get(provider)
        if preset is None:
            raise ValueError(f"未知的 provider: {provider!r}，可选 {sorted(PRESETS)}")

        key = api_key or os.environ.get(preset.env_key)
        if not key:
            raise RuntimeError(
                f"未找到 {preset.env_key}。请设置环境变量，或在测试中使用 MockLLM。"
            )

        self._client = OpenAI(
            api_key=key, base_url=base_url or preset.base_url, timeout=timeout
        )
        self._model = model or preset.model
        self._provider = provider
        # Qwen3 这类模型默认先"想"一段再答，实测 Qwen3-8B 单轮从 6.3s 降到 1.0s。
        # 只在显式打开时才传，别的供应商不认这个字段，乱传会 400。
        self._extra_body: dict[str, Any] = {}
        if os.environ.get("KB_DISABLE_THINKING", "").strip().lower() in {"1", "true", "yes"}:
            self._extra_body["enable_thinking"] = False

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return self._provider

    def _create(self, messages: Sequence[Message], temperature: float, max_tokens: int, stream: bool):
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        return self._client.chat.completions.create(**kwargs)

    def complete(
        self,
        messages: Sequence[Message],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        response = self._create(messages, temperature, max_tokens, stream=False)
        return response.choices[0].message.content or ""

    def stream(
        self,
        messages: Sequence[Message],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        response = self._create(messages, temperature, max_tokens, stream=True)
        for chunk in response:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta


def build_llm(provider: str = "auto", **kwargs):
    """auto 模式按环境变量探测，优先 DeepSeek，其次百炼。

    都没有时返回 None，调用方应回退到 MockLLM。
    """
    if provider == "auto":
        found = available_providers()
        if not found:
            return None
        provider = "deepseek" if "deepseek" in found else found[0]

    if provider in {"mock", "none"}:
        return None

    return OpenAICompatibleLLM(provider=provider, **kwargs)
