"""LLM 协议。

只依赖两个方法：一次性返回与流式返回。
这样接 DeepSeek、Qwen 或本地 vLLM 都只是换一个实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal, Protocol, Sequence, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass(slots=True)
class Message:
    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@runtime_checkable
class LLM(Protocol):
    def complete(
        self,
        messages: Sequence[Message],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str: ...

    def stream(
        self,
        messages: Sequence[Message],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Iterator[str]: ...
