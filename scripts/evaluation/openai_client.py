"""OpenAI judge LM — calls the OpenAI chat completions API directly via OPENAI_API_KEY."""
from __future__ import annotations
import os
import dspy


class _OAMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls = None


class _OAChoice:
    def __init__(self, content: str) -> None:
        self.message = _OAMessage(content)
        self.logprobs = None


class _OAUsage:
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens

    def __iter__(self):
        yield "prompt_tokens", self.prompt_tokens
        yield "completion_tokens", self.completion_tokens
        yield "total_tokens", self.total_tokens


class _OAResponse:
    def __init__(self, text: str, usage: _OAUsage, model: str) -> None:
        self.choices = [_OAChoice(text)]
        self.usage = usage
        self.model = model


class OpenAIJudgeLM(dspy.BaseLM):
    """dspy.BaseLM subclass that calls the OpenAI API directly using OPENAI_API_KEY."""

    def __init__(self, model: str, reasoning_effort: str | None = None) -> None:
        super().__init__(model=model, cache=False)
        import openai
        self.api_key = os.getenv("OPENAI_API_KEY")
        self._client = openai.OpenAI(api_key=self.api_key)
        self.reasoning_effort = reasoning_effort

    def forward(self, prompt: str | None = None, messages: list | None = None, **kwargs) -> _OAResponse:
        if prompt and not messages:
            messages = [{"role": "user", "content": prompt}]

        extra: dict = {}
        if self.reasoning_effort:
            extra["reasoning_effort"] = self.reasoning_effort

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages or [],
            **extra,
        )

        text = resp.choices[0].message.content or "" if resp.choices else ""
        u = resp.usage
        usage = _OAUsage(
            prompt_tokens=u.prompt_tokens if u else 0,
            completion_tokens=u.completion_tokens if u else 0,
        )
        return _OAResponse(text=text, usage=usage, model=self.model)
