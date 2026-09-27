"""
llm_client.py
-------------
One place where LLM calls are configured, so every method in the harness uses
the same model, temperature and token budget (SPEC section 9).

Why this exists alongside llm_api.py: llm_api.call_llm sets no temperature and
a per-call-site max_tokens, and main.py builds async clients inline with three
different token budgets (4096 / 512 / 256). That is fine for the pre-existing
code paths, which this module does not touch, but the four-method comparison
needs one configuration applied uniformly and recorded in the results.

Sampling temperature is NOT sent to Anthropic. Claude Sonnet 5 / Opus 5 and
later reject temperature, top_p and top_k with a 400, and anthropic SDK 1.x has
no such parameter. `temperature` defaults to None (provider default) and is only
forwarded on the OpenAI route when set; the value used is recorded in the
results metadata either way.

Claude 5 models think adaptively by default, so a response can start with a
thinking block: the reply is the concatenation of the text blocks, never
content[0]. A refusal or a reply cut off at max_tokens raises LLMError instead
of returning partial JSON.

Token budget is 16000 (thinking tokens count against it). Raising a cap can
only avoid truncation, never introduce it.

Keys and base URL come from the environment only:

  ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL   (Parley: https://parley.api.mit.edu)
  OPENAI_API_KEY    / OPENAI_BASE_URL      (Parley: .../v1)

Note that model ids on the MIT Parley gateway are Parley's own short names
(e.g. claude-sonnet-5), not vendor ids. A bad id returns a 400 listing every
available model.
"""

from __future__ import annotations

import os
from typing import Optional

# main.py never loaded .env, so `python main.py --auto` required exported env
# vars. The harness loads it, which affects only the new code paths.
from paths import ENV_FILE

try:
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE)
except ImportError:  # pragma: no cover - optional dependency
    pass


class LLMError(RuntimeError):
    """Raised when a call fails for a reason the caller should see."""


class LLMClient:
    """
    Thin synchronous + asynchronous wrapper around one provider.

    Deliberately does not retry. Retry policy differs by method — Method D
    retries on schema violations with the violation stated back to the model
    (SPEC sections 4 and 5), while the three pre-existing methods do not retry
    at all — so it belongs in the methods, not here.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-5",
        temperature: Optional[float] = None,
        max_tokens: int = 16000,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_calls = 0   # every request made through this client, for cost reporting

    # ------------------------------------------------------------------
    @classmethod
    def from_settings(cls, settings) -> "LLMClient":
        """Build from a harness.LLMSettings."""
        return cls(
            provider=settings.provider,
            model=settings.model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    # ------------------------------------------------------------------
    def complete(self, prompt: str) -> str:
        self.n_calls += 1
        if self.provider == "anthropic":
            return self._anthropic_sync(prompt)
        return self._openai_sync(prompt)

    async def acomplete(self, prompt: str) -> str:
        self.n_calls += 1
        if self.provider == "anthropic":
            return await self._anthropic_async(prompt)
        return await self._openai_async(prompt)

    # ------------------------------------------------------------------
    def _anthropic_sync(self, prompt: str) -> str:
        try:
            import anthropic
        except ImportError:
            raise LLMError("pip install anthropic") from None
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise LLMError("Set ANTHROPIC_API_KEY (or put it in .env)")
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return _anthropic_text(msg)

    async def _anthropic_async(self, prompt: str) -> str:
        try:
            import anthropic
        except ImportError:
            raise LLMError("pip install anthropic") from None
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise LLMError("Set ANTHROPIC_API_KEY (or put it in .env)")
        client = anthropic.AsyncAnthropic()
        msg = await client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return _anthropic_text(msg)

    def _openai_sync(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise LLMError("pip install openai") from None
        if not os.getenv("OPENAI_API_KEY"):
            raise LLMError("Set OPENAI_API_KEY (or put it in .env)")
        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **self._openai_sampling(),
        )
        return resp.choices[0].message.content or ""

    async def _openai_async(self, prompt: str) -> str:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise LLMError("pip install openai") from None
        if not os.getenv("OPENAI_API_KEY"):
            raise LLMError("Set OPENAI_API_KEY (or put it in .env)")
        client = AsyncOpenAI()
        resp = await client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **self._openai_sampling(),
        )
        return resp.choices[0].message.content or ""


    def _openai_sampling(self) -> dict:
        return {} if self.temperature is None else {"temperature": self.temperature}


def _anthropic_text(msg) -> str:
    """The reply text of a Messages API response; raises on refusal or truncation."""
    if msg.stop_reason == "refusal":
        details = getattr(msg, "stop_details", None)
        raise LLMError(f"model refused (category: {getattr(details, 'category', None)})")
    if msg.stop_reason == "max_tokens":
        raise LLMError("reply cut off at max_tokens; raise max_tokens")
    text = "".join(b.text for b in msg.content if b.type == "text")
    if not text:
        raise LLMError(f"no text in the reply (stop_reason: {msg.stop_reason})")
    return text


class StubClient(LLMClient):
    """
    Returns canned responses instead of calling an API.

    Used by the identity tests to drive the new adapters and the pre-existing
    code paths with byte-identical LLM output, which is what makes
    "behaviour unchanged" checkable without spending API calls on a
    temperature-1.0 model whose output is not reproducible anyway.
    """

    def __init__(self, responses, **kwargs):
        super().__init__(**kwargs)
        self._responses = list(responses)
        self.prompts: list[str] = []
        self._i = 0

    def _next(self, prompt: str) -> str:
        self.n_calls += 1
        self.prompts.append(prompt)
        if not self._responses:
            raise LLMError("StubClient has no responses configured")
        out = self._responses[self._i % len(self._responses)]
        self._i += 1
        return out(prompt) if callable(out) else out

    def complete(self, prompt: str) -> str:
        return self._next(prompt)

    async def acomplete(self, prompt: str) -> str:
        return self._next(prompt)
