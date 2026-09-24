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

Temperature is pinned explicitly to the Anthropic default of 1.0 rather than
left unset. The request is behaviourally identical to the pre-existing calls,
but the value can now go into the cache key and the results metadata instead
of being an implicit server-side default that could change under us.

Token budget is unified upward to 4096, the largest the pre-existing code
used. Raising a cap can only avoid truncation, never introduce it, so the
existing methods' outputs are unaffected.

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
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
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
        temperature: float = 1.0,
        max_tokens: int = 4096,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

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
        if self.provider == "anthropic":
            return self._anthropic_sync(prompt)
        return self._openai_sync(prompt)

    async def acomplete(self, prompt: str) -> str:
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
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

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
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

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
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
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
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


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
