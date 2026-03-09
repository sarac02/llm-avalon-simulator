from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def _extract_json_obj(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    if not t:
        return {}
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


class AvalonLLMCaller:
    def __init__(
        self,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.45,
        max_tokens: int = 500,
        timeout: int = 60,
        retries: int = 2,
    ):
        self.model = model or os.getenv("AVALON_MODEL", "api-gpt-oss-120b")
        self.api_base = api_base or os.getenv("AVALON_API_BASE", "https://tritonai-api.ucsd.edu")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("AVALON_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = max(1, retries)
        self._client = None

    @staticmethod
    def _clean(text: str) -> str:
        t = text or ""
        if "</think>" in t:
            t = t.split("</think>")[-1]
        #fallback incase the model doesnt output the </think> tag
        elif "<think>" in t:
            t = re.sub(r"<think>.*", "", t, flags=re.DOTALL)
        return t.strip()

    def generate(self, *, system: str, user: str, max_tokens: int = 300) -> str:
        if OpenAI is None:
            raise RuntimeError("openai package missing. Install with `pip install openai`.")
        if not self.api_key:
            raise RuntimeError("Missing API key. Export OPENAI_API_KEY (or AVALON_API_KEY).")
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        last_exc = None
        for _ in range(self.retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                    timeout=float(self.timeout),
                )
                return self._clean(resp.choices[0].message.content or "")
            except Exception as exc:
                last_exc = exc
        raise last_exc
