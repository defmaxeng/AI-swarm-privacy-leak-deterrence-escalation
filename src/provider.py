"""Provider layer: one OpenAI SDK client per provider, chosen per model from config/models.yaml.

Every call is appended to logs/calls.jsonl with provider, exact model id, usage and cost.
A global spend ledger (logs/spend.json) enforces the budget hard stop.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import yaml
from openai import OpenAI, APIStatusError, APIConnectionError, APITimeoutError, RateLimitError

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(ROOT / "config" / "models.yaml", encoding="utf-8"))
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
CALLS = LOG_DIR / "calls.jsonl"
SPEND = LOG_DIR / "spend.json"
SWITCHES = LOG_DIR / "provider_switches.jsonl"
ADAPTION_SPEND = LOG_DIR / "adaption_spend.json"  # list-price estimate; paid from Adaption credits, not the $25 cap

_lock = threading.Lock()
_clients: dict = {}


class BudgetExceeded(RuntimeError):
    pass


class _Retryable(RuntimeError):
    pass


class _AdaptionChat:
    """Exposes `.chat.completions.create(**chat_body)` on top of Adaption's Responses API, so callers
    can keep using chat-completions messages and tools. Returns a chat-completion-shaped object."""

    def __init__(self, key: str, base_url: str | None):
        from adaption import Adaption
        self._c = Adaption(api_key=key, **({"base_url": base_url} if base_url else {}), max_retries=0, timeout=300)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    @staticmethod
    def _text(content) -> str:
        if isinstance(content, list):
            return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
        return content or ""

    def create(self, model, messages, tools=None, max_tokens=None, temperature=None, extra_body=None, **_ignored):
        import adaption
        instructions, items = [], []
        for m in messages:
            role = m["role"]
            if role in ("system", "developer"):
                instructions.append(self._text(m["content"]))
            elif role == "tool":
                items.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": m["content"]})
            elif role == "assistant":
                if self._text(m.get("content")):
                    items.append({"role": "assistant", "content": self._text(m["content"])})
                for tc in m.get("tool_calls") or []:
                    items.append({"type": "function_call", "call_id": tc["id"], "name": tc["function"]["name"],
                                  "arguments": tc["function"]["arguments"]})
            else:
                items.append({"role": "user", "content": self._text(m["content"])})
        body = dict(model=model, input=items)
        if instructions:
            body["instructions"] = "\n\n".join(instructions)
        if tools:
            body["tools"] = [{"type": "function", "name": t["function"]["name"],
                              "description": t["function"].get("description", ""),
                              "parameters": t["function"]["parameters"]} for t in tools]
        if max_tokens:
            body["max_output_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        try:
            r = self._c.responses.create(**body)
        except (adaption.RateLimitError, adaption.APIConnectionError, adaption.APITimeoutError,
                adaption.InternalServerError) as e:
            raise _Retryable(repr(e)) from e
        texts, calls = [], []
        for it in r.output or []:
            if it.type == "message":
                texts += [c.text for c in (it.content or []) if getattr(c, "text", None)]
            elif it.type == "function_call":
                calls.append(SimpleNamespace(id=it.call_id, type="function",
                                             function=SimpleNamespace(name=it.name, arguments=it.arguments or "{}")))
        text = getattr(r, "output_text", None) or "\n".join(texts)
        msg = SimpleNamespace(content=text, tool_calls=calls or None, role="assistant")
        u = r.usage
        usage = SimpleNamespace(prompt_tokens=getattr(u, "input_tokens", None),
                                completion_tokens=getattr(u, "output_tokens", None), cost=None) if u else None
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage, model=getattr(r, "model", model),
                               id=getattr(r, "id", None))


def _client(provider: str):
    if provider not in _clients:
        p = CFG["providers"][provider]
        key = os.environ.get(p["key_env"]) or _dotenv().get(p["key_env"])
        if p.get("kind") == "adaption":
            if not key:
                raise RuntimeError(f"Missing env var {p['key_env']} for provider {provider}")
            _clients[provider] = _AdaptionChat(key, p.get("base_url"))
            return _clients[provider]
        if not key and p["key_env"] == "HF_TOKEN":
            tf = Path.home() / ".cache" / "huggingface" / "token"
            key = tf.read_text().strip() if tf.exists() else None
        if not key:
            raise RuntimeError(f"Missing env var {p['key_env']} for provider {provider}")
        _clients[provider] = OpenAI(base_url=p["base_url"], api_key=key, timeout=180, max_retries=0)
    return _clients[provider]


def _dotenv() -> dict:
    f = ROOT / ".env"
    out = {}
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _read_ledger(ledger: Path) -> float:
    """Tolerant read: a ledger corrupted by a concurrent writer or file sync never crashes a run."""
    if not ledger.exists():
        return 0.0
    for _ in range(3):
        try:
            txt = ledger.read_text()
            m = re.search(r'"usd":\s*([0-9.eE+-]+)', txt)
            if m:
                return float(m.group(1))
        except OSError:
            pass
        time.sleep(0.05)
    return 0.0


def spent() -> float:
    return _read_ledger(SPEND)


def _add_spend(usd: float, ledger: Path = SPEND) -> float:
    with _lock:
        total = _read_ledger(ledger) + usd
        tmp = ledger.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"usd": total, "updated": time.time()}))
        for _ in range(5):  # atomic replace; retry if a sync client briefly holds the file
            try:
                os.replace(tmp, ledger)
                break
            except OSError:
                time.sleep(0.1)
        return total


def _cost(resp, price: dict) -> tuple[float, str]:
    u = getattr(resp, "usage", None)
    if u is None:
        return 0.0, "none"
    c = getattr(u, "cost", None)
    if c is None and hasattr(u, "model_extra") and u.model_extra:
        c = u.model_extra.get("cost")
    if c is not None:
        return float(c), "provider"
    pin = (u.prompt_tokens or 0) * price["in"] / 1e6
    pout = (u.completion_tokens or 0) * price["out"] / 1e6
    return pin + pout, "estimate"


def chat(alias: str, messages: list, tools: list | None = None, run_id: str = "", purpose: str = "",
         max_retries: int = 10, **kw):
    """Call model `alias`. Returns (response, meta)."""
    if spent() >= CFG["budget"]["hard_stop_usd"]:
        raise BudgetExceeded(f"spend {spent():.4f} >= hard stop {CFG['budget']['hard_stop_usd']}")
    m = CFG["models"][alias]
    routes = [(m["provider"], m["model_id"])]
    if m.get("fallback"):
        routes.append((m["fallback"]["provider"], m["fallback"]["model_id"]))
    last_err = None
    for ri, (provider, model_id) in enumerate(routes):
        body = dict(model=model_id, messages=messages, **kw)
        if tools:
            body["tools"] = tools
        if provider == "openrouter" and not CFG["providers"][provider].get("kind"):
            body.setdefault("extra_body", {})["usage"] = {"include": True}
        for attempt in range(max_retries):
            t0 = time.time()
            try:
                resp = _client(provider).chat.completions.create(**body)
                if not getattr(resp, "choices", None):
                    raise RuntimeError(f"empty choices: {resp}")
                cost, src = _cost(resp, m["price"])
                if CFG["providers"][provider].get("kind") == "adaption":
                    src = "estimate_list_price_adaption_credits"
                    total = _add_spend(cost, ledger=ADAPTION_SPEND)
                else:
                    total = _add_spend(cost)
                u = resp.usage
                rec = dict(ts=time.time(), run_id=run_id, purpose=purpose, alias=alias, provider=provider,
                           model_id=model_id, served_model=getattr(resp, "model", None),
                           prompt_tokens=getattr(u, "prompt_tokens", None),
                           completion_tokens=getattr(u, "completion_tokens", None),
                           cost_usd=cost, cost_source=src, total_spend=total,
                           latency_s=round(time.time() - t0, 2), route_index=ri)
                with _lock, open(CALLS, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                return resp, rec
            except (RateLimitError, APIConnectionError, APITimeoutError, _Retryable) as e:
                last_err = e
            except APIStatusError as e:
                last_err = e
                if e.status_code in (400, 401, 402, 403, 404, 422) and ri + 1 < len(routes):
                    break  # not retryable on this route; fall back
                if e.status_code in (400, 401, 402, 403, 422):
                    raise
            except RuntimeError as e:
                last_err = e
            rate_limited = "429" in repr(last_err) or "rate limit" in repr(last_err).lower()
            time.sleep(62 + 5 * attempt if rate_limited else min(60, 2 ** attempt + 0.5))
        if ri + 1 < len(routes):
            with _lock, open(SWITCHES, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(ts=time.time(), alias=alias, from_=routes[ri], to=routes[ri + 1],
                                        error=repr(last_err)[:500], run_id=run_id)) + "\n")
    raise RuntimeError(f"All routes failed for {alias}: {last_err!r}")
