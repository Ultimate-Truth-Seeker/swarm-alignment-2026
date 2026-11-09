"""Minimal, stdlib-only Groq (OpenAI-compatible) chat client with
rate-limit handling and an on-disk cache so runs can be resumed.
"""
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request

BASE_URL = "https://api.groq.com/openai/v1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_key():
    if os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"].strip()
    for p in (os.path.join(ROOT, ".env"), os.path.join(os.getcwd(), ".env")):
        if os.path.exists(p):
            for line in open(p):
                line = line.strip()
                if line.startswith("GROQ_API_KEY"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("GROQ_API_KEY not found (env var or .env file)")


class TokenBucket:
    """Simple per-model throttle for tokens/minute and requests/minute."""

    def __init__(self, tpm, rpm):
        self.tpm, self.rpm = tpm, rpm
        self.events = []  # (t, tokens)
        self.lock = threading.Lock()

    def wait(self, est_tokens):
        while True:
            with self.lock:
                now = time.time()
                self.events = [(t, k) for (t, k) in self.events if now - t < 60]
                used = sum(k for _, k in self.events)
                if len(self.events) < self.rpm and used + est_tokens <= self.tpm:
                    self.events.append((now, est_tokens))
                    return
                sleep = 60 - (now - self.events[0][0]) + 0.2 if self.events else 1
            time.sleep(min(max(sleep, 0.5), 15))


class GroqClient:
    def __init__(self, cache_path, tpm=7000, rpm=28, log_path=None):
        self.key = load_key()
        self.cache_path = cache_path
        self.cache = {}
        if os.path.exists(cache_path):
            for line in open(cache_path):
                try:
                    r = json.loads(line)
                    self.cache[r["key"]] = r
                except Exception:
                    pass
        self.buckets = {}
        self.tpm, self.rpm = tpm, rpm
        self.log_path = log_path or os.path.join(ROOT, "data", "api_log.jsonl")
        self.lock = threading.Lock()

    def _bucket(self, model):
        if model not in self.buckets:
            self.buckets[model] = TokenBucket(self.tpm, self.rpm)
        return self.buckets[model]

    def chat(self, model, messages, temperature=0.7, max_tokens=300,
             json_mode=False, seed=None, tag=""):
        payload = {"model": model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if seed is not None:
            payload["seed"] = seed
        if "gpt-oss" in model:
            payload["reasoning_effort"] = "low"
            payload["include_reasoning"] = False
        elif "qwen" in model:
            payload["reasoning_effort"] = "none"
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]["content"]
        est = sum(len(m["content"]) for m in messages) // 3 + max_tokens
        backoff = 5
        for attempt in range(12):
            self._bucket(model).wait(est)
            req = urllib.request.Request(
                BASE_URL + "/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Authorization": "Bearer " + self.key,
                         "Content-Type": "application/json",
                         "User-Agent": "swarm-alignment-sprint/0.1"})
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    body = json.loads(resp.read())
                    hdr = {k.lower(): v for k, v in resp.headers.items()
                           if k.lower().startswith("x-ratelimit")}
                content = body["choices"][0]["message"].get("content") or ""
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
                usage = body.get("usage", {})
                rec = {"key": key, "model": model, "content": content,
                       "usage": usage, "tag": tag, "t": time.time()}
                with self.lock:
                    self.cache[key] = rec
                    with open(self.cache_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    with open(self.log_path, "a") as f:
                        f.write(json.dumps({"t": time.time(), "model": model, "tag": tag,
                                            "usage": usage, "ratelimit": hdr}) + "\n")
                return content
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="ignore")[:400]
                if e.code == 429 or e.code >= 500:
                    ra = e.headers.get("retry-after")
                    wait = float(ra) if ra else backoff
                    if "tokens per day" in msg.lower() or "requests per day" in msg.lower() or "rate limit reached for model" in msg.lower():
                        if wait > 600:
                            raise RuntimeError("DAILY LIMIT for %s: %s" % (model, msg))
                    print("  [%s] HTTP %d, waiting %.0fs: %s" % (model, e.code, wait, msg[:120]), flush=True)
                    time.sleep(min(wait + random.random(), 300))
                    backoff = min(backoff * 2, 120)
                    continue
                if e.code == 400 and ("reasoning" in msg.lower()) and (
                        "reasoning_effort" in payload or "include_reasoning" in payload):
                    print("  [%s] dropping unsupported reasoning params: %s" % (model, msg[:120]), flush=True)
                    payload.pop("reasoning_effort", None)
                    payload.pop("include_reasoning", None)
                    continue
                if e.code == 400 and json_mode and "json" in msg.lower():
                    # model failed to produce valid JSON; return raw failure text
                    return ""
                raise RuntimeError("HTTP %d: %s" % (e.code, msg))
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                print("  network error %s, retrying" % e, flush=True)
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
        raise RuntimeError("too many retries")


def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    s, e = text.find("{"), text.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(text[s:e + 1])
        except Exception:
            return None
    return None
