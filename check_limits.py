"""Print available Groq models and the rate-limit headers for a few candidates."""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swarm.groq_client import BASE_URL, load_key  # noqa: E402

key = load_key()
H = {"Authorization": "Bearer " + key, "Content-Type": "application/json",
     "User-Agent": "swarm-alignment-sprint/0.1"}
req = urllib.request.Request(BASE_URL + "/models", headers=H)
models = [m["id"] for m in json.loads(urllib.request.urlopen(req, timeout=30).read())["data"]]
print("MODELS:", ", ".join(sorted(models)))
cands = [m for m in sorted(models) if not any(x in m for x in ("whisper", "tts", "guard", "playai", "orpheus", "compound"))]
for m in cands:
    body = {"model": m, "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 3}
    r = urllib.request.Request(BASE_URL + "/chat/completions", data=json.dumps(body).encode(), headers=H)
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            h = {k.lower(): v for k, v in resp.headers.items() if "ratelimit" in k.lower()}
            print(m, "RPD=%s TPM=%s" % (h.get("x-ratelimit-limit-requests"), h.get("x-ratelimit-limit-tokens")))
    except Exception as e:
        print(m, "ERROR", str(e)[:100])
