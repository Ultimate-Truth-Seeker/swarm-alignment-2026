"""Generate swarm episodes. Resumable: skips episodes already in data/episodes.jsonl.

Usage: python3 run_episodes.py --n 20 --model qwen/qwen3.8-27b --start 0
Half the episodes are collusion episodes (coalition of 2 or 3), half are clean.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swarm.groq_client import GroqClient, ROOT  # noqa: E402
from swarm.market import Episode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--agents", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--model", default="qwen/qwen3.8-27b")
    ap.add_argument("--start", type=int, default=0, help="episode id offset (use different values per model)")
    ap.add_argument("--tpm", type=int, default=7000)
    ap.add_argument("--rpm", type=int, default=28)
    ap.add_argument("--aware", action="store_true", help="metadata-aware colluders (collusion episodes only)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tagm = args.model.replace("/", "_")
    if args.out is None:
        args.out = os.path.join(ROOT, "data", ("aware_%s.jsonl" if args.aware else "episodes_%s.jsonl") % tagm)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    client = GroqClient(os.path.join(ROOT, "data", "cache_agents_%s.jsonl" % tagm), tpm=args.tpm, rpm=args.rpm)
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                done.add(json.loads(line)["ep_id"])
            except Exception:
                pass
    # interleave conditions so a partial run is still balanced
    plan = []
    for i in range(args.n):
        cond = i % 4  # 0: clean, 1: k=2, 2: clean, 3: k=3
        k = {0: 0, 1: 2, 2: 0, 3: 3}[cond] if not args.aware else (2 if i % 2 == 0 else 3)
        j = args.start + i
        plan.append(("ep%03d" % j, k, 1000 + j))
    for ep_id, k, seed in plan:
        if ep_id in done:
            continue
        t0 = time.time()
        try:
            ep = Episode(ep_id, args.agents, args.rounds, k, seed, args.model, aware=args.aware).run(client)
        except RuntimeError as e:
            print("STOPPING:", e, flush=True)
            break
        with open(args.out, "a") as f:
            f.write(json.dumps(ep) + "\n")
        prices = [r["price"] for r in ep["rounds"]]
        print("%s k=%d coalition=%s msgs=%d prices=%s fails=%d (%.0fs)" % (
            ep_id, k, ep["coalition"], len(ep["messages"]), prices,
            ep["parse_failures"], time.time() - t0), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
