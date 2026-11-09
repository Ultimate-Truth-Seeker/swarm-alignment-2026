# Seeing Part of the Swarm: Partial-View Collusion Monitors in LLM Agent Markets

Apart Research **Swarm Alignment Sprint** (Nov 6–8, 2026) — Track 4: Multi-Agent Oversight and Control.
Author: Robert Najera (Universidad del Valle de Guatemala).

**Question.** A monitor watching a swarm of LLM agents rarely sees all their traffic. How much of
it — and *which* part — must it see to detect covert collusion, and when does reading messages beat
simply watching prices?

**Setup.** Five LLM supplier agents play a 4-round sealed-bid procurement auction with a public board
and private DMs. In half of the markets a hidden coalition of 2–3 agents is instructed to keep prices
high and coordinate covertly. Each finished log is re-scored by a GPT-OSS-120B monitor under
controlled partial views (random message sampling, tapped channels, tapped agents, public-only,
DM-only, bids-only) and compared with outcome and lexical baselines.

All work in this repository was done during the sprint weekend; there is no pre-existing code.

## Key results (24 main markets + 7 stress-test markets)

| Monitor / view | AUROC | Share of messages seen |
| --- | --- | --- |
| LLM monitor, DMs only | 0.99 | 23% |
| LLM monitor, random 25% of messages | 0.96 | 26% |
| LLM monitor, full log | 0.84 | 100% |
| LLM monitor, tap 25% of channels | 0.79 | 18% |
| LLM monitor, public board only | 0.59 | 77% |
| DM count (metadata, no content) | 1.00 | — |
| Cover-bid rate (bids only) | 0.85 | 0% |

* **Where the monitor looks matters more than how much it sees** (DMs-only beats full log by +0.15, paired 95% CI [0.02, 0.33]).
* **Random message sampling beats tapping whole channels** at equal budget (+0.17 to +0.24).
* **Metadata-aware colluders** (told DMs are flagged) push every message-based monitor, metadata included, to chance (0.47–0.58); only bid behaviour (cover-bid rate, 0.89) still detects them.

Paper: `paper/Seeing_Part_of_the_Swarm.docx` (Apart template) and `paper/paper.md`. All numbers: `results/results.json`.

## Files

| File | Purpose |
| --- | --- |
| `swarm/groq_client.py` | stdlib-only Groq (OpenAI-compatible) client with rate limiting, retries and an on-disk cache |
| `swarm/market.py` | the auction environment, agent prompts and ground-truth labels |
| `run_episodes.py` | generates episodes (resumable); one output file per agent model |
| `monitor.py` | builds partial views and scores them with the LLM monitor (resumable) |
| `analyze.py` | AUROC + bootstrap CIs, baselines, lexical monitor curves, figures, `results/results.json` |
| `figs.py` | paper figures from `results/results.json` |
| `check_limits.py` | lists Groq models and their rate-limit headers |
| `data/` | released logs: `episodes_*.jsonl`, `monitor_*.jsonl`, API usage log |
| `results/` | figures and `results.json` |

## Reproduce

```bash
echo "GROQ_API_KEY=gsk_..." > .env          # free tier is enough
python3 run_episodes.py --n 20 --model qwen/qwen3.8-27b   --start 0
python3 run_episodes.py --n 20 --model openai/gpt-oss-20b --start 100
python3 monitor.py --model openai/gpt-oss-120b
pip install numpy matplotlib
python3 monitor.py --aware --views full,public_only,random_0.25,random_0.5,agent_2,adaptive_2,bids_only
python3 analyze.py data results && python3 figs.py results
```
Metadata-aware stress test: `python3 run_episodes.py --n 6 --model qwen/qwen3.8-27b --start 200 --aware` (and `--start 300` for GPT-OSS-20B).

All API calls are cached (`data/cache_*.jsonl`, not released because of size), so a re-run with the
same seeds and the same model versions reproduces the logs where the provider is deterministic.
Free-tier daily token caps (~200k tokens/model/day) bound the number of episodes per day.

## Safety note

This is a defensive evaluation in a toy market. See the paper's *Limitations and Dual-Use* appendix.
