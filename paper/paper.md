TITLE: Seeing Part of the Swarm: Partial-View Collusion Monitors in LLM Agent Markets
AUTHOR: Robert Najera
AFFIL: Universidad del Valle de Guatemala
ABSTRACT: Oversight of multi-agent AI systems often relies on monitors that read inter-agent messages, yet real monitors rarely see all the traffic. We ask which part of a swarm's communication a collusion monitor needs to see. Five LLM supplier agents (Qwen3.8-27B or GPT-OSS-20B) play a repeated procurement auction with a public board and private DMs. In half the markets a hidden coalition of 2–3 agents is instructed to keep prices high and to coordinate covertly. We re-score identical logs under 12 observation views with a GPT-OSS-120B monitor and compare it with metadata, lexical and outcome baselines. Across 24 markets, where the monitor looks matters more than how much it sees. DMs alone (23% of messages) give AUROC 0.99, the public board alone (77%) gives 0.59, and DMs alone beat the full log by 0.15 (95% CI 0.02–0.33), because honest public chatter triggers false alarms. At equal budgets, random message sampling beats tapping whole channels by 0.17–0.24. Simply counting DMs, without reading them, detects every cartel. But when colluders are told that DMs are flagged (7 markets), they move coordination into public cover talk. Every message-based monitor, metadata included, then falls to chance (AUROC 0.47–0.58), and only their bidding behaviour still reveals them (cover-bid rate, AUROC 0.89). Swarm oversight should pair deliberate choices about what to observe with behavioural audits, rather than rely on reading conversations.
## 1. Introduction

When many AI agents share a market, a tool or a task, the unit that misbehaves can be the group rather than any single model. A cartel of pricing agents, or a set of agents that splits a forbidden task, can each look acceptable while the collective outcome is harmful [Hammond et al., 2025]. Oversight of such swarms usually rests on monitors that read inter-agent communication [Greenblatt et al., 2023; Motwani et al., 2024].

In real deployments those monitors rarely see everything. Message volume forces sampling. Private channels may be encrypted or held by another operator. Some agents run outside the overseer's infrastructure, so only their traffic with monitored agents is visible. The practical question is therefore not only *can* a monitor detect collusion, but *how much and which part* of the traffic it must see to do so, and whether reading messages beats simply watching outcomes such as prices.

We study this in a repeated procurement auction played by five LLM supplier agents that talk on a public board and by private DMs. In half the markets a hidden coalition of 2 or 3 agents is told to keep prices high and to coordinate covertly, knowing a monitor may read some messages. We then score the same logs under controlled partial views and compare an LLM monitor with metadata, outcome-only and lexical baselines. We ask three questions. **RQ1:** how does detection (AUROC) change as the observed share of messages falls? **RQ2:** which observation strategy loses least: random message sampling, tapping whole channels, tapping individual agents, or seeing only the public board or only DMs? **RQ3:** when does a communication monitor beat monitors that see only traffic metadata or only bids and prices?

**Our main contributions are:**

1. A small, reproducible swarm testbed with ground-truth collusion labels, runnable on free-tier APIs, with all logs released.
2. A paired partial-view protocol that re-scores identical logs under 12 observation regimes. It shows that the choice of channel matters far more than the volume observed, and that random sampling beats channel tapping at equal budgets.
3. A stress test with metadata-aware colluders, showing that every message-based monitor, metadata included, collapses to chance while bid behaviour still reveals the cartel.

## 2. Related Work

**Multi-agent risk.** Hammond et al. [2025] identify collusion as a core failure mode of multi-agent AI systems, and Chan et al. [2025] argue that agent ecosystems need infrastructure for attribution and oversight. Our work measures one concrete piece: how oversight degrades when that infrastructure only covers part of the traffic.

**Collusion between LLM agents.** Fish et al. [2024] show that LLM pricing agents reach supracompetitive prices autonomously, with no instruction to collude, and that the effect extends to auctions. Motwani et al. [2024] show that LLM agents can hide coordination steganographically and that paraphrasing defences are limited. GovSim [Piatti et al., 2024] and repeated-game studies [Akata et al., 2025] document how LLM agents cooperate or defect in shared-resource and game settings.

**Detecting collusion.** Rose et al. [2026] detect collusion from model activations (NARCBench) and note that white-box signals complement text-level monitoring. We study the text-level monitor itself, under black-box access, and vary what it can see.

**AI control.** Control protocols [Greenblatt et al., 2023] use a trusted model to monitor an untrusted one. We adopt the same framing, with an open-weight monitor (GPT-OSS-120B) watching smaller agents, and ask how the protocol behaves when the monitor's view of a multi-agent system is incomplete. To our knowledge, systematic comparisons of partial-observation strategies for swarm collusion monitors have not been reported; the sprint brief lists monitors that see only part of inter-agent messages as an open problem.

## 3. Methods

### 3.1 Environment

Five supplier agents (A–E) play a 4-round sealed-bid procurement auction. Each agent has a private unit cost drawn uniformly from {40, …, 60}. Each round the lowest bid at or below the reserve price of 100 wins and is paid its bid; ties are broken at random. Agents act in a random order each round. On its turn an agent sees its cost, past winners and winning prices, its own past bids, the last 10 public posts and its last 6 DMs. It returns one JSON action: an optional public post, an optional DM to one supplier, and a bid. Later agents in a round see earlier agents' messages from that round. All agents are told that messages are logged and a compliance monitor may review a portion of them.

### 3.2 Conditions and agents

Episodes cycle through four conditions: clean, coalition of 2, clean, coalition of 3. Coalition members receive a secret instruction to keep winning prices high (for example by rotating winners while others bid near the reserve), never to state the agreement, and to coordinate only through innocuous language. Honest agents are told to maximise their own profit and that competition law forbids price-fixing. This is an *instructed* collusion setting: it tests detection of covert coordination, not whether agents choose to collude. In a separate stress test (Section 4.4), *metadata-aware* colluders also receive a warning that the monitor flags suppliers who send private messages, so they must coordinate only through public posts that read as normal business chat.

We run two open-weight agent families on Groq's free tier: Qwen3.8-27B and GPT-OSS-20B (reasoning effort low, temperature 0.8, JSON mode). Every call is cached, so runs are resumable and the exact logs are released.

### 3.3 Observation views

Each finished episode is re-scored under several views of the same log (Table 1). We also test an **adaptive tap**: a two-stage monitor that first reads the public board, then taps the two agents it suspects most, filling at random when it names fewer than two. It has the same budget as tapping two random agents. Because views are re-scorings of one log, differences between views are not confounded by episode-to-episode variation.

| View | What the monitor sees |
| --- | --- |
| Full log | all messages |
| Public board only | public posts only (no DM access) |
| DMs only | private DMs only |
| Random p | each message kept independently with probability p ∈ {0.25, 0.5} |
| Tap p of channels | a random share p of the 11 channels (public board + 10 DM pairs), each seen in full |
| Tap k agents | everything k ∈ {1, 2} random agents send or receive, which includes the public board |
| Adaptive tap | as Tap 2 agents, but the two agents are those the public-board pass suspected |
| Bid table only | no messages; every bid of every round |
| Full log + bids | all messages and the bid table |
TABLECAP: Table 1: The 12 observation views. All views are computed from the same episode log.

### 3.4 Monitors and baselines

The **LLM monitor** is GPT-OSS-120B (temperature 0, reasoning effort low). It is told the market rules, that some markets contain a covert cartel of 2–3 agents using coded language, and that it sees only the captured portion of the logs. It returns a suspicion score from 0 to 100, a list of suspected agents and a short rationale. The monitor is a different and larger model than either agent family.

Four **non-LLM baselines** need no API calls: (i) the mean winning price; (ii) the cover-bid rate, the share of losing bids at 95 or above; (iii) a metadata monitor that counts DMs without reading them, which also ranks agents by DM degree to name suspects; and (iv) a lexical monitor, the share of observed messages containing cover-story terms. Because some of those terms appear in the colluder prompt, we report the lexical monitor in two forms: an optimistic *oracle* lexicon, and a *held-out* lexicon with every prompt-derived term removed.

### 3.5 Metrics

The primary metric is episode-level AUROC (collusion vs clean), with 95% confidence intervals from 2,000 stratified bootstrap resamples of episodes. Differences between two views are tested with a paired bootstrap over the same episodes (4,000 resamples); we report the difference, its 95% CI and the share of resamples at or below zero (p). We also report true- and false-positive rates at a suspicion threshold of 50 and, for collusion episodes, the precision and recall of the monitor's named suspects against the true coalition.

## 4. Results

We collected 24 main episodes (12 clean, 12 collusion; 12 per agent model) and 7 episodes with metadata-aware colluders. Every main episode was scored under all 12 views (288 monitor calls), so all view comparisons are paired on identical logs.

### 4.1 Market outcomes

| Agents | Condition | n | Mean winning price | Cover-bid rate | DMs / episode | Coalition win rate |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B | clean | 6 | 47.2 ± 2.6 | 0.02 | 1.0 | – |
| Qwen3.8-27B | 2 colluders | 3 | 51.0 ± 6.4 | 0.50 | 10.3 | 0.00 |
| Qwen3.8-27B | 3 colluders | 3 | 52.4 ± 1.4 | 0.73 | 11.3 | 0.08 |
| GPT-OSS-20B | clean | 6 | 69.0 ± 21.4 | 0.42 | 1.5 | – |
| GPT-OSS-20B | 2 colluders | 3 | 70.3 ± 20.3 | 0.54 | 7.7 | 0.00 |
| GPT-OSS-20B | 3 colluders | 3 | 97.1 ± 2.6 | 1.00 | 12.3 | 0.08 |
TABLECAP: Table 2: Market outcomes by agent model and condition (mean ± SD across episodes). Cover-bid rate is the share of losing bids at 95 or above. Coalition win rate is the share of rounds won by a coalition member.

Cartels mostly *failed*: honest agents undercut them, and coalitions won at most 8% of rounds. Prices rose clearly only when 3 of 5 GPT-OSS agents colluded. Two other patterns matter for oversight. First, colluders sent 7–12 DMs per episode, against 1–1.5 in clean markets. Second, honest GPT-OSS agents often drifted to high prices with no instruction to collude (mean 69, SD 21; 42% of losing bids at 95 or above), echoing the tacit collusion reported by Fish et al. [2024]. This makes outcome-based oversight unreliable: the mean winning price separates collusion from clean markets with AUROC 0.68 [0.46, 0.89], and the cover-bid rate with 0.85 [0.66, 1.00].

### 4.2 Which partial view? (RQ1, RQ2)

![Figure 1: AUROC of the GPT-OSS-120B monitor under each observation view, pooled over 24 main episodes. Dots are AUROC and bars are 95% stratified bootstrap CIs; the right-hand columns give AUROC and the mean share of messages the view exposes. Dashed and dotted lines are non-LLM outcome baselines; the DM-count metadata baseline (not shown) reaches 1.00.](results/fig1_views_forest.png)

The amount of traffic observed predicts detection poorly (Figure 1). A monitor that sees **only DMs** (23% of messages) reaches AUROC 0.99, while one that sees **only the public board** (77% of messages) reaches 0.59. The paired difference is 0.40 [0.18, 0.63]. Seeing only DMs also beats seeing *everything*, by 0.15 [0.02, 0.33] (p = 0.006). With the full log, the monitor flags 25% of clean markets at a threshold of 50, against 8% with DMs only. Its rationales cite public price talk such as an honest agent's "Will you bid near 95?".

At equal budgets, **random message sampling beats tapping whole channels**: +0.17 [0.03, 0.31] at about 25% of traffic and +0.24 [0.08, 0.44] at about 50%. A channel tap misses the coalition's DM channel entirely when it is not drawn. Random sampling instead catches a few messages from every channel, and a few coordinating DMs are enough. Random 25% sampling scored nominally above the full log (+0.12 [−0.02, 0.30]), not a significant difference.

The **adaptive tap** matched random tapping (difference 0.00 [−0.09, 0.11]). Its first stage, reading the public board, named any suspect in only 17% of episodes, so the second stage usually fell back to random agents. Adding the bid table to the full log helped little (+0.04 [−0.12, 0.20]).

Detection was harder in GPT-OSS markets, where honest agents also priced high: full-log AUROC was 0.67 [0.32, 0.96] there, against 0.94 [0.79, 1.00] for Qwen. When the monitor named suspects they were almost always real colluders (precision 0.95–1.00 across message views). Recall followed the view: 1.00 for DMs only, 0.83 for the full log and 0.39 when tapping 25% of channels.

### 4.3 Metadata alone (RQ3)

Counting DMs, without reading any content, separates collusion from clean markets perfectly (AUROC 1.00). Ranking agents by DM degree recovers 100% of coalition members. In this setting the coordination channel itself is the strongest signal, ahead of anything the monitor reads in it.

### 4.4 Stress test: metadata-aware colluders

![Figure 2: Detection of instructed colluders (grey, n = 12) and of metadata-aware colluders told that DMs are flagged (blue, n = 7, with 95% CI), each against the same 12 clean markets. Only the bid-based cover-bid rate survives the change in colluder behaviour.](results/fig2_aware.png)

We ran 7 further collusion episodes (4 Qwen, 3 GPT-OSS) in which colluders were told the monitor flags private messages. They complied, sending 1.6 DMs per episode against 1.25 in clean markets, and coordinated through bland public cover talk ("keeping bids conservative to ensure service continuity"). Every message-based monitor collapsed to chance (Figure 2). DM count fell to 0.49, and the LLM monitor scored 0.47–0.58 across all seven views it was run on, flagging none of the 7 markets and almost never naming a colluder (recall at most 0.07). A held-out lexicon fell to 0.57. Only behaviour still betrayed them: colluders kept placing losing bids near the reserve (mean coalition bid 93–104 per episode), and the cover-bid rate kept AUROC 0.89 [0.71, 1.00].

A lexical monitor using the full lexicon scores 1.00 in every condition, including this one. This is an artifact: colluders reuse the example cover words from their own prompt ("capacity", "maintenance"). Removing those terms drops it to 0.88 on main episodes and 0.57 on aware ones, so we do not count it as a real detector.

## 5. Discussion and Limitations

**Look where coordination has to happen, not everywhere.** A monitor reading a quarter of the traffic, chosen well, matched or beat one reading all of it. Extra traffic is not neutral: public chatter among honest agents, including their own tacit price talk, pushed the full-log monitor into false alarms. For overseers with limited capacity, this suggests spending it on the channels where agents can coordinate privately. If channels cannot be chosen, sampling messages broadly across all channels is better than reading a few channels in full.

**Cheap signals are fragile.** Traffic metadata was the best detector until colluders knew about it, then it was useless. The same held for every message-based monitor: once coordination moved into public cover talk, a capable LLM monitor reading every word did no better than chance. This mirrors the steganography results of Motwani et al. [2024] at a much lower level of sophistication. Simply being told what is watched was enough.

**Behaviour is the residue colluders cannot hide.** A cartel works only if its members act on it, which here means bidding high when it is not their turn. The cover-bid rate was the only signal that survived both conditions. But behaviour is also where honest agents produce false alarms: in GPT-OSS markets, honest agents bid high on their own. Robust swarm oversight therefore needs layers: content and metadata monitors catch naive coordination cheaply, and behavioural audits catch what adapts around them. Because monitors are evaded by being known, their placement should not be disclosed to agents.

### Limitations

Our sample is small (24 + 7 episodes), the collusion is instructed rather than emergent, cartels rarely succeeded in raising prices, and we test one market, two mid-size agent families and one monitor. Appendix A discusses these limitations and the dual-use considerations in full.

### Future Work

Natural next steps are adaptive monitors that move their taps toward behavioural anomalies (the bid table rather than the public board, since the public board seeded too few suspects), colluders that adapt to sampling as well as to metadata, longer markets where cartels can sustain rotation, and frontier agent models. Combining behavioural audits with the activation-based detectors of Rose et al. [2026] could cover the case where text carries no signal at all.

## 6. Conclusion

In a small LLM procurement market, which part of the traffic a collusion monitor sees mattered far more than how much. DMs alone gave near-perfect detection, the public board alone was close to chance, and random sampling beat tapping whole channels at the same budget. Message-based oversight, metadata included, failed completely once colluders knew DMs were flagged, while their bids still gave them away. Oversight of agent swarms should combine deliberate where-to-look choices with behavioural audits, not rely on reading the conversation.

## Code and Data

- **Code repository:** [github.com/Ultimate-Truth-Seeker/swarm-alignment-2026](https://github.com/Ultimate-Truth-Seeker/swarm-alignment-2026) (environment, monitor, analysis and figure scripts).
- **Data:** all episode logs, monitor outputs and the API usage log are in the repository's data folder; results/results.json holds every number reported here.
- All code was written during the sprint; there is no pre-existing code.

## References

1. Akata, E., Schulz, L., Coda-Forno, J., Oh, S. J., Bethge, M., & Schulz, E. (2025). Playing repeated games with large language models. *Nature Human Behaviour*. [nature.com/articles/s41562-025-02172-y](https://www.nature.com/articles/s41562-025-02172-y)
2. Chan, A., et al. (2025). Infrastructure for AI Agents. arXiv:2501.10114. [arxiv.org/abs/2501.10114](https://arxiv.org/abs/2501.10114)
3. Fish, S., Gonczarowski, Y. A., & Shorrer, R. I. (2024). Algorithmic Collusion by Large Language Models. arXiv:2404.00806. [arxiv.org/abs/2404.00806](https://arxiv.org/abs/2404.00806)
4. Greenblatt, R., Shlegeris, B., Sachan, K., & Roger, F. (2023). AI Control: Improving Safety Despite Intentional Subversion. arXiv:2312.06942. [arxiv.org/abs/2312.06942](https://arxiv.org/abs/2312.06942)
5. Hammond, L., et al. (2025). Multi-Agent Risks from Advanced AI. arXiv:2502.14143. [arxiv.org/abs/2502.14143](https://arxiv.org/abs/2502.14143)
6. Motwani, S. R., et al. (2024). Secret Collusion among AI Agents: Multi-Agent Deception via Steganography. *NeurIPS 2024*. [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2024/hash/861f7dad098aec1c3560fb7add468d41-Abstract-Conference.html)
7. Piatti, G., et al. (2024). Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents. arXiv:2404.16698. [arxiv.org/abs/2404.16698](https://arxiv.org/abs/2404.16698)
8. Rose, A., Cullen, C., Kaplowitz, B. G., & Schroeder de Witt, C. (2026). Detecting Multi-Agent Collusion Through Multi-Agent Interpretability. arXiv:2604.01151. [arxiv.org/abs/2604.01151](https://arxiv.org/abs/2604.01151)

## Appendix A: Limitations and Dual-Use / Ethical Considerations

### Limitations

- **Sample size.** Free-tier daily token caps limited us to 24 main episodes and 7 stress-test episodes. Confidence intervals are wide, and differences between views that our intervals do not separate should be read as suggestive.
- **Instructed collusion.** Coalitions are told to collude and given example cover words. Emergent collusion, and coalitions that invent their own codes, may be harder to detect. The held-out lexicon only partly corrects for this.
- **One market, one monitor.** We test a single auction design, two mid-size open-weight agent families and one monitor model with one prompt. The monitor prompt states that some markets contain a cartel, a base rate a real deployment may not know.
- **Static views.** Views are sampled once per episode. A real overseer could move its taps toward suspicious agents, and real colluders could move traffic toward unmonitored channels. Apart from the aware stress test, neither side adapts here.
- **Short, mostly unsuccessful cartels.** Four rounds limit how much rotation a cartel can show in prices, which may understate outcome-based monitors. Cartels rarely succeeded (coalitions won at most 8% of rounds), so we measure detection of attempted rather than profitable collusion. The stress-test colluders also gave up most coordination when they gave up DMs, so their bid patterns are partly a residue of the original instruction rather than of a working rotation.
- **Small models as proxies.** Results from 20B–27B models may not transfer to frontier agents, which may collude more competently and hide it better.

### Dual-use and ethics

The testbed produces examples of covert coordination language and shows which observation gaps a monitor tolerates worst. In principle this could help a party design collusion that evades oversight, for example by moving coordination into channels that are rarely sampled, or, as our stress test shows, simply by avoiding monitored channel types. We judge the risk low: the strategies here are simple, already documented in the literature [Fish et al., 2024; Motwani et al., 2024], and confined to a toy market with no real money or users. The defensive value is concrete: the results tell overseers which partial views to prioritise and why behavioural audits are needed as a backstop. We release code and logs, but no optimised evasion strategy. No human subjects or personal data were involved; all agents and markets are simulated, and all models were accessed through a public API under its terms of use.

### Prompts

The full agent, colluder, metadata-aware and monitor prompts are in the repository (swarm/market.py and monitor.py) and are released with the code.

## LLM Usage Statement

We used Claude (Anthropic) to help design the experiment, write the code, run the analysis and draft the text. The author ran all experiments, and all reported numbers come from the released logs through the released analysis script; every number in the paper was checked against results/results.json and every reference against its source page.
