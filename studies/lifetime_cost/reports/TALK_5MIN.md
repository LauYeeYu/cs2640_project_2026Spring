# AdaptiveCache — 5-minute talk script

**Total: 5 minutes. 5 slides. Speaker notes are what to *say*, in the rhythm and length you'd actually deliver.**

---

## Slide 1 — Hook + thesis (0:00 → 0:50)

**Visual:** title slide. One sentence: *"Long-running LLM agents pay a cliff tax. Can we beat it with smart compaction?"* Below it: `fig6_project_arc.png` (small).

**Say:**

> LLM agents that run for hundreds of turns spend most of their dollars on cached prefix reuse. Cached tokens cost ten percent of uncached. So when an agent's context grows, the natural fix is compaction — drop or summarize old tool outputs to keep the prompt small. But compaction has a hidden cost: any change to the prefix invalidates the downstream cache. The next API call gets re-billed at ten times the rate. We call this the cliff tax.
>
> Our project asked the simplest, sharpest version of the question: **can a training-free heuristic compaction policy Pareto-dominate `none` — no compaction at all — on real agent benchmarks?** I'll tell you what we built, where it wins, where it loses, and the one mechanism finding that surprised us.

---

## Slide 2 — What we built + the cliff tax (0:50 → 1:50)

**Visual:** `fig5_cliff_amplification.png`.

**Say:**

> We built a complete lifetime-cost evaluation harness. Eight policies — naive summary, prefix-preserving, microcompact, position-aware, an LLM reorganizer, our novel action-graph supersession family. Two benchmarks: SWE-bench Lite live mode without docker, and τ-bench retail with a new multi-customer chain harness we built. Two model classes: Qwen3-30B-A3B local, and Haiku 4.5 over the API.
>
> Most importantly we built a *real* test validator — replay the agent's edits onto a fresh checkout, apply SWE-bench's test patch, run pytest in a Python 3.9 venv per task. That found that the line-overlap oracle the field uses overcounts correctness by **30 percent**.
>
> The cliff tax, measured on real Haiku trajectories, is **ten to fifteen cents per compaction event**. To break even you need at least five steps after the cliff. That's the number that determines everything.

---

## Slide 3 — Where compaction DOES win (1:50 → 2:50)

**Visual:** `fig7_compaction_wins.png`.

**Say:**

> Here are the wins. Left panel: on Haiku at four SWE-bench Lite tasks, our novel **action-graph supersession** policy — `consumption_evict` — resolves three out of four at one dollar three cents per task. `none` resolves only two out of four, and pays one dollar twenty. **More resolutions at lower cost.** The signal is real: it tags tool observations as consumed when the agent's *own* later tool calls make them obviously stale — `read_file` consumed by `edit_file`, search results consumed by following the lead, run_tests cascading. No prior compaction system uses agent tool-graph semantics this way.
>
> Right panel: on the weaker Qwen3 agent, `none` scores zero out of four — it falls into false-submit and context-overflow loops at temperature point five. `smart_evict` scores two out of four. Compaction *prevents* catastrophic failure. **Below an agent-quality threshold, compaction is necessary, not just optimization.**

---

## Slide 4 — The mechanism finding (2:50 → 3:50)

**Visual:** `fig3_placeholder_ablation.png`.

**Say:**

> Here's the surprising mechanism, which replicates across two seeds. We took the same `consumption_evict` signal and ablated only the placeholder — what to leave behind when we evict.
>
> Plain placeholder: `[evicted: ~N tokens, consumed by later edit]`. Forces the agent to re-fetch if it wants to act.
>
> Facts placeholder: same plus a one-line summary of the file's function definitions. *More* information, kept around at low token cost.
>
> Outline placeholder: location breadcrumbs, `lineN: header`, plus an explicit "re-read to act" instruction.
>
> On `pytest-7490`, the only validatable task where the four policies' trajectories diverged: plain and outline both resolve. Facts **fails** — and burns seven to eighteen edit-revert iterations on the wrong function. The function-name list anchored the agent on surface structure. **Less is more, because any structural hint can become an anchor.** That's a clean ablation, a real mechanism, and a publishable design lesson.

---

## Slide 5 — The cliff wins at scale, and what's next (3:50 → 5:00)

**Visual:** `fig1_pareto_swebench.png` left half, `fig2_cost_decomposition.png` right half — or just fig1.

**Say:**

> But at N=10 on Haiku, the cliff tax wins. Every compaction policy ties `none` on resolve and costs **two to two-point-four times more**. The cost decomposition shows why: input-uncached dominates because every cliff re-bills the prefix. Even `consumption_evict`, which fires only when something is *provably* stale, loses on cost.
>
> One more negative-but-constructive finding: on τ-bench retail with our multi-customer chain harness, even at chain-size ten where max prompt reaches fifty-four thousand tokens — well past the trigger — `consumption_evict` fires **zero** times. Why? Its supersession rules are coding-specific. Retail tools don't match `read_file` and `edit_file`. Action-graph supersession isn't a one-shot policy — it's a *family parameterized by per-domain rules*.
>
> So: heuristic compaction loses to no-compaction on a strong agent — that's the headline negative result. But the placeholder-design ablation gives us a publishable positive nugget. And the negative result motivates Paper 2 directly: the cliff is intrinsic to byte-level eviction. The way around it is to **keep the bytes** — KV-pointer recall — and that's what's next.

**End on:** *"Thank you. Questions?"*

---

## Pacing checklist

| Slide | Target | Words |
|---|---|---:|
| 1 — Hook | 50s | ~125 |
| 2 — Setup + cliff | 60s | ~150 |
| 3 — Where it wins | 60s | ~155 |
| 4 — Placeholder mechanism | 60s | ~150 |
| 5 — Negative result + next | 70s | ~175 |
| **Total** | **~5:00** | **~755** |

(Adjust by trimming the supersession rule list in slide 3 or the placeholder definitions in slide 4 if running long.)

## If you have only 3 minutes

Cut slides 2 and 5's preamble. Open with slide 3 (where it wins), use slide 4 as the "even when it loses, here's the mechanism" pivot, and close with the one-line cliff-tax negative result + Paper 2 motivation. The mechanism finding is what to protect.
