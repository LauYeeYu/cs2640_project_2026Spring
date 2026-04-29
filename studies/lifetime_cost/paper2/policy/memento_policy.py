"""Memento policy — lazy compaction of tool messages.

The policy waits until total context exceeds a trigger ratio of the
budget, then generates mementos for the OLDEST un-memento'd tool
messages until projected context drops below a target ratio. This
mirrors the smart_evict / prefix_preserving trigger logic from Paper 1.

Why lazy: an eager "memento every big obs" is wasteful — most agent
steps don't need memento at all (the agent moves on; the obs gets
naturally pushed back by new content but doesn't have to leave KV
since the budget isn't hit). Each memento is a Haiku API call (~3-4s
latency, ~$0.005). Generating one per turn dominates wall-clock and
adds up over an eval.

Lazy generation keeps the same structural property (latest big obs
gets evicted via masking when context fills up) but does so only when
needed.

The policy is also the natural place to record memento-generation
cost as a CompactionEvent so analysis treats it the same as any other
compaction.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

# Pipeline imports (relative — works in both worktree and editable installs).
from ...pipeline.policies.base import CompactionContext, CompactionPolicy
from ...pipeline.types import CompactionEvent
from ...pipeline.tokenization import count_messages

from ..memento_writer import HaikuMementoWriter


def _tool_obs(msg: Dict[str, Any]) -> Optional[str]:
    """Return the obs text if the message is a tool message, else None."""
    if msg.get("role") != "tool":
        return None
    c = msg.get("content")
    return c if isinstance(c, str) else None


class MementoPolicy(CompactionPolicy):
    """Lazy memento generator — tags oldest tool obs only when budget bites.

    Args:
        min_obs_chars: Don't memento obs smaller than this — too small to
            be worth the API call + evict cost.
        trigger_ratio: Fire when total tokens exceed `trigger_ratio * budget`.
        target_ratio: Stop firing once tokens estimated to fall below
            `target_ratio * budget` after generating mementos.
        max_obs_chars: Truncate large obs sent to Haiku.
        writer: Inject a custom writer (useful for tests).
    """

    name = "memento"

    def __init__(
        self,
        *,
        min_obs_chars: int = 500,
        trigger_ratio: float = 0.85,
        target_ratio: float = 0.55,
        writer: Optional[HaikuMementoWriter] = None,
        memento_model: str = "claude-haiku-4-5",
        max_obs_chars: int = 8000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._min_obs_chars = min_obs_chars
        self._trigger_ratio = trigger_ratio
        self._target_ratio = target_ratio
        self._writer = writer or HaikuMementoWriter(
            model=memento_model, max_obs_chars=max_obs_chars
        )

    def _estimate_tokens(self, messages: List[Dict[str, Any]], ctx: CompactionContext) -> int:
        """Use the runner-provided tokenizer for total context estimate."""
        return count_messages(messages, ctx.tokenizer)

    def maybe_compact(
        self,
        messages: List[Dict[str, Any]],
        ctx: CompactionContext,
    ) -> Tuple[List[Dict[str, Any]], Optional[CompactionEvent]]:
        # 1. Trigger check — total tokens must exceed trigger threshold.
        total_tok = self._estimate_tokens(messages, ctx)
        trigger = int(self._trigger_ratio * ctx.budget)
        if total_tok < trigger:
            return messages, None

        # 2. Find tool messages that are big enough and not yet memento'd,
        #    in OLDEST-FIRST order (we evict the oldest aggressively-bigger
        #    obs first; recent obs stay visible).
        # Heuristic: average ~4 chars/token. Skip recent N=2 tool msgs to
        # leave the agent's most recent context intact.
        candidates: List[int] = []
        recent_skip = 2
        # walk in chronological order; collect indices, then drop the last N
        all_tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        keep_recent_set = set(all_tool_indices[-recent_skip:]) if all_tool_indices else set()
        for i in all_tool_indices:
            if i in keep_recent_set:
                continue
            m = messages[i]
            if m.get("memento"):
                continue
            obs = m.get("content", "")
            if not isinstance(obs, str) or len(obs) < self._min_obs_chars:
                continue
            candidates.append(i)

        if not candidates:
            return messages, None

        # 3. Generate mementos in oldest-first order, until projected context
        #    falls below target_ratio * budget. Each generated memento
        #    "saves" roughly (obs_tokens - memento_tokens) ≈ 90% of obs size.
        target = int(self._target_ratio * ctx.budget)
        t0 = time.perf_counter()
        in_toks_total = 0
        out_toks_total = 0
        cost_total = 0.0
        bytes_tagged = 0
        projected_total = total_tok

        for i in candidates:
            if projected_total < target:
                break
            msg = messages[i]
            tool_name, tool_args = _trace_tool_call(messages, i)
            text, usage = self._writer.write(
                obs=msg["content"],
                tool_name=tool_name,
                tool_args=tool_args,
            )
            msg["memento"] = text
            in_toks_total += usage.input_tokens
            out_toks_total += usage.output_tokens
            cost_total += usage.cost_usd
            bytes_tagged += len(msg["content"])
            # Project size reduction: obs goes from ~len/4 tokens to
            # roughly len(memento)/4 tokens once rendered as inline plain text.
            obs_tok = len(msg["content"]) // 4
            mem_tok = len(text) // 4
            projected_total -= max(0, obs_tok - mem_tok)

        if in_toks_total == 0:
            # Nothing fired (target already met by the time we ran)
            return messages, None

        wall_ms = int((time.perf_counter() - t0) * 1000)
        evt = CompactionEvent(
            step=ctx.step,
            policy=self.name,
            msgs_before=len(messages),
            msgs_after=len(messages),  # messages unchanged at the list level
            tokens_before=0,           # filled by analysis if needed
            tokens_after=0,
            compaction_input_cached_tokens=0,
            compaction_input_uncached_tokens=in_toks_total,
            compaction_output_tokens=out_toks_total,
            compaction_call_tokens=in_toks_total + out_toks_total,
            wallclock_ms=wall_ms,
        )
        return messages, evt


def _trace_tool_call(
    messages: List[Dict[str, Any]], tool_msg_idx: int
) -> Tuple[str, Dict[str, Any]]:
    """Look back for the matching tool_call by tool_call_id."""
    target_id = messages[tool_msg_idx].get("tool_call_id")
    for j in range(tool_msg_idx - 1, -1, -1):
        m = messages[j]
        if m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls") or []
        for tc in tcs:
            if tc.get("id") == target_id:
                fn = tc.get("function") or {}
                return fn.get("name", "unknown"), fn.get("arguments") or {}
    return "unknown", {}
