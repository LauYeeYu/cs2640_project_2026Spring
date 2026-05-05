"""Unit tests for Phase 5: model-controlled recall tool.

Verifies:
* Policy tags mementos with [memento_id=mem-N] when recall_tool_enabled.
* `_recall_table` has the obs text under that id.
* `get_recall_tool` returns a Tool whose handler calls queue_recall_fn
  with the looked-up obs text.
* Errors gracefully on unknown ids.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Avoid importing the vllm-heavy adapter chain.
os.environ.setdefault("PAPER2_TEST_NO_VLLM", "1")

from studies.lifetime_cost.paper2.policy.memento_policy import MementoPolicy


def _stub_writer():
    w = MagicMock()
    w.write.return_value = ("SUMMARY", MagicMock(input_tokens=10, output_tokens=5, cost_usd=0.001))
    return w


def test_disabled_policy_does_not_tag():
    p = MementoPolicy(recall_tool_enabled=False, writer=_stub_writer())
    assert p.get_recall_tool(lambda s: None) is None
    assert p.get_system_prompt_addendum() is None
    assert p._recall_table == {}


def test_enabled_policy_returns_tool_and_addendum():
    p = MementoPolicy(recall_tool_enabled=True, writer=_stub_writer())
    queue_recall_fn = MagicMock(return_value="obs:fakehash")
    tool = p.get_recall_tool(queue_recall_fn)
    assert tool is not None
    assert tool.name == "recall"
    assert "memento_id" in tool.parameters["properties"]
    assert tool.parameters["required"] == ["memento_id"]
    assert "memento_id" in (p.get_system_prompt_addendum() or "")


def test_recall_handler_round_trip():
    p = MementoPolicy(recall_tool_enabled=True, writer=_stub_writer())
    # Simulate the policy having tagged a memento during compaction.
    p._recall_table["mem-3"] = "FULL OBS TEXT"
    queue_recall_fn = MagicMock(return_value="obs:abc")
    tool = p.get_recall_tool(queue_recall_fn)

    res = tool.fn({"memento_id": "mem-3"})
    queue_recall_fn.assert_called_once_with("FULL OBS TEXT")
    assert "OK" in res
    assert "mem-3" in res


def test_recall_handler_unknown_id():
    p = MementoPolicy(recall_tool_enabled=True, writer=_stub_writer())
    p._recall_table["mem-1"] = "X"
    queue_recall_fn = MagicMock()
    tool = p.get_recall_tool(queue_recall_fn)

    res = tool.fn({"memento_id": "mem-99"})
    queue_recall_fn.assert_not_called()
    assert "[recall error]" in res
    assert "mem-1" in res  # known list shown


def test_recall_handler_missing_id():
    p = MementoPolicy(recall_tool_enabled=True, writer=_stub_writer())
    tool = p.get_recall_tool(lambda s: None)
    res = tool.fn({})
    assert "[recall error]" in res


def test_compaction_tags_memento():
    """maybe_compact should prefix [memento_id=mem-N] and register obs in table."""
    from studies.lifetime_cost.pipeline.policies.base import CompactionContext
    from studies.lifetime_cost.pipeline.tokenization import get_tokenizer

    tokenizer = get_tokenizer("tiktoken:cl100k_base")
    big_obs = "X" * 4000  # 4KB, comfortably above min_obs_chars=500

    # Build messages: system, user, assistant_with_tool_call, tool_obs (big), tool_obs (big), user
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "function": {"name": "read", "arguments": {}}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": big_obs},
        {"role": "assistant", "tool_calls": [
            {"id": "c2", "function": {"name": "read", "arguments": {}}}
        ]},
        {"role": "tool", "tool_call_id": "c2", "content": big_obs},
        {"role": "assistant", "tool_calls": [
            {"id": "c3", "function": {"name": "read", "arguments": {}}}
        ]},
        {"role": "tool", "tool_call_id": "c3", "content": big_obs},
        {"role": "user", "content": "u2"},
    ]

    p = MementoPolicy(
        min_obs_chars=300,
        recall_tool_enabled=True,
        # Force trigger: tiny budget so estimated tokens > trigger.
        trigger_ratio=0.01,
        target_ratio=0.001,  # Force fire on all candidates
        writer=_stub_writer(),
    )
    ctx = CompactionContext(step=0, budget=100, hard_budget=200,
                            tokenizer=tokenizer, summarizer=lambda x: ("", 0, 0, 0))
    out, evt = p.maybe_compact(msgs, ctx)
    assert evt is not None, "compaction should have fired"

    # Two of the three big tool msgs should be memento'd (last 2 are kept,
    # so msg[3] gets tagged; recent_skip=2 means we skip last 2 tool msgs).
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    tagged = [m for m in tool_msgs if m.get("memento")]
    assert len(tagged) >= 1, "at least one tool msg should be memento'd"
    for m in tagged:
        assert m["memento"].startswith("[memento_id=mem-"), m["memento"][:50]
        assert "memento_id" in m
        # Round-trip via the recall table
        assert p._recall_table[m["memento_id"]] == big_obs


if __name__ == "__main__":
    test_disabled_policy_does_not_tag()
    test_enabled_policy_returns_tool_and_addendum()
    test_recall_handler_round_trip()
    test_recall_handler_unknown_id()
    test_recall_handler_missing_id()
    test_compaction_tags_memento()
    print("OK — all 6 tests passed")
