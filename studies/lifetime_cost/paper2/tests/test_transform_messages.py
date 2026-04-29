"""Unit tests for the message-transform logic. No GPU/vLLM required."""

from studies.lifetime_cost.paper2.adapters.memento_vllm import (
    transform_messages,
    wrap_tool_message_for_masking,
    SUMMARY_START_STR,
    SUMMARY_END_STR,
)


def test_wrap_no_memento():
    out = wrap_tool_message_for_masking("hello", None)
    assert out == "<tool_response>\nhello\n</tool_response>"


def test_wrap_with_memento():
    out = wrap_tool_message_for_masking("file contents", "summary text")
    expected = (
        "<tool_response>\nfile contents\n</tool_response>"
        f"{SUMMARY_START_STR}summary text{SUMMARY_END_STR}"
    )
    assert out == expected


def test_transform_passes_non_tool_messages():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    out = transform_messages(msgs)
    assert out == msgs  # unchanged


def test_transform_tool_no_memento():
    msgs = [
        {"role": "user", "content": "fix"},
        {"role": "tool", "content": "obs", "tool_call_id": "a"},
    ]
    out = transform_messages(msgs)
    assert len(out) == 2
    assert out[0] == msgs[0]
    assert out[1]["role"] == "user"
    assert out[1]["content"] == "<tool_response>\nobs\n</tool_response>"


def test_transform_tool_with_memento():
    msgs = [
        {"role": "tool", "content": "the obs", "memento": "summary"},
    ]
    out = transform_messages(msgs)
    assert out[0]["role"] == "user"
    body = out[0]["content"]
    assert body.startswith("<tool_response>\nthe obs\n</tool_response>")
    assert body.endswith(f"{SUMMARY_START_STR}summary{SUMMARY_END_STR}")


def test_summary_tokens_are_qwen3_reserved():
    # Sanity: these are the FIM tokens in Qwen3-4B vocab (151659/151660)
    assert SUMMARY_START_STR == "<|fim_prefix|>"
    assert SUMMARY_END_STR == "<|fim_middle|>"


if __name__ == "__main__":
    test_wrap_no_memento()
    test_wrap_with_memento()
    test_transform_passes_non_tool_messages()
    test_transform_tool_no_memento()
    test_transform_tool_with_memento()
    test_summary_tokens_are_qwen3_reserved()
    print("ALL UNIT TESTS PASSED")
