"""Tests for server module: adapter ID extraction, chat formatting."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mola.server import BASE_MODEL_SELECTOR, Message, _format_chat, extract_adapter_id


# --- Helpers ---


def _make_mola_model(adapter_names: list[str] | None = None):
    """Create a mock MOLAModel with optional adapter names."""
    model = MagicMock()
    model.model_path = "test-model"
    model.adapter_manager.adapters = {}
    model.tokenizer = None

    def get_adapter(name):
        if adapter_names and name in adapter_names:
            return MagicMock()
        return None

    model.adapter_manager.get = MagicMock(side_effect=get_adapter)
    model.list_adapters.return_value = [
        {"name": n} for n in (adapter_names or [])
    ]
    return model


# --- extract_adapter_id ---


class TestExtractAdapterId:
    def test_direct_adapter_name(self):
        model = _make_mola_model(["code-assist"])
        assert extract_adapter_id("code-assist", model) == "code-assist"

    def test_base_keyword(self):
        """The reserved 'base' keyword always selects the base model."""
        model = _make_mola_model(["code-assist"])
        assert extract_adapter_id(BASE_MODEL_SELECTOR, model) is None

    def test_base_slash_adapter(self):
        model = _make_mola_model(["code-assist"])
        assert extract_adapter_id("qwen/code-assist", model) == "code-assist"

    def test_model_path_accepted_as_base(self):
        """The exact model_path (even with '/') must be accepted as base model."""
        model = _make_mola_model(["code-assist"])
        model.model_path = "mlx-community/Qwen3.5-35B-A3B-4bit"
        assert extract_adapter_id("mlx-community/Qwen3.5-35B-A3B-4bit", model) is None

    def test_unknown_bare_name_raises(self):
        """A bare name that is neither an adapter nor 'base' must error."""
        model = _make_mola_model(["sql"])
        with pytest.raises(ValueError, match="Unknown model selector"):
            extract_adapter_id("sqll", model)

    def test_slash_unknown_suffix_raises(self):
        """base/unknown must be an explicit error."""
        model = _make_mola_model(["code-assist"])
        with pytest.raises(ValueError, match="not a loaded adapter"):
            extract_adapter_id("qwen/typo-name", model)

    def test_unknown_hf_id_raises(self):
        """An HF-style ID that does NOT match the loaded model_path is an error."""
        model = _make_mola_model(["code-assist"])
        model.model_path = "mlx-community/Qwen3.5-35B-A3B-4bit"
        with pytest.raises(ValueError):
            extract_adapter_id("mlx-community/Llama-3-8B-4bit", model)

    def test_nested_slash_raises(self):
        model = _make_mola_model(["code-assist"])
        with pytest.raises(ValueError):
            extract_adapter_id("org/repo/sub", model)

    def test_slash_with_known_suffix(self):
        model = _make_mola_model(["code-assist"])
        assert extract_adapter_id("mymodel/code-assist", model) == "code-assist"

    def test_prefix_is_ignored(self):
        """The prefix in 'prefix/adapter' is not validated — suffix-based selector."""
        model = _make_mola_model(["rust"])
        assert extract_adapter_id("banana/rust", model) == "rust"
        assert extract_adapter_id("anything-goes/rust", model) == "rust"

    def test_empty_string_raises(self):
        """Empty string is an unknown bare name — error."""
        model = _make_mola_model([])
        with pytest.raises(ValueError):
            extract_adapter_id("", model)

    def test_error_message_lists_valid_values(self):
        """The error message should help the user by listing valid selectors."""
        model = _make_mola_model(["rust", "sql"])
        with pytest.raises(ValueError, match="rust") as exc_info:
            extract_adapter_id("rsut", model)
        msg = str(exc_info.value)
        assert "base" in msg
        assert "sql" in msg


# --- _format_chat ---


class TestFormatChat:
    def test_chatml_fallback(self):
        msgs = [Message(role="user", content="Hello")]
        result = _format_chat(msgs)
        assert "<|im_start|>user" in result
        assert "Hello" in result
        assert result.endswith("assistant\n")

    def test_uses_tokenizer_template(self):
        msgs = [Message(role="user", content="Hi")]
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<TEMPLATE>Hi</TEMPLATE>"
        result = _format_chat(msgs, tokenizer=tokenizer)
        assert result == "<TEMPLATE>Hi</TEMPLATE>"
        tokenizer.apply_chat_template.assert_called_once()

    def test_falls_back_on_tokenizer_error(self):
        msgs = [Message(role="user", content="Hi")]
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.side_effect = Exception("no template")
        result = _format_chat(msgs, tokenizer=tokenizer)
        assert "<|im_start|>" in result

    def test_no_tokenizer(self):
        msgs = [Message(role="system", content="Be nice"), Message(role="user", content="Hi")]
        result = _format_chat(msgs, tokenizer=None)
        assert "<|im_start|>system" in result
        assert "<|im_start|>user" in result
