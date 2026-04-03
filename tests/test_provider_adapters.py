"""
Unit tests for the OpenAI, Google, and Anthropic provider adapters.

All external SDK calls are mocked so these tests run without any installed
provider package or live API key.
"""

from __future__ import annotations

import base64
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "view-description": {"type": "string"},
        "reasoning": {"type": "string"},
        "action": {"type": "string", "enum": ["w", "a", "s", "d", "done"]},
        "summary": {"type": "string"},
        "selected_cube": {"type": "string"},
        "identification": {
            "type": "object",
            "properties": {
                "my_color": {"type": "string"},
                "source": {"type": "string"},
                "confidence": {"type": "string"},
            },
            "required": ["my_color", "source", "confidence"],
            "additionalProperties": False,
        },
    },
    "required": ["view-description", "reasoning", "action", "summary", "selected_cube", "identification"],
    "additionalProperties": False,
}

_SYSTEM_MSG = {"role": "system", "content": "You are a robot."}
_USER_TEXT_MSG = {"role": "user", "content": "Describe the scene."}


def _fake_image_path(tmp_path: Path) -> str:
    """Write a minimal 1-byte JPEG-ish file and return its path as a string."""
    p = tmp_path / "obs.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0")  # JPEG magic bytes
    return str(p)


# ===========================================================================
# OpenAI adapter tests
# ===========================================================================


class TestOpenAITextConversion:
    def test_system_becomes_input_text_item(self, tmp_path):
        """System message stays a system-role message in Responses API input."""
        from providers.openai import _build_input_messages

        items = _build_input_messages([_SYSTEM_MSG])
        assert items == [{
            "role": "system",
            "content": [{"type": "input_text", "text": "You are a robot."}],
        }]

    def test_user_text_becomes_input_text_item(self, tmp_path):
        from providers.openai import _build_input_messages

        items = _build_input_messages([_USER_TEXT_MSG])
        assert items == [{
            "role": "user",
            "content": [{"type": "input_text", "text": "Describe the scene."}],
        }]


class TestOpenAIImageConversion:
    def test_image_path_becomes_input_image_item(self, tmp_path):
        from providers.openai import _build_input_messages

        img = _fake_image_path(tmp_path)
        msg = {"role": "user", "content": "Look.", "images": [img]}
        items = _build_input_messages([msg])

        image_items = [i for i in items[0]["content"] if i["type"] == "input_image"]
        assert len(image_items) == 1
        url = image_items[0]["image_url"]
        assert url.startswith("data:image/jpeg;base64,")
        # Verify the base64 payload decodes back to the original bytes
        b64_part = url.split(",", 1)[1]
        assert base64.b64decode(b64_part) == Path(img).read_bytes()

    def test_unreadable_image_is_skipped(self, tmp_path):
        from providers.openai import _build_input_messages

        msg = {"role": "user", "content": "Look.", "images": ["/nonexistent/path.jpg"]}
        items = _build_input_messages([msg])
        # Only the text item should remain
        assert all(i["type"] != "input_image" for i in items[0]["content"])


class TestOpenAISchemaHandoff:
    def test_schema_passed_to_response_format(self, monkeypatch, tmp_path):
        """chat() passes output_schema into the json_schema format parameter."""
        captured: dict = {}

        mock_response = MagicMock()
        mock_response.output_text = '{"action": "w"}'

        mock_client = MagicMock()
        mock_client.responses.create.side_effect = lambda **kwargs: (
            captured.update(kwargs) or mock_response
        )

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict(sys.modules, {"openai": mock_openai}):
            import providers.openai as openai_provider

            monkeypatch.setenv("OPENAI_API_KEY", "test-key")
            content, thinking = openai_provider.chat(
                model="gpt-5",
                messages=[_USER_TEXT_MSG],
                output_schema=_SCHEMA,
            )

        assert thinking is None
        assert content == '{"action": "w"}'
        text_format = captured["text"]["format"]
        assert text_format["type"] == "json_schema"
        assert text_format["json_schema"]["schema"] is _SCHEMA
        assert captured["input"][0]["role"] == "user"


# ===========================================================================
# Google adapter tests
# ===========================================================================


class TestGoogleTextConversion:
    def test_system_message_becomes_system_instruction(self, monkeypatch, tmp_path):
        """System message is extracted as system_instruction; not in contents."""
        captured: dict = {}

        mock_response = MagicMock()
        mock_response.text = '{"action": "w"}'

        mock_models = MagicMock()
        mock_models.generate_content.side_effect = lambda **kwargs: (
            captured.update(kwargs) or mock_response
        )

        mock_client_instance = MagicMock()
        mock_client_instance.models = mock_models

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client_instance

        mock_types = MagicMock()
        mock_types.Part.from_text.side_effect = lambda text: {"_text": text}
        mock_types.Content.side_effect = lambda role, parts: {"role": role, "parts": parts}

        def fake_config(**kwargs):
            return {"_config": kwargs}

        mock_types.GenerateContentConfig.side_effect = fake_config

        google_pkg = types.ModuleType("google")
        google_genai_pkg = types.ModuleType("google.genai")
        google_genai_pkg.Client = mock_genai.Client
        google_genai_pkg.types = mock_types

        with patch.dict(sys.modules, {"google": google_pkg, "google.genai": google_genai_pkg}):
            import providers.google as google_provider

            monkeypatch.setenv("GEMINI_API_KEY", "test-key")
            google_provider.chat(
                model="gemini-2.5-flash",
                messages=[_SYSTEM_MSG, _USER_TEXT_MSG],
                output_schema=_SCHEMA,
            )

        cfg = captured["config"]["_config"]
        assert cfg["system_instruction"] == "You are a robot."

    def test_user_text_becomes_part(self, monkeypatch, tmp_path):
        """User text content is passed to Part.from_text."""
        text_parts: list = []

        mock_response = MagicMock()
        mock_response.text = '{"action": "w"}'

        mock_models = MagicMock()
        mock_models.generate_content.return_value = mock_response

        mock_client_instance = MagicMock()
        mock_client_instance.models = mock_models

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client_instance

        mock_types = MagicMock()
        mock_types.Part.from_text.side_effect = lambda text: text_parts.append(text) or {"_text": text}
        mock_types.Content.side_effect = lambda role, parts: {"role": role, "parts": parts}
        mock_types.GenerateContentConfig.side_effect = lambda **kw: {"_config": kw}

        google_pkg = types.ModuleType("google")
        google_genai_pkg = types.ModuleType("google.genai")
        google_genai_pkg.Client = mock_genai.Client
        google_genai_pkg.types = mock_types

        with patch.dict(sys.modules, {"google": google_pkg, "google.genai": google_genai_pkg}):
            import providers.google as google_provider

            monkeypatch.setenv("GEMINI_API_KEY", "test-key")
            google_provider.chat(
                model="gemini-2.5-flash",
                messages=[_USER_TEXT_MSG],
                output_schema=_SCHEMA,
            )

        assert "Describe the scene." in text_parts


class TestGoogleImageConversion:
    def test_image_path_becomes_part_from_bytes(self, monkeypatch, tmp_path):
        """Image path is read and passed to Part.from_bytes with correct mime type."""
        bytes_calls: list = []

        mock_response = MagicMock()
        mock_response.text = '{"action": "w"}'

        mock_models = MagicMock()
        mock_models.generate_content.return_value = mock_response

        mock_client_instance = MagicMock()
        mock_client_instance.models = mock_models

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client_instance

        mock_types = MagicMock()
        mock_types.Part.from_bytes.side_effect = lambda data, mime_type: bytes_calls.append(
            {"data": data, "mime_type": mime_type}
        ) or {"_bytes": True}
        mock_types.Part.from_text.side_effect = lambda text: {"_text": text}
        mock_types.Content.side_effect = lambda role, parts: {"role": role, "parts": parts}
        mock_types.GenerateContentConfig.side_effect = lambda **kw: {"_config": kw}

        google_pkg = types.ModuleType("google")
        google_genai_pkg = types.ModuleType("google.genai")
        google_genai_pkg.Client = mock_genai.Client
        google_genai_pkg.types = mock_types

        img = _fake_image_path(tmp_path)
        msg = {"role": "user", "content": "Look.", "images": [img]}

        with patch.dict(sys.modules, {"google": google_pkg, "google.genai": google_genai_pkg}):
            import providers.google as google_provider

            monkeypatch.setenv("GEMINI_API_KEY", "test-key")
            google_provider.chat(
                model="gemini-2.5-flash",
                messages=[msg],
                output_schema=_SCHEMA,
            )

        assert len(bytes_calls) == 1
        assert bytes_calls[0]["data"] == Path(img).read_bytes()
        assert bytes_calls[0]["mime_type"] == "image/jpeg"


class TestGoogleSchemaHandoff:
    def test_response_schema_received_by_config(self, monkeypatch, tmp_path):
        """output_schema (with propertyOrdering injected) is passed to GenerateContentConfig."""
        config_calls: list = []

        mock_response = MagicMock()
        mock_response.text = '{"action": "w"}'

        mock_models = MagicMock()
        mock_models.generate_content.return_value = mock_response

        mock_client_instance = MagicMock()
        mock_client_instance.models = mock_models

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client_instance

        mock_types = MagicMock()
        mock_types.Part.from_text.side_effect = lambda text: {"_text": text}
        mock_types.Content.side_effect = lambda role, parts: {"role": role, "parts": parts}
        mock_types.GenerateContentConfig.side_effect = lambda **kw: config_calls.append(kw) or kw

        google_pkg = types.ModuleType("google")
        google_genai_pkg = types.ModuleType("google.genai")
        google_genai_pkg.Client = mock_genai.Client
        google_genai_pkg.types = mock_types

        with patch.dict(sys.modules, {"google": google_pkg, "google.genai": google_genai_pkg}):
            import providers.google as google_provider

            monkeypatch.setenv("GEMINI_API_KEY", "test-key")
            google_provider.chat(
                model="gemini-2.5-flash",
                messages=[_USER_TEXT_MSG],
                output_schema=_SCHEMA,
            )

        assert config_calls, "GenerateContentConfig was never called"
        schema_passed = config_calls[0]["response_json_schema"]
        # propertyOrdering should have been injected into every object node
        assert "propertyOrdering" in schema_passed
        assert "propertyOrdering" in schema_passed["properties"]["identification"]

    def test_property_ordering_injected_recursively(self):
        from providers.google import _add_property_ordering

        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}, "y": {"type": "string"}},
                    "required": ["x"],
                    "additionalProperties": False,
                },
            },
        }
        result = _add_property_ordering(schema)
        assert result["propertyOrdering"] == ["a", "b"]
        assert result["properties"]["b"]["propertyOrdering"] == ["x", "y"]
        # Original schema is not mutated
        assert "propertyOrdering" not in schema


# ===========================================================================
# Anthropic adapter tests
# ===========================================================================


class TestAnthropicTextConversion:
    def test_system_message_becomes_top_level_system(self, tmp_path):
        from providers.anthropic import _build_request

        system_prompt, anthropic_messages = _build_request([_SYSTEM_MSG, _USER_TEXT_MSG])

        assert system_prompt == "You are a robot."
        assert anthropic_messages == [{
            "role": "user",
            "content": [{"type": "text", "text": "Describe the scene."}],
        }]

    def test_user_text_becomes_text_block(self, tmp_path):
        from providers.anthropic import _build_request

        _, anthropic_messages = _build_request([_USER_TEXT_MSG])

        assert anthropic_messages == [{
            "role": "user",
            "content": [{"type": "text", "text": "Describe the scene."}],
        }]


class TestAnthropicImageConversion:
    def test_image_path_becomes_base64_image_block(self, tmp_path):
        from providers.anthropic import _build_request

        img = _fake_image_path(tmp_path)
        msg = {"role": "user", "content": "Look.", "images": [img]}

        _, anthropic_messages = _build_request([msg])
        image_block = anthropic_messages[0]["content"][0]

        assert image_block["type"] == "image"
        assert image_block["source"]["type"] == "base64"
        assert image_block["source"]["media_type"] == "image/jpeg"
        assert base64.b64decode(image_block["source"]["data"]) == Path(img).read_bytes()

    def test_unreadable_image_is_skipped(self, tmp_path):
        from providers.anthropic import _build_request

        msg = {"role": "user", "content": "Look.", "images": ["/nonexistent/path.jpg"]}

        _, anthropic_messages = _build_request([msg])
        assert anthropic_messages == [{
            "role": "user",
            "content": [{"type": "text", "text": "Look."}],
        }]


class TestAnthropicSchemaHandoff:
    def test_schema_passed_to_output_config(self, monkeypatch, tmp_path):
        captured: dict = {}

        mock_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"action": "w"}')]
        )
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = lambda **kwargs: (
            captured.update(kwargs) or mock_response
        )

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            import providers.anthropic as anthropic_provider

            monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
            content, thinking = anthropic_provider.chat(
                model="claude-sonnet-4-5",
                messages=[_SYSTEM_MSG, _USER_TEXT_MSG],
                output_schema=_SCHEMA,
            )

        assert thinking is None
        assert content == '{"action": "w"}'
        assert captured["system"] == "You are a robot."
        assert captured["output_config"]["format"]["type"] == "json_schema"
        assert captured["output_config"]["format"]["schema"] is _SCHEMA


class TestDispatchCanonicalization:
    def test_claude_alias_routes_to_anthropic(self, monkeypatch):
        import llm.dispatch as chat_module

        called = {}

        def fake_chat(**kwargs):
            called.update(kwargs)
            return '{"action": "w"}', None

        monkeypatch.setattr("providers.anthropic.chat", fake_chat)

        content, thinking = chat_module.chat(
            client_type="claude",
            model="claude-sonnet-4-5",
            messages=[_USER_TEXT_MSG],
            output_schema=_SCHEMA,
        )

        assert thinking is None
        assert content == '{"action": "w"}'
        assert called["model"] == "claude-sonnet-4-5"

    def test_canonicalize_client_type_maps_claude(self):
        from llm.dispatch import canonicalize_client_type

        assert canonicalize_client_type("claude") == "anthropic"
        assert canonicalize_client_type("anthropic") == "anthropic"


class TestControllerCanonicalization:
    def test_controller_uses_canonical_anthropic_client_for_paths_and_api_key(self, monkeypatch, tmp_path):
        from runtime.controller import Controller

        class _FakeEnv:
            position = (0.0, 0.0)
            direction = "N"
            object_positions_2d = {}

            def bind_keys(self, callback):
                return None

            def close(self):
                return None

        class _FakeLogger:
            def __init__(self, output_file):
                self.output_file = output_file

            def reset(self):
                return None

        monkeypatch.setattr("runtime.controller._create_env", lambda **kwargs: _FakeEnv())
        monkeypatch.setattr("runtime.controller.Logger", _FakeLogger)

        controller = Controller(
            nav_mode="claude-sonnet-4-5",
            client_type="claude",
            anthropic_api_key="anthropic-key",
            output_dir=str(tmp_path / "outputs"),
            env_type="mujoco",
            mujoco_headless=True,
        )

        try:
            assert controller.client_type == "anthropic"
            assert controller.api_key == "anthropic-key"
            assert "/anthropic/" in controller.output_dir.replace("\\", "/")
        finally:
            controller._close_resources()


# ===========================================================================
# Parser regression test
# ===========================================================================


class TestParseJsonReplyRoundtrip:
    """Provider JSON output normalizes correctly through _parse_json_reply."""

    def test_valid_response_roundtrips(self):
        from llm.navigation.parser import _parse_json_reply

        payload = {
            "view-description": "A red cube is ahead.",
            "reasoning": "The red cube matches my color.",
            "action": "done",
            "summary": "Selecting red cube.",
            "selected_cube": "red",
            "identification": {
                "my_color": "red",
                "source": "saw-another-robot",
                "confidence": "high",
            },
        }
        raw = json.dumps(payload)
        view, reasoning, action, summary, cube, identification = _parse_json_reply(raw, previous_action=None)

        assert action == "done"
        assert cube == "red"
        assert identification["my_color"] == "red"
        assert identification["confidence"] == "high"

    def test_action_alias_normalizes(self):
        from llm.navigation.parser import _parse_json_reply

        payload = {
            "view-description": "Empty corridor.",
            "reasoning": "Moving forward.",
            "action": "forward",
            "summary": "Going ahead.",
            "selected_cube": "none",
            "identification": {"my_color": "unknown", "source": "other", "confidence": "low"},
        }
        _, _, action, _, _, _ = _parse_json_reply(json.dumps(payload), previous_action=None)
        assert action == "w"

class TestJudgeImports:
    def test_canonical_judge_modules_import_after_ollama_move(self):
        import experiments.evaluation.judge_reliability  # noqa: F401
        import experiments.evaluation.judge  # noqa: F401
