"""
Unit tests for Ollama and the hosted provider adapters.

All external SDK calls are mocked so these tests run without any installed
provider package or live API key.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import types
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.parametrize("transient_failures", [0, 1])
def test_ollama_stops_retrying_after_a_successful_response(monkeypatch, transient_failures):
    from providers import ollama

    response = io.BytesIO(json.dumps({"message": {"content": '{"action":"w"}'}}).encode())
    urlopen = MagicMock(side_effect=[urllib.error.URLError("temporary")] * transient_failures + [response])
    monkeypatch.setattr(ollama.urllib.request, "urlopen", urlopen)
    content, thinking = ollama.chat(
        host="http://localhost:11434", model="test-model", messages=[], retry_delay_sec=0,
    )
    assert json.loads(content) == {"action": "w"}
    assert thinking is None
    assert urlopen.call_count == transient_failures + 1

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
        assert text_format["name"] == "navigation_response"
        assert text_format["schema"] is _SCHEMA
        assert text_format["strict"] is True
        assert captured["input"][0]["role"] == "user"


# ===========================================================================
# OpenRouter adapter tests
# ===========================================================================


class TestOpenRouterTextConversion:
    def test_system_message_stays_plain_text_without_images(self, tmp_path):
        from providers.openrouter import _build_messages

        items = _build_messages([_SYSTEM_MSG])
        assert items == [{
            "role": "system",
            "content": "You are a robot.",
        }]


class TestOpenRouterImageConversion:
    def test_image_path_becomes_image_url_item_after_text(self, tmp_path):
        from providers.openrouter import _build_messages

        img = _fake_image_path(tmp_path)
        msg = {"role": "user", "content": "Look.", "images": [img]}
        items = _build_messages([msg])

        assert items[0]["content"][0] == {"type": "text", "text": "Look."}
        image_item = items[0]["content"][1]
        assert image_item["type"] == "image_url"
        url = image_item["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == Path(img).read_bytes()


class TestOpenRouterSchemaHandoff:
    def test_schema_passed_to_chat_completions_with_provider_guard(self, monkeypatch):
        captured: dict = {}

        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"action": "w"}'))]
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda **kwargs: (
            captured.update(kwargs) or mock_response
        )

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict(sys.modules, {"openai": mock_openai}):
            import providers.openrouter as openrouter_provider

            monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
            content, thinking = openrouter_provider.chat(
                model="meta-llama/llama-3.2-11b-vision-instruct",
                messages=[_USER_TEXT_MSG],
                output_schema=_SCHEMA,
                http_referer="https://example.com",
                app_title="self-awareness",
            )

        assert thinking is None
        assert content == '{"action": "w"}'
        assert captured["response_format"]["type"] == "json_schema"
        assert captured["response_format"]["json_schema"]["schema"] is _SCHEMA
        assert captured["extra_headers"] == {
            "HTTP-Referer": "https://example.com",
            "X-OpenRouter-Title": "self-awareness",
        }
        assert captured["extra_body"] == {"provider": {"require_parameters": True}}

    def test_provider_routing_controls_are_forwarded(self, monkeypatch):
        captured: dict = {}

        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"action": "w"}'))]
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda **kwargs: (
            captured.update(kwargs) or mock_response
        )

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict(sys.modules, {"openai": mock_openai}):
            import providers.openrouter as openrouter_provider

            monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
            content, thinking = openrouter_provider.chat(
                model="anthropic/claude-sonnet-4.6",
                messages=[_USER_TEXT_MSG],
                output_schema=_SCHEMA,
                require_parameters=True,
                allow_fallbacks=False,
                provider_order=["anthropic"],
                provider_ignore=["azure"],
            )

        assert thinking is None
        assert content == '{"action": "w"}'
        assert captured["extra_body"] == {
            "provider": {
                "require_parameters": True,
                "allow_fallbacks": False,
                "order": ["anthropic"],
                "ignore": ["azure"],
            }
        }


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

    def test_prefers_structured_parsed_payload_over_malformed_text(self, monkeypatch):
        mock_response = MagicMock()
        mock_response.text = '{\n  "view-description'
        mock_response.parsed = {
            "view-description": "Mirror room with cubes.",
            "reasoning": "Need to keep exploring.",
            "action": "w",
            "summary": "Move forward.",
            "selected_cube": "none",
            "identification": {
                "my_color": "unknown",
                "source": "other",
                "confidence": "low",
            },
        }

        mock_models = MagicMock()
        mock_models.generate_content.return_value = mock_response

        mock_client_instance = MagicMock()
        mock_client_instance.models = mock_models

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client_instance

        mock_types = MagicMock()
        mock_types.Part.from_text.side_effect = lambda text: {"_text": text}
        mock_types.Content.side_effect = lambda role, parts: {"role": role, "parts": parts}
        mock_types.GenerateContentConfig.side_effect = lambda **kw: {"_config": kw}

        google_pkg = types.ModuleType("google")
        google_genai_pkg = types.ModuleType("google.genai")
        google_genai_pkg.Client = mock_genai.Client
        google_genai_pkg.types = mock_types

        with patch.dict(sys.modules, {"google": google_pkg, "google.genai": google_genai_pkg}):
            import providers.google as google_provider

            monkeypatch.setenv("GEMINI_API_KEY", "test-key")
            content, thinking = google_provider.chat(
                model="gemini-2.5-flash",
                messages=[_USER_TEXT_MSG],
                output_schema=_SCHEMA,
            )

        assert thinking is None
        assert json.loads(content) == mock_response.parsed


class TestGoogleVertexAIConfiguration:
    def test_vertex_ai_client_uses_project_and_location(self, monkeypatch):
        captured: dict = {}
        config_calls: list = []

        mock_response = MagicMock()
        mock_response.text = '{"action": "w"}'

        mock_models = MagicMock()
        mock_models.generate_content.return_value = mock_response

        def _fake_client(**kwargs):
            captured.update(kwargs)
            client = MagicMock()
            client.models = mock_models
            return client

        mock_types = MagicMock()
        mock_types.Part.from_text.side_effect = lambda text: {"_text": text}
        mock_types.Content.side_effect = lambda role, parts: {"role": role, "parts": parts}
        mock_types.GenerateContentConfig.side_effect = lambda **kw: config_calls.append(kw) or {"_config": kw}
        mock_types.HttpOptions.side_effect = lambda **kw: {"_http_options": kw}

        google_pkg = types.ModuleType("google")
        google_genai_pkg = types.ModuleType("google.genai")
        google_genai_pkg.Client = _fake_client
        google_genai_pkg.types = mock_types

        with patch.dict(sys.modules, {"google": google_pkg, "google.genai": google_genai_pkg}):
            import providers.google as google_provider

            monkeypatch.delenv("GEMINI_API_KEY", raising=False)
            monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
            google_provider.chat(
                model="gemini-2.5-flash",
                messages=[_USER_TEXT_MSG],
                output_schema=_SCHEMA,
                vertexai=True,
                project="gemini-self-perspetion",
                location="us-central1",
                max_output_tokens=256,
            )

        assert captured == {
            "vertexai": True,
            "project": "gemini-self-perspetion",
            "location": "us-central1",
            "http_options": {"_http_options": {"timeout": 180000}},
        }
        assert config_calls[0]["max_output_tokens"] == 256

    def test_gemini_25_pro_uses_low_thinking_budget(self, monkeypatch):
        config_calls: list = []

        mock_response = MagicMock()
        mock_response.text = '{"action": "w"}'

        mock_models = MagicMock()
        mock_models.generate_content.return_value = mock_response

        def _fake_client(**kwargs):
            client = MagicMock()
            client.models = mock_models
            return client

        mock_types = MagicMock()
        mock_types.Part.from_text.side_effect = lambda text: {"_text": text}
        mock_types.Content.side_effect = lambda role, parts: {"role": role, "parts": parts}
        mock_types.GenerateContentConfig.side_effect = lambda **kw: config_calls.append(kw) or {"_config": kw}
        mock_types.ThinkingConfig.side_effect = lambda **kw: {"_thinking": kw}

        google_pkg = types.ModuleType("google")
        google_genai_pkg = types.ModuleType("google.genai")
        google_genai_pkg.Client = _fake_client
        google_genai_pkg.types = mock_types

        with patch.dict(sys.modules, {"google": google_pkg, "google.genai": google_genai_pkg}):
            import providers.google as google_provider

            monkeypatch.setenv("GEMINI_API_KEY", "test-key")
            google_provider.chat(
                model="gemini-2.5-pro",
                messages=[_USER_TEXT_MSG],
                output_schema=_SCHEMA,
            )

        thinking_config = config_calls[0]["thinking_config"]
        assert thinking_config == {
            "_thinking": {
                "thinking_budget": 128,
                "include_thoughts": False,
            }
        }

    def test_retries_once_on_resource_exhausted_then_succeeds(self, monkeypatch):
        call_count = {"n": 0}
        sleep_calls: list[float] = []

        class FakeRetryableError(Exception):
            def __init__(self):
                self.code = 429
                self.status = "RESOURCE_EXHAUSTED"
                self.message = "Resource exhausted. Please try again later."
                super().__init__(self.message)

        mock_response = MagicMock()
        mock_response.text = '{"action": "w"}'

        def _fake_generate_content(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise FakeRetryableError()
            return mock_response

        mock_models = MagicMock()
        mock_models.generate_content.side_effect = _fake_generate_content

        def _fake_client(**kwargs):
            client = MagicMock()
            client.models = mock_models
            return client

        mock_types = MagicMock()
        mock_types.Part.from_text.side_effect = lambda text: {"_text": text}
        mock_types.Content.side_effect = lambda role, parts: {"role": role, "parts": parts}
        mock_types.GenerateContentConfig.side_effect = lambda **kw: {"_config": kw}

        google_pkg = types.ModuleType("google")
        google_genai_pkg = types.ModuleType("google.genai")
        google_genai_pkg.Client = _fake_client
        google_genai_pkg.types = mock_types

        with patch.dict(sys.modules, {"google": google_pkg, "google.genai": google_genai_pkg}):
            import providers.google as google_provider

            monkeypatch.setattr(google_provider.time, "sleep", lambda seconds: sleep_calls.append(seconds))
            monkeypatch.setenv("GEMINI_API_KEY", "test-key")
            content, thinking = google_provider.chat(
                model="gemini-2.5-pro",
                messages=[_USER_TEXT_MSG],
                output_schema=_SCHEMA,
            )

        assert thinking is None
        assert content == '{"action": "w"}'
        assert call_count["n"] == 2
        assert sleep_calls == [2.0]

    def test_retry_parameters_are_configurable(self, monkeypatch):
        call_count = {"n": 0}
        sleep_calls: list[float] = []

        class FakeRetryableError(Exception):
            def __init__(self):
                self.code = 429
                self.status = "RESOURCE_EXHAUSTED"
                self.message = "Resource exhausted. Please try again later."
                super().__init__(self.message)

        mock_response = MagicMock()
        mock_response.text = '{"action": "w"}'

        def _fake_generate_content(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise FakeRetryableError()
            return mock_response

        mock_models = MagicMock()
        mock_models.generate_content.side_effect = _fake_generate_content

        def _fake_client(**kwargs):
            client = MagicMock()
            client.models = mock_models
            return client

        mock_types = MagicMock()
        mock_types.Part.from_text.side_effect = lambda text: {"_text": text}
        mock_types.Content.side_effect = lambda role, parts: {"role": role, "parts": parts}
        mock_types.GenerateContentConfig.side_effect = lambda **kw: {"_config": kw}

        google_pkg = types.ModuleType("google")
        google_genai_pkg = types.ModuleType("google.genai")
        google_genai_pkg.Client = _fake_client
        google_genai_pkg.types = mock_types

        with patch.dict(sys.modules, {"google": google_pkg, "google.genai": google_genai_pkg}):
            import providers.google as google_provider

            monkeypatch.setattr(google_provider.time, "sleep", lambda seconds: sleep_calls.append(seconds))
            monkeypatch.setenv("GEMINI_API_KEY", "test-key")
            content, thinking = google_provider.chat(
                model="gemini-2.5-pro",
                messages=[_USER_TEXT_MSG],
                output_schema=_SCHEMA,
                max_retries=9,
                initial_retry_delay_sec=7.0,
                max_retry_delay_sec=60.0,
            )

        assert thinking is None
        assert content == '{"action": "w"}'
        assert call_count["n"] == 2
        assert sleep_calls == [7.0]

    def test_retries_once_on_deadline_exceeded_then_succeeds(self, monkeypatch):
        call_count = {"n": 0}
        sleep_calls: list[float] = []

        class FakeDeadlineExceededError(Exception):
            def __init__(self):
                self.code = 504
                self.status = "DEADLINE_EXCEEDED"
                self.message = "Deadline expired before operation could complete."
                super().__init__(self.message)

        mock_response = MagicMock()
        mock_response.text = '{"action": "w"}'

        def _fake_generate_content(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise FakeDeadlineExceededError()
            return mock_response

        mock_models = MagicMock()
        mock_models.generate_content.side_effect = _fake_generate_content

        def _fake_client(**kwargs):
            client = MagicMock()
            client.models = mock_models
            return client

        mock_types = MagicMock()
        mock_types.Part.from_text.side_effect = lambda text: {"_text": text}
        mock_types.Content.side_effect = lambda role, parts: {"role": role, "parts": parts}
        mock_types.GenerateContentConfig.side_effect = lambda **kw: {"_config": kw}

        google_pkg = types.ModuleType("google")
        google_genai_pkg = types.ModuleType("google.genai")
        google_genai_pkg.Client = _fake_client
        google_genai_pkg.types = mock_types

        with patch.dict(sys.modules, {"google": google_pkg, "google.genai": google_genai_pkg}):
            import providers.google as google_provider

            monkeypatch.setattr(google_provider.time, "sleep", lambda seconds: sleep_calls.append(seconds))
            monkeypatch.setenv("GEMINI_API_KEY", "test-key")
            content, thinking = google_provider.chat(
                model="gemini-2.5-pro",
                messages=[_USER_TEXT_MSG],
                output_schema=_SCHEMA,
            )

        assert thinking is None
        assert content == '{"action": "w"}'
        assert call_count["n"] == 2
        assert sleep_calls == [2.0]


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

    def test_vertex_ai_client_uses_project_location_and_max_retries(self, monkeypatch):
        captured_client: dict = {}
        captured_request: dict = {}

        mock_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"action": "w"}')]
        )

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = lambda **kwargs: (
            captured_request.update(kwargs) or mock_response
        )

        def _fake_vertex_client(**kwargs):
            captured_client.update(kwargs)
            return mock_client

        mock_anthropic = MagicMock()
        mock_anthropic.AnthropicVertex.side_effect = _fake_vertex_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            import providers.anthropic as anthropic_provider

            monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
            content, thinking = anthropic_provider.chat(
                model="claude-sonnet-4-6",
                messages=[_SYSTEM_MSG, _USER_TEXT_MSG],
                output_schema=_SCHEMA,
                vertexai=True,
                project="gemini-self-perspetion",
                location="global",
                max_output_tokens=1024,
                max_retries=7,
                timeout_sec=123,
            )

        assert thinking is None
        assert content == '{"action": "w"}'
        assert captured_client == {
            "region": "global",
            "project_id": "gemini-self-perspetion",
            "timeout": 123,
            "max_retries": 7,
        }
        assert captured_request["model"] == "claude-sonnet-4-6"
        assert captured_request["max_tokens"] == 1024
        assert captured_request["output_config"]["format"]["schema"] is _SCHEMA


class TestDispatchCanonicalization:
    def test_openrouter_routes_to_openrouter_provider(self, monkeypatch):
        import llm.dispatch as chat_module

        called = {}

        def fake_chat(**kwargs):
            called.update(kwargs)
            return '{"action": "w"}', None

        monkeypatch.setattr("providers.openrouter.chat", fake_chat)

        content, thinking = chat_module.chat(
            client_type="openrouter",
            model="meta-llama/llama-3.2-11b-vision-instruct",
            messages=[_USER_TEXT_MSG],
            output_schema=_SCHEMA,
        )

        assert thinking is None
        assert content == '{"action": "w"}'
        assert called["model"] == "meta-llama/llama-3.2-11b-vision-instruct"

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
            vertexai=True,
            project="gemini-self-perspetion",
            location="global",
            max_output_tokens=1024,
            max_retries=7,
        )

        assert thinking is None
        assert content == '{"action": "w"}'
        assert called["model"] == "claude-sonnet-4-5"
        assert called["vertexai"] is True
        assert called["project"] == "gemini-self-perspetion"
        assert called["location"] == "global"
        assert called["max_output_tokens"] == 1024
        assert called["max_retries"] == 7

    def test_canonicalize_client_type_maps_claude(self):
        from llm.dispatch import canonicalize_client_type

        assert canonicalize_client_type("claude") == "anthropic"
        assert canonicalize_client_type("anthropic") == "anthropic"


class TestControllerCanonicalization:
    def test_controller_uses_openrouter_client_for_paths_and_api_key(self, monkeypatch, tmp_path):
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
            nav_mode="meta-llama/llama-3.2-11b-vision-instruct",
            client_type="openrouter",
            openrouter_api_key="openrouter-key",
            output_dir=str(tmp_path / "outputs"),
            env_type="mujoco",
            mujoco_headless=True,
        )

        try:
            assert controller.client_type == "openrouter"
            assert controller.api_key == "openrouter-key"
            assert "/openrouter/" in controller.output_dir.replace("\\", "/")
        finally:
            controller._close_resources()

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

    def test_mirror_contingency_source_is_preserved(self):
        from llm.navigation.parser import _parse_json_reply

        payload = {
            "view-description": "A robot is visible in the mirror.",
            "reasoning": "The mirror reveals my body color.",
            "action": "w",
            "summary": "Moving closer to inspect the mirror.",
            "selected_cube": "none",
            "identification": {
                "my_color": "yellow",
                "source": "mirror-contingency",
                "confidence": "high",
            },
        }
        _, _, _, _, _, identification = _parse_json_reply(json.dumps(payload), previous_action=None)
        assert identification["source"] == "mirror-contingency"

class TestJudgeImports:
    def test_canonical_judge_modules_import_after_ollama_move(self):
        import experiments.evaluation.judge_reliability  # noqa: F401
        import experiments.evaluation.judge  # noqa: F401


@pytest.mark.parametrize("provider", ["openai", "openrouter", "anthropic", "google"])
@pytest.mark.parametrize("failure", ["missing", "directory", "permission"])
def test_unreadable_observation_stops_before_api_request(monkeypatch, tmp_path, provider, failure):
    import importlib

    sdk = MagicMock()
    if provider == "google":
        google = types.ModuleType("google")
        google.genai = sdk
        monkeypatch.setitem(sys.modules, "google", google)
        monkeypatch.setitem(sys.modules, "google.genai", sdk)
        request = sdk.Client.return_value.models.generate_content
    elif provider == "anthropic":
        monkeypatch.setitem(sys.modules, "anthropic", sdk)
        request = sdk.Anthropic.return_value.messages.create
    else:
        monkeypatch.setitem(sys.modules, "openai", sdk)
        client = sdk.OpenAI.return_value
        request = client.responses.create if provider == "openai" else client.chat.completions.create

    bad_path = tmp_path / "unreadable.jpg"
    expected = FileNotFoundError
    if failure == "directory":
        bad_path.mkdir()
        expected = IsADirectoryError
    elif failure == "permission":
        expected = PermissionError
        read_bytes = Path.read_bytes

        def read_image(path):
            if path == bad_path:
                raise PermissionError(13, "Permission denied", str(path))
            return read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", read_image)

    adapter = importlib.import_module(f"providers.{provider}")
    with pytest.raises(expected, match="unreadable.jpg"):
        adapter.chat(
            model="test-vision-model", api_key="test-key", output_schema=_SCHEMA,
            messages=[{"role": "user", "content": "Look.",
                       "images": [_fake_image_path(tmp_path), str(bad_path)]}],
        )
    request.assert_not_called()
