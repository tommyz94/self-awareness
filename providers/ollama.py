import base64
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _encode_image_reference(image: Any) -> Any:
    if not isinstance(image, str):
        return image

    try:
        image_path = Path(image)
    except (OSError, ValueError):
        return image

    try:
        if not image_path.is_file():
            return image
        image_bytes = image_path.read_bytes()
    except OSError:
        return image

    return base64.b64encode(image_bytes).decode("utf-8")


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_messages: List[Dict[str, Any]] = []

    for message in messages:
        normalized_message = dict(message)
        images = normalized_message.get("images")

        if isinstance(images, list):
            normalized_message["images"] = [_encode_image_reference(image) for image in images]

        normalized_messages.append(normalized_message)

    return normalized_messages


def chat(
    host: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
    timeout_sec: int = 180,
    output_format: Any = "json",
    stream_thinking: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Call Ollama /api/chat and return (content, thinking).
    thinking is None when stream_thinking=False or when the model does not
    produce reasoning tokens.
    When stream_thinking=True, thinking tokens are printed to stdout in real-time.
    """
    payload = {
        "model": model,
        "messages": _normalize_messages(messages),
        "stream": stream_thinking,
        "format": output_format,
        "options": {"temperature": temperature},
    }

    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            if not stream_thinking:
                body = response.read().decode("utf-8")
            else:
                content_acc = ""
                thinking_acc = ""
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = chunk.get("message", {})
                    thinking_token = msg.get("thinking") or ""
                    content_token = msg.get("content") or ""
                    if thinking_token:
                        print(thinking_token, end="", flush=True)
                        thinking_acc += thinking_token
                    content_acc += content_token
                if thinking_acc:
                    print()  # newline after streamed thinking
                return content_acc, thinking_acc or None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {host}. Make sure Ollama is running."
        ) from exc

    try:
        parsed = json.loads(body)
        msg = parsed["message"]
        content = msg["content"]
        thinking = (msg.get("thinking") or None) if stream_thinking else None
        return content, thinking
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Unexpected Ollama response: {body}") from exc


def list_models(host: str, timeout_sec: int = 10) -> List[str]:
    """
    Return the list of model names available in local Ollama.
    """
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/tags",
        headers={"Content-Type": "application/json"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {host}. Make sure Ollama is running."
        ) from exc

    try:
        parsed = json.loads(body)
        models = parsed.get("models", [])
        return [m["name"] for m in models if isinstance(m, dict) and "name" in m]
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Unexpected Ollama tags response: {body}") from exc
