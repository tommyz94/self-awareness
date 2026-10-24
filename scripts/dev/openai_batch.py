#!/usr/bin/env python3
"""
Utility for OpenAI's discounted Batch API workflow.

The Batch API processes asynchronous requests within a 24-hour window for
roughly 50% lower cost than the equivalent synchronous calls. This helper
validates a JSONL request file locally, uploads it with ``purpose="batch"``,
creates the batch, optionally waits for completion, and can download the
resulting output and error files.

Example:

    OPENAI_API_KEY=... python3 scripts/dev/openai_batch.py submit \
      scripts/dev/openai_batch.example.jsonl \
      --endpoint /v1/responses \
      --wait \
      --download-dir outputs/openai_batches/demo
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

SUPPORTED_ENDPOINTS = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/moderations",
    "/v1/responses",
}
TERMINAL_STATUSES = {"cancelled", "completed", "expired", "failed"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate, submit, and inspect OpenAI Batch API jobs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a local Batch API JSONL input file without calling the API.",
    )
    _add_input_file_arg(validate_parser)
    _add_endpoint_arg(validate_parser)
    validate_parser.set_defaults(func=_cmd_validate)

    submit_parser = subparsers.add_parser(
        "submit",
        help="Upload a JSONL input file and create a batch.",
    )
    _add_input_file_arg(submit_parser)
    _add_endpoint_arg(submit_parser)
    _add_common_auth_args(submit_parser)
    submit_parser.add_argument(
        "--completion-window",
        default="24h",
        help="Requested completion window. OpenAI currently documents 24h.",
    )
    submit_parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Optional batch metadata pair. May be repeated.",
    )
    submit_parser.add_argument(
        "--wait",
        action="store_true",
        help="Poll until the batch reaches a terminal status.",
    )
    submit_parser.add_argument(
        "--poll-interval-sec",
        type=float,
        default=30.0,
        help="Polling interval in seconds when --wait is used.",
    )
    submit_parser.add_argument(
        "--download-dir",
        default="",
        help=(
            "Optional directory for batch summary and downloaded output/error "
            "files. If omitted, nothing is written locally after submission."
        ),
    )
    submit_parser.set_defaults(func=_cmd_submit)

    status_parser = subparsers.add_parser(
        "status",
        help="Fetch the latest status for an existing batch id.",
    )
    _add_batch_id_arg(status_parser)
    _add_common_auth_args(status_parser)
    status_parser.set_defaults(func=_cmd_status)

    download_parser = subparsers.add_parser(
        "download",
        help="Download output and error files for an existing batch.",
    )
    _add_batch_id_arg(download_parser)
    _add_common_auth_args(download_parser)
    download_parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for batch.json, output.jsonl, and errors.jsonl.",
    )
    download_parser.set_defaults(func=_cmd_download)

    cancel_parser = subparsers.add_parser(
        "cancel",
        help="Cancel an in-flight batch.",
    )
    _add_batch_id_arg(cancel_parser)
    _add_common_auth_args(cancel_parser)
    cancel_parser.set_defaults(func=_cmd_cancel)

    return parser.parse_args()


def _add_input_file_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input_file",
        help="Path to the Batch API JSONL input file.",
    )


def _add_batch_id_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "batch_id",
        help="OpenAI batch id, e.g. batch_abc123.",
    )


def _add_endpoint_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--endpoint",
        default="/v1/responses",
        choices=sorted(SUPPORTED_ENDPOINTS),
        help="Endpoint used by every line in the batch input file.",
    )


def _add_common_auth_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional API key fallback. OPENAI_API_KEY still takes precedence.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=120,
        help="OpenAI SDK request timeout in seconds.",
    )


def _cmd_validate(args: argparse.Namespace) -> int:
    input_path = Path(args.input_file)
    requests = _load_batch_requests(input_path)
    _validate_batch_requests(requests, endpoint=args.endpoint)

    print(f"Validated {len(requests)} request(s) in {input_path}.")
    print(f"Endpoint: {args.endpoint}")
    print("All requests use unique custom_id values and POST the expected body.")
    return 0


def _cmd_submit(args: argparse.Namespace) -> int:
    input_path = Path(args.input_file)
    requests = _load_batch_requests(input_path)
    _validate_batch_requests(requests, endpoint=args.endpoint)

    client = _build_client(api_key=args.api_key, timeout_sec=args.timeout_sec)
    metadata = _parse_metadata(args.metadata)

    with input_path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=args.endpoint,
        completion_window=args.completion_window,
        metadata=metadata or None,
    )
    _print_batch_summary(batch)

    final_batch = batch
    if args.wait:
        final_batch = _wait_for_terminal_status(
            client,
            batch_id=batch.id,
            poll_interval_sec=args.poll_interval_sec,
        )
        _print_batch_summary(final_batch)

    if args.download_dir:
        downloaded = _download_batch_artifacts(
            client,
            batch=final_batch,
            output_dir=Path(args.download_dir),
        )
        for path in downloaded:
            print(f"Wrote {path}")
    elif args.wait and getattr(final_batch, "status", "") == "completed":
        print("Batch completed. Re-run with --download-dir to save output files locally.")

    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    client = _build_client(api_key=args.api_key, timeout_sec=args.timeout_sec)
    batch = client.batches.retrieve(args.batch_id)
    _print_batch_summary(batch)
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    client = _build_client(api_key=args.api_key, timeout_sec=args.timeout_sec)
    batch = client.batches.retrieve(args.batch_id)
    downloaded = _download_batch_artifacts(
        client,
        batch=batch,
        output_dir=Path(args.output_dir),
    )
    for path in downloaded:
        print(f"Wrote {path}")
    return 0


def _cmd_cancel(args: argparse.Namespace) -> int:
    client = _build_client(api_key=args.api_key, timeout_sec=args.timeout_sec)
    batch = client.batches.cancel(args.batch_id)
    _print_batch_summary(batch)
    return 0


def _parse_metadata(values: Iterable[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise SystemExit(f"Invalid metadata pair {raw!r}; expected KEY=VALUE.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid metadata pair {raw!r}; key cannot be empty.")
        metadata[key] = value
    return metadata


def _load_batch_requests(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise SystemExit(f"Batch input file not found: {path}") from exc

    requests: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"{path}:{line_number} is not valid JSON: {exc.msg}"
            ) from exc

        if not isinstance(parsed, dict):
            raise SystemExit(f"{path}:{line_number} must contain a JSON object per line.")
        requests.append(parsed)

    if not requests:
        raise SystemExit(f"Batch input file is empty: {path}")
    return requests


def _validate_batch_requests(
    requests: list[dict[str, Any]],
    *,
    endpoint: str,
) -> None:
    seen_custom_ids: set[str] = set()
    for index, request in enumerate(requests, start=1):
        custom_id = request.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id.strip():
            raise SystemExit(f"Request {index} is missing a non-empty custom_id.")
        if custom_id in seen_custom_ids:
            raise SystemExit(f"Duplicate custom_id {custom_id!r} in request {index}.")
        seen_custom_ids.add(custom_id)

        method = request.get("method")
        if method != "POST":
            raise SystemExit(
                f"Request {index} ({custom_id}) must use method POST, got {method!r}."
            )

        url = request.get("url")
        if url != endpoint:
            raise SystemExit(
                f"Request {index} ({custom_id}) targets {url!r}, expected {endpoint!r}."
            )

        body = request.get("body")
        if not isinstance(body, dict):
            raise SystemExit(f"Request {index} ({custom_id}) is missing a JSON object body.")
        if "model" not in body:
            raise SystemExit(f"Request {index} ({custom_id}) is missing body.model.")


def _build_client(*, api_key: str, timeout_sec: int) -> Any:
    try:
        import openai
    except ImportError as exc:
        raise SystemExit(
            "The openai package is not installed. Install it with "
            '`pip install "openai>=1.0.0"` before using this script.'
        ) from exc

    resolved_key = os.getenv("OPENAI_API_KEY") or api_key
    if not resolved_key:
        raise SystemExit(
            "OpenAI API key not found. Set OPENAI_API_KEY or pass --api-key."
        )

    return openai.OpenAI(api_key=resolved_key, timeout=timeout_sec)


def _wait_for_terminal_status(
    client: Any,
    *,
    batch_id: str,
    poll_interval_sec: float,
) -> Any:
    if poll_interval_sec <= 0:
        raise SystemExit("--poll-interval-sec must be greater than zero.")

    last_status = ""
    while True:
        batch = client.batches.retrieve(batch_id)
        status = getattr(batch, "status", "")
        if status != last_status:
            counts = getattr(batch, "request_counts", None)
            print(
                json.dumps(
                    {
                        "batch_id": batch_id,
                        "status": status,
                        "request_counts": _to_jsonable(counts),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            last_status = status

        if status in TERMINAL_STATUSES:
            return batch

        time.sleep(poll_interval_sec)


def _download_batch_artifacts(client: Any, *, batch: Any, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "batch.json"
    summary_path.write_text(
        json.dumps(_batch_summary(batch), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    downloaded_paths = [summary_path]
    output_file_id = getattr(batch, "output_file_id", None)
    error_file_id = getattr(batch, "error_file_id", None)

    if output_file_id:
        output_path = output_dir / "output.jsonl"
        _write_file_content(client.files.content(output_file_id), output_path)
        downloaded_paths.append(output_path)

    if error_file_id:
        error_path = output_dir / "errors.jsonl"
        _write_file_content(client.files.content(error_file_id), error_path)
        downloaded_paths.append(error_path)

    if len(downloaded_paths) == 1:
        print(
            "No output or error files are available yet. "
            "The batch may still be validating or in progress."
        )

    return downloaded_paths


def _write_file_content(content: Any, path: Path) -> None:
    write_to_file = getattr(content, "write_to_file", None)
    if callable(write_to_file):
        write_to_file(str(path))
        return

    text_attr = getattr(content, "text", None)
    if isinstance(text_attr, str):
        path.write_text(text_attr, encoding="utf-8")
        return
    if callable(text_attr):
        text_value = text_attr()
        if isinstance(text_value, str):
            path.write_text(text_value, encoding="utf-8")
            return

    bytes_attr = getattr(content, "content", None)
    if isinstance(bytes_attr, (bytes, bytearray)):
        path.write_bytes(bytes(bytes_attr))
        return

    read = getattr(content, "read", None)
    if callable(read):
        data = read()
        if isinstance(data, str):
            path.write_text(data, encoding="utf-8")
        elif isinstance(data, (bytes, bytearray)):
            path.write_bytes(bytes(data))
        else:
            path.write_text(str(data), encoding="utf-8")
        return

    path.write_text(str(content), encoding="utf-8")


def _print_batch_summary(batch: Any) -> None:
    print(json.dumps(_batch_summary(batch), indent=2, sort_keys=True))


def _batch_summary(batch: Any) -> dict[str, Any]:
    return {
        "id": getattr(batch, "id", None),
        "status": getattr(batch, "status", None),
        "endpoint": getattr(batch, "endpoint", None),
        "completion_window": getattr(batch, "completion_window", None),
        "input_file_id": getattr(batch, "input_file_id", None),
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "request_counts": _to_jsonable(getattr(batch, "request_counts", None)),
        "errors": _to_jsonable(getattr(batch, "errors", None)),
        "metadata": _to_jsonable(getattr(batch, "metadata", None)),
    }


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _to_jsonable(model_dump())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _to_jsonable(to_dict())
    return str(value)


def main() -> int:
    args = _parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
