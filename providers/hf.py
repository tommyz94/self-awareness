"""
Hugging Face / Local Transformers provider adapter for the navigation VLM.

Supports running models natively in-process on Apple Silicon (MPS), CUDA, or CPU
without requiring an external API or inference server.
Specialized support for Qwen-Drive-1.0 (QwenDriveForPlanning) with sdpa attention.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

_LOADED_MODELS: Dict[Tuple[str, torch.device, torch.dtype], Tuple[Any, Any, torch.device]] = {}


def _get_default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_default_dtype(device: torch.device) -> torch.dtype:
    if device.type in ("mps", "cuda"):
        return torch.bfloat16
    return torch.float32


def load_model(
    model_name: str,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype | str] = None,
) -> Tuple[Any, Any, torch.device]:
    """
    Load and cache model and processor in memory.
    """
    device = _get_default_device() if device is None or device == "auto" else torch.device(device)
    if dtype is None or dtype == "auto":
        dtype = _get_default_dtype(device)
    elif isinstance(dtype, str):
        dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if dtype not in dtypes:
            raise ValueError(f"Unknown HF dtype {dtype!r}; use auto, float32, float16 or bfloat16.")
        dtype = dtypes[dtype]

    cache_key = (model_name, device, dtype)
    if cache_key in _LOADED_MODELS:
        return _LOADED_MODELS[cache_key]

    print(f"[HF Provider] Loading model '{model_name}' on {device} ({dtype})...")

    # Specialized loading for Qwen-Drive models
    if "qwen-drive" in model_name.lower():
        from qwen_drive import QwenDriveForPlanning

        # In VQA / MSR task mode, only the VLM backbone is required (planner=None)
        # Use sdpa for native Apple Silicon MPS / CPU support
        model = QwenDriveForPlanning.from_pretrained(
            model_name,
            planner=None,
            dtype=dtype,
            attn_implementation="sdpa",
        ).to(device).eval()
        processor = model.processor
    else:
        # Generic Vision-Language Model fallback via transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=dtype,
            attn_implementation="sdpa",
            trust_remote_code=True,
        ).to(device).eval()

    _LOADED_MODELS[cache_key] = (model, processor, device)
    print(f"[HF Provider] Model '{model_name}' loaded successfully.")
    return _LOADED_MODELS[cache_key]


def _extract_messages_content(
    messages: List[Dict[str, Any]]
) -> Tuple[str, str, List[str]]:
    """
    Extract (system_prompt, user_content, image_paths) from simulator messages.
    """
    system_prompt = ""
    user_content = ""
    image_paths: List[str] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        imgs = msg.get("images", [])

        if role == "system":
            system_prompt = content
        elif role == "user":
            user_content = content
            if isinstance(imgs, list):
                image_paths.extend(imgs)

    return system_prompt, user_content, image_paths


def _split_thinking_and_json(raw_text: str) -> Tuple[str, Optional[str]]:
    """
    Extract optional <think>...</think> reasoning trace and remaining JSON text.
    """
    thinking = None
    content = raw_text.strip()

    if "<think>" in content and "</think>" in content:
        parts = content.split("</think>", 1)
        thinking_block = parts[0].split("<think>", 1)[-1]
        thinking = thinking_block.strip()
        content = parts[1].strip()
    elif "</think>" in content:
        parts = content.split("</think>", 1)
        thinking = parts[0].strip()
        content = parts[1].strip()

    # Strip markdown code block fences if present
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    return content, thinking


def chat(
    model: str,
    messages: List[Dict[str, Any]],
    output_schema: Optional[Any] = None,
    stream_thinking: bool = False,
    max_output_tokens: Optional[int] = None,
    temperature: float = 0.2,
    timeout_sec: int = 600,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype | str] = None,
    **kwargs: Any,
) -> Tuple[str, Optional[str]]:
    """
    Generate navigation action response using native Hugging Face in-process inference.

    Returns:
        (content_json_string, thinking_or_None)
    """
    model_instance, processor, device = load_model(model, device=device, dtype=dtype)
    system_prompt, user_content, image_paths = _extract_messages_content(messages)

    if not image_paths:
        raise ValueError("[HF Provider] At least one observation image is required.")

    # Format inputs
    if hasattr(processor, "encode_vqa"):
        # Qwen-Drive processor
        inputs = processor.encode_vqa(
            images=image_paths,
            question=user_content,
            system=system_prompt if system_prompt else None,
            device=device,
        )

        gen_params: Dict[str, Any] = {
            "max_new_tokens": max_output_tokens or 1024,
            "repetition_penalty": 1.0,
        }
        if temperature > 0.0:
            gen_params["do_sample"] = True
            gen_params["temperature"] = temperature
            gen_params["top_p"] = 0.95
            gen_params["top_k"] = 50
        else:
            gen_params["do_sample"] = False

        with torch.inference_mode():
            generated = model_instance.vlm.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                mm_token_type_ids=model_instance._modality_ids(inputs["input_ids"]),
                **gen_params,
            )

        prompt_len = inputs["input_ids"].shape[1]
        output_ids = generated[0, prompt_len:]
        raw_answer = processor.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    else:
        # Generic Vision-Language processor
        from PIL import Image

        pil_images = [Image.open(p).convert("RGB") for p in image_paths]
        text_prompt = f"{system_prompt}\n\n{user_content}" if system_prompt else user_content
        formatted = processor(text=text_prompt, images=pil_images, return_tensors="pt").to(device)

        gen_params = {
            "max_new_tokens": max_output_tokens or 1024,
            "do_sample": temperature > 0.0,
            "temperature": max(temperature, 0.01),
        }
        with torch.inference_mode():
            generated = model_instance.generate(**formatted, **gen_params)

        prompt_len = formatted["input_ids"].shape[1]
        output_ids = generated[0, prompt_len:]
        raw_answer = processor.decode(output_ids, skip_special_tokens=True).strip()

    content, thinking = _split_thinking_and_json(raw_answer)

    if stream_thinking and thinking:
        print(f"\n[Thinking]:\n{thinking}\n")

    return content, thinking
