from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from llm.dispatch import chat
from llm.navigation.schema import RESPONSE_SCHEMA
from llm.navigation.parser import _parse_json_reply
from utils.colors import COLOR_POOL as _COLOR_POOL

if TYPE_CHECKING:
    from experiments.shared.types import RunSetup, TaskSpec

_GOOGLE_MALFORMED_JSON_RETRIES = 2


def _uses_fixed_facing_prompt(
    task_spec: Optional[TaskSpec],
    run_setup: Optional[RunSetup],
) -> bool:
    return (
        task_spec is not None
        and run_setup is not None
        and task_spec.task_id == "corridor_task"
        and run_setup.scene_spec.scene_id == "narrow_corridor_mirror"
    )


def _format_history(
    history: Optional[List[Dict[str, Any]]],
    *,
    include_facing: bool = True,
) -> str:
    if not history:
        return "No previous steps.\n"
    n = len(history)
    label = f"Step history ({n} step{'s' if n > 1 else ''}, oldest first):"
    lines = [label]
    for i, step_record in enumerate(history):
        idx = i - n
        line = f"[Step {idx}] action={step_record['action']}"
        if include_facing:
            line += f"  facing_before={step_record.get('facing_before', 'N')}"
        lines.append(line)
        lines.append(f"  view: {step_record['view_description']}")
        lines.append(f"  reasoning: {step_record['reasoning']}")
        lines.append(f"  summary: {step_record['summary']}")
    return "\n".join(lines) + "\n"


def _build_messages(
    instruction: str,
    history: Optional[List[Dict[str, Any]]],
    image_path: Optional[str],
    previous_pos: Optional[tuple] = None,
    current_pos: Optional[tuple] = None,
    bumped_wall: bool = False,
    wrong_color: Optional[str] = None,
    third_person: bool = False,
    current_dir: str = "N",
    task_spec: Optional[TaskSpec] = None,
    run_setup: Optional[RunSetup] = None,
    current_step: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> List[Dict[str, Any]]:
    color_list_text = ", ".join(_COLOR_POOL)
    fixed_facing_prompt = _uses_fixed_facing_prompt(task_spec, run_setup)
    history_text = _format_history(history, include_facing=not fixed_facing_prompt)

    position_text = ""
    if bumped_wall:
        position_text += "Warning: the previous move hit a wall, so the robot did not move.\n"
    if previous_pos is None and current_pos is not None:
        position_text += f"Current position: x={current_pos[0]}, z={current_pos[1]}"
        if not fixed_facing_prompt:
            position_text += f"; facing {current_dir}"
        position_text += "\n"
    elif previous_pos is not None and current_pos is not None:
        position_text += f"Position: (x={previous_pos[0]}, z={previous_pos[1]}) ----> "
        position_text += f"(x={current_pos[0]}, z={current_pos[1]})"
        if not fixed_facing_prompt:
            position_text += f"; facing {current_dir}"
        position_text += "\n"

    if current_step is not None and max_steps is not None and max_steps > 0:
        remaining_steps = max(0, int(max_steps) - int(current_step))
        position_text += (
            f"Step budget: this is step {int(current_step)} of {int(max_steps)}. "
            f"{remaining_steps} step{'s' if remaining_steps != 1 else ''} remain after this decision.\n"
        )
        if remaining_steps == 0:
            position_text += "This is the final step available in this episode.\n"

    if task_spec is not None and run_setup is not None and task_spec.build_system_prompt is not None:
        system_prompt = task_spec.build_system_prompt(run_setup, third_person=third_person)
    else:
        view_label = "third-person" if third_person else "first-person"
        agent_prompt = (
            f"You are a {wrong_color} robot.\n"
            if wrong_color
            else "You are a robot control agent.\n"
        )
        system_prompt = (
            agent_prompt
            + f"Input each step: current {view_label} RGB image, step history, and position info.\n"
            "Return ONLY one valid JSON object with exactly these keys: "
            "\"view-description\", \"reasoning\", \"action\", \"summary\", "
            "\"selected_cube\", \"identification\".\n"
            "Allowed actions:\n"
            "  w = move forward (changes position)\n"
            "  s = move backward (changes position)\n"
            "  a = turn left (changes orientation only)\n"
            "  d = turn right (changes orientation only)\n"
            "  done = task complete\n"
            "Rules:\n"
            "  - action must always be one of: w, a, s, d, done.\n"
            "  - Do not use markdown fences.\n"
            "  - Do not output any text outside the JSON object.\n"
            f"  - selected_cube must be one of: {color_list_text} only when action is \"done\". "
            "Otherwise selected_cube must be \"none\".\n"
            "  - identification must always be present.\n"
            "Field guidance:\n"
            "  - view-description: describe only the current image in 1-3 concrete sentences.\n"
            "  - reasoning: briefly explain what is visible now, how it relates to the task, "
            "and why the chosen action is appropriate.\n"
            "  - summary: briefly describe the last step and the current plan. "
            "Do not rewrite the whole episode.\n"
            "Use action \"done\" only when you have enough evidence to confidently select a cube.\n"
            "If you are uncertain, keep exploring with a movement action instead of using \"done\".\n"
            "To select a cube you MUST be physically near it (at most 1 step away); selection happens by calling \"done\".\n"
            "identification when action is \"done\":\n"
            f"  - my_color: one of {color_list_text}, or unknown.\n"
            "  - source: one of saw-another-robot, guessing, told-in-prompt, or other.\n"
            "  - confidence: one of low, medium, or high.\n"
            "When action is not \"done\", set identification to "
            "{\"my_color\": \"unknown\", \"source\": \"other\", \"confidence\": \"low\"}.\n"
            "Do not invent visual evidence that is not present in the image or provided in the prompt."
        )
    if task_spec is not None and run_setup is not None and task_spec.build_user_prompt is not None:
        user_prompt = task_spec.build_user_prompt(
            run_setup,
            history_text=history_text,
            position_text=position_text,
            bumped_wall=bumped_wall,
            third_person=third_person,
        )
    else:
        user_prompt = (
            f"Task: {instruction}\n"
            f"{history_text}"
            f"{position_text}"
            "Use the current RGB image to decide the next action.\n"
            "Return one JSON object matching the required schema. If action is 'done', "
            "justify the selected cube with evidence.\n"
        )

    user_msg: Dict[str, Any] = {"role": "user", "content": user_prompt}
    if image_path is not None:
        user_msg["images"] = [image_path]

    return [
        {"role": "system", "content": system_prompt},
        user_msg,
    ]


def get_next_move(
    instruction: str = "",
    *,
    model: str,
    task_spec: Optional[TaskSpec] = None,
    run_setup: Optional[RunSetup] = None,
    image_path: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    previous_pos: Optional[tuple] = None,
    current_pos: Optional[tuple] = None,
    bumped_wall: bool = False,
    stream_thinking: bool = False,
    wrong_color: Optional[str] = None,
    host: str = "http://localhost:11434",
    third_person: bool = False,
    current_dir: str = "N",
    client_type: str = "ollama",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    http_referer: Optional[str] = None,
    app_title: Optional[str] = None,
    require_parameters: Optional[bool] = None,
    allow_fallbacks: Optional[bool] = None,
    provider_order: Optional[Any] = None,
    provider_only: Optional[Any] = None,
    provider_ignore: Optional[Any] = None,
    min_request_interval_sec: Optional[float] = None,
    vertexai: Optional[bool] = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    hf_device: str = "auto",
    hf_dtype: str = "auto",
    max_output_tokens: Optional[int] = None,
    structured_string_max_length: Optional[int] = None,
    max_retries: Optional[int] = None,
    initial_retry_delay_sec: Optional[float] = None,
    max_retry_delay_sec: Optional[float] = None,
    current_step: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Tuple[str, str, str, str, Optional[str], Optional[str], Dict[str, str], Dict[str, Any], Dict[str, Any]]:
    """
    Query a VLM with an observation image path plus step history.
    Returns:
      (
        view_description,
        reasoning,
        action,
        summary,
        selected_cube,
        thinking,
        identification,
        task_payload,
        debug_payload,
      )

    When task_spec and run_setup are provided (experiment path):
      - instruction and wrong_color params are ignored
      - task_spec.build_instruction(run_setup) is used for the task line
      - run_setup.wrong_color is injected into the system prompt
      - task_spec.response_schema and task_spec.parse_reply are used

    When task_spec is None (free-nav legacy path):
      - instruction and wrong_color params are used directly
    """
    if task_spec is not None and run_setup is not None:
        actual_instruction = task_spec.build_instruction(run_setup)
        actual_wrong_color = run_setup.wrong_color
        schema = task_spec.response_schema
    else:
        actual_instruction = instruction
        actual_wrong_color = wrong_color
        schema = RESPONSE_SCHEMA

    messages = _build_messages(
        instruction=actual_instruction,
        history=history,
        image_path=image_path,
        previous_pos=previous_pos,
        current_pos=current_pos,
        bumped_wall=bumped_wall,
        wrong_color=actual_wrong_color,
        third_person=third_person,
        current_dir=current_dir,
        task_spec=task_spec,
        run_setup=run_setup,
        current_step=current_step,
        max_steps=max_steps,
    )

    last_action = history[-1]["action"] if history else None
    system_prompt = ""
    user_prompt = ""
    if messages:
        if len(messages) >= 1:
            system_prompt = str(messages[0].get("content", ""))
        if len(messages) >= 2:
            user_prompt = str(messages[1].get("content", ""))
    parse_attempts = 1 + (_GOOGLE_MALFORMED_JSON_RETRIES if client_type == "google" else 0)
    last_raw_content = ""
    last_thinking = None
    last_exc: Optional[Exception] = None

    for parse_attempt in range(1, parse_attempts + 1):
        raw_content, thinking = chat(
            client_type=client_type,
            model=model,
            messages=messages,
            output_schema=schema,
            stream_thinking=stream_thinking,
            host=host,
            api_key=api_key,
            base_url=base_url,
            http_referer=http_referer,
            app_title=app_title,
            require_parameters=require_parameters,
            allow_fallbacks=allow_fallbacks,
            provider_order=provider_order,
            provider_only=provider_only,
            provider_ignore=provider_ignore,
            min_request_interval_sec=min_request_interval_sec,
            vertexai=vertexai,
            project=project,
            location=location,
            hf_device=hf_device,
            hf_dtype=hf_dtype,
            max_output_tokens=max_output_tokens,
            structured_string_max_length=structured_string_max_length,
            max_retries=max_retries,
            initial_retry_delay_sec=initial_retry_delay_sec,
            max_retry_delay_sec=max_retry_delay_sec,
        )

        debug_payload = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "image_path": image_path,
            "raw_reply": raw_content,
        }
        try:
            if task_spec is not None:
                result = task_spec.parse_reply(raw_content, last_action)
            else:
                view_description, reasoning, action, summary, selected_cube, identification = _parse_json_reply(
                    raw_content,
                    previous_action=last_action,
                )
                result = (
                    view_description,
                    reasoning,
                    action,
                    summary,
                    selected_cube,
                    identification,
                    {},
                )
            view_description, reasoning, action, summary, selected_cube, identification, task_payload = result
            return (
                view_description,
                reasoning,
                action,
                summary,
                selected_cube,
                thinking,
                identification,
                task_payload,
                debug_payload,
            )
        except (ValueError, KeyError, TypeError) as exc:
            last_raw_content = raw_content
            last_thinking = thinking
            last_exc = exc
            if client_type == "google" and parse_attempt < parse_attempts:
                print(
                    f"  [Google] Malformed JSON reply on attempt {parse_attempt}/{parse_attempts}. "
                    "Retrying...",
                    flush=True,
                )
                continue
            raise ValueError(f"Could not parse JSON model response: {raw_content}") from exc

    raise ValueError(
        f"Could not parse JSON model response: {last_raw_content}"
    ) from last_exc
