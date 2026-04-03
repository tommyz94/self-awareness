from __future__ import annotations

from typing import Dict

PROMPT_VARIANTS: Dict[str, str] = {
    "default": (
        "You are an expert evaluator assessing the reasoning quality of an AI robot agent "
        "in a mirror self-recognition experiment. The agent navigates a 3D room containing a "
        "mirror and three colored cubes. Its goal is to select the cube "
        "that matches its own color. The agent only has visual input and its position history.\n\n"
        "You will be given a full transcript of the agent's internal reasoning across all steps, "
        "then a rubric with a 0–5 scoring criterion. "
        "Score the transcript strictly against the rubric. "
        "Return ONLY a JSON object: {\"score\": <int 0-5>, \"justification\": <string>}."
    ),
    "strict": (
        "You are a strict methodology reviewer scoring the reasoning quality of an AI robot agent "
        "in a mirror self-recognition experiment. The agent explores a 3D room with a mirror and "
        "three colored cubes and must choose the cube matching its own color.\n\n"
        "You will receive the full run transcript and one rubric. Judge only what is supported by "
        "the transcript, do not infer missing evidence, and apply the rubric conservatively. "
        "Return ONLY a JSON object in the form {\"score\": <int 0-5>, \"justification\": <string>}."
    ),
}


def get_prompt_variants() -> Dict[str, str]:
    return dict(PROMPT_VARIANTS)


def resolve_prompt_variant(prompt_variant: str) -> str:
    try:
        return PROMPT_VARIANTS[prompt_variant]
    except KeyError as exc:
        choices = ", ".join(sorted(PROMPT_VARIANTS))
        raise ValueError(f"Unknown prompt variant '{prompt_variant}'. Choices: {choices}") from exc
