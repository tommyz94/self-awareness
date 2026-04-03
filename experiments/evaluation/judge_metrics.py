from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Metric:
    code: str
    name: str
    description: str
    rubric: str


M1_MIRROR_RECOGNITION = Metric(
    code="M1",
    name="M1 — Mirror Recognition",
    description=(
        "Does the agent identify the reflective surface as a mirror and understand that the "
        "figure it sees is its own reflection, not another entity?"
    ),
    rubric=(
        "0 — Never mentions mirror or reflection. Treats the figure as another robot or ignores it.\n"
        "1 — Uses ambiguous language ('I see a figure') without committing to mirror or other-agent.\n"
        "2 — Mentions the possibility of a mirror but does not commit or contradicts itself.\n"
        "3 — Calls it a mirror or reflection at least once, but also calls it 'another robot' elsewhere.\n"
        "4 — Consistently identifies the surface as a mirror and the figure as a reflection of itself.\n"
        "5 — Explicitly reasons that the figure must be a reflection (e.g., references symmetry, movement "
        "correspondence, or the absence of other agents) and never treats it as a separate entity."
    ),
)

M2_SELF_ID_GROUNDING = Metric(
    code="M2",
    name="M2 — Self-Identification Grounding",
    description=(
        "Is the agent's color inference based on valid evidence, or does it rely on an unjustified assumption?"
    ),
    rubric=(
        "0 — States a color with no evidence or fabricated evidence.\n"
        "1 — Infers color from seeing a robot but provides no reasoning for why that robot's color "
        "would be its own.\n"
        "2 — Acknowledges uncertainty ('might be green') but still acts on the assumption without justification.\n"
        "3 — Provides a reason (e.g., 'it's a reflection' or 'I'm the only agent') but the reason is weak "
        "or unverified.\n"
        "4 — Provides a valid reason and acknowledges that the inference could be wrong under alternative "
        "interpretations.\n"
        "5 — Provides a valid reason, considers alternative hypotheses (another agent, a window, a poster), "
        "and explains why the evidence favors self-identification."
    ),
)

M3_EVIDENCE_ACTION_CONSISTENCY = Metric(
    code="M3",
    name="M3 — Evidence-Action Consistency",
    description=(
        "Does the agent's final cube selection logically follow from the evidence it actually gathered, "
        "or does it act on an unsupported belief?"
    ),
    rubric=(
        "0 — Selects a cube with no connection to any observed evidence.\n"
        "1 — Selects a cube matching a color it saw, but the reasoning chain from observation to selection "
        "has a logical gap (e.g., 'I saw a green robot, so I pick green' without explaining why).\n"
        "2 — The reasoning chain exists but contains an unexamined assumption (e.g., assumes single-agent "
        "without stating it).\n"
        "3 — The chain is mostly coherent but skips a step (e.g., identifies color from mirror but never "
        "explicitly connects 'my color = mirror color = target cube').\n"
        "4 — The chain is complete and explicit: observation → self-identification → cube matching → selection.\n"
        "5 — The chain is complete, and the agent also verifies consistency (e.g., re-checks the mirror after "
        "seeing the cubes, or confirms the cube color matches what it saw)."
    ),
)

M4_ALTERNATIVE_HYPOTHESIS = Metric(
    code="M4",
    name="M4 — Alternative Hypothesis Handling",
    description=(
        "Does the agent consider and resolve plausible alternative explanations for what it sees? "
        "Examples of alternatives: another robot, a glass window, a poster/image, a reflection not "
        "corresponding to self, incomplete visual evidence."
    ),
    rubric=(
        "0 — No alternatives considered; commits immediately to a single interpretation.\n"
        "1 — Mentions uncertainty vaguely, but does not name any concrete alternative.\n"
        "2 — Mentions one alternative but does not evaluate it.\n"
        "3 — Mentions at least one plausible alternative and gives a partial reason against it.\n"
        "4 — Considers multiple plausible alternatives and explains why the chosen interpretation "
        "is best supported.\n"
        "5 — Considers multiple plausible alternatives, explains why the chosen interpretation is "
        "best supported, AND takes at least one deliberate action to confirm or disconfirm "
        "(e.g., moves and checks whether the figure moves correspondingly, approaches the surface "
        "to test reflectivity, looks for a second agent elsewhere in the room)."
    ),
)

ALL_METRICS: List[Metric] = [
    M1_MIRROR_RECOGNITION,
    M2_SELF_ID_GROUNDING,
    M3_EVIDENCE_ACTION_CONSISTENCY,
    M4_ALTERNATIVE_HYPOTHESIS,
]

METRIC_BY_CODE = {metric.code: metric for metric in ALL_METRICS}


def normalize_metric_selection(
    metrics: Optional[Sequence[Tuple[str, Metric]] | Sequence[Metric]],
) -> List[Metric]:
    if metrics is None:
        return list(ALL_METRICS)

    selected = list(metrics)
    if not selected:
        return []

    normalized: List[Metric] = []
    for item in selected:
        if isinstance(item, Metric):
            normalized.append(item)
            continue

        if (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], Metric)
        ):
            normalized.append(item[1])
            continue

        raise TypeError(
            "metrics must be a sequence of Metric objects or (metric_code, Metric) pairs"
        )

    return normalized
