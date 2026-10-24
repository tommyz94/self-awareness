"""Canonical experiment names for new runs; old data IDs belong to analysis."""
from __future__ import annotations

PAPER_EXPERIMENTS = {
    "e1": "e1",
    "e2": "e2",
    "e3": "e3",
    "e4": "e4",
    "e5": "e5",
}

EXPERIMENTAL_EXPERIMENTS = {
    name: f"experimental/{name}" for name in (
        "direct_perception", "front_identity", "exploration", "embodiment",
        "corridor", "mirror_side_choice",
    )
}


def resolve_experiment_selection(
    experiment_id: str | None,
    paper_experiment_id: str | None = None,
) -> tuple[str | None, str | None]:
    """Official bare IDs and paper/ names agree; prototypes require a namespace."""
    canonical_id = experiment_id
    if experiment_id and experiment_id.startswith("paper/"):
        public_id = experiment_id.removeprefix("paper/")
        if public_id not in PAPER_EXPERIMENTS:
            raise ValueError(f"Unknown paper experiment {experiment_id!r}.")
        if paper_experiment_id not in (None, public_id):
            raise ValueError("Conflicting paper experiment identifiers.")
        paper_experiment_id = public_id
        canonical_id = public_id
    elif experiment_id in PAPER_EXPERIMENTS:
        if paper_experiment_id not in (None, experiment_id):
            raise ValueError("Conflicting paper experiment identifiers.")
        paper_experiment_id = experiment_id
    elif experiment_id is not None:
        if experiment_id not in EXPERIMENTAL_EXPERIMENTS.values():
            raise ValueError(
                f"Unknown experiment {experiment_id!r}. Official IDs are e1-e5; "
                "use experimental/<name> for prototypes. Old data IDs are resolved only during analysis."
            )

    if paper_experiment_id is not None:
        expected = PAPER_EXPERIMENTS.get(paper_experiment_id)
        if expected is None or canonical_id != expected:
            raise ValueError(
                f"Paper {paper_experiment_id!r} requires experiment {expected!r}, "
                f"not {canonical_id!r}."
            )
    return canonical_id, paper_experiment_id
