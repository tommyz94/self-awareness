"""Public paper names and stable implementation IDs.

Historical JSON files and implementation packages keep their original IDs.
Use ``paper/e1`` through ``paper/e5`` for manuscript-facing commands; never
rename historical run directories to translate between the two vocabularies.
"""
from __future__ import annotations

PAPER_EXPERIMENTS = {
    "e1": "e1",
    "e2": "e2",
    "e3": "e3",
    "e4": "e5",
    "e5": "e7",
}

EXTRA_EXPERIMENTS = {
    "direct_perception": "e0",
    "exploration": "e6",
    "embodiment": "e8",
    "corridor": "corridor",
}


def resolve_experiment_selection(
    experiment_id: str | None,
    paper_experiment_id: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve an explicit namespace without reinterpreting bare legacy IDs."""
    implementation_id = experiment_id
    if experiment_id and experiment_id.startswith("paper/"):
        public_id = experiment_id.removeprefix("paper/")
        if public_id not in PAPER_EXPERIMENTS:
            raise ValueError(f"Unknown paper experiment {experiment_id!r}.")
        if paper_experiment_id not in (None, public_id):
            raise ValueError("Conflicting paper experiment identifiers.")
        paper_experiment_id = public_id
        implementation_id = PAPER_EXPERIMENTS[public_id]
    elif experiment_id and experiment_id.startswith("extras/"):
        name = experiment_id.removeprefix("extras/")
        if name not in EXTRA_EXPERIMENTS:
            raise ValueError(f"Unknown extra experiment {experiment_id!r}.")
        implementation_id = EXTRA_EXPERIMENTS[name]

    if paper_experiment_id is not None:
        expected = PAPER_EXPERIMENTS.get(paper_experiment_id)
        if expected is None or implementation_id != expected:
            raise ValueError(
                f"Paper {paper_experiment_id!r} requires implementation {expected!r}, "
                f"not {implementation_id!r}."
            )
    return implementation_id, paper_experiment_id
