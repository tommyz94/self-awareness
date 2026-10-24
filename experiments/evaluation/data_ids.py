"""Translate data identities only; runtime experiments always use paper IDs."""

PAPER_TO_CALCULATOR = {"e1": "e1", "e2": "e2", "e3": "e3", "e4": "e5", "e5": "e7"}
CALCULATOR_TO_PAPER = {old: paper for paper, old in PAPER_TO_CALCULATOR.items()}


def calculator_id(source_id, metadata, *, scheme=None):
    """New records declare paper-v1; unversioned original datasets keep old IDs."""
    context = metadata.get("run_context") or {}
    if not isinstance(context, dict):
        raise ValueError("run_context must be an object when present")
    recorded_scheme = metadata.get("experiment_id_scheme") or context.get("experiment_id_scheme")
    paper_id = metadata.get("paper_experiment_id") or context.get("paper_experiment_id")
    if scheme and recorded_scheme and scheme != recorded_scheme:
        raise ValueError("Manifest and run disagree on experiment_id_scheme")
    scheme = scheme or recorded_scheme
    if scheme == "paper-v1":
        if source_id not in PAPER_TO_CALCULATOR:
            raise ValueError(f"Unsupported paper experiment: {source_id!r}")
        expected = paper_id or source_id
        if expected != source_id:
            raise ValueError("Paper identity disagrees with experiment_id")
        return PAPER_TO_CALCULATOR[source_id]
    if scheme not in (None, "legacy-v1"):
        raise ValueError(f"Unsupported data identity scheme: {scheme!r}")
    if source_id not in CALCULATOR_TO_PAPER:
        raise ValueError(f"Unversioned data ID {source_id!r} is not in the original benchmark; new records must declare paper-v1")
    expected = paper_id or CALCULATOR_TO_PAPER[source_id]
    if expected != CALCULATOR_TO_PAPER[source_id]:
        raise ValueError("Original data identity disagrees with paper_experiment_id")
    return source_id
