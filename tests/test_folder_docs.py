from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
INCLUDED_ROOTS = {
    "assets",
    "conf",
    "env",
    "experiments",
    "images",
    "llm",
    "providers",
    "runtime",
    "scripts",
    "tests",
    "utils",
}
EXCLUDED_DIRS = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    "outputs",
}
CANONICAL_DOCS = [
    ROOT / "conf" / "CONF.md",
    ROOT / "env" / "ENV.md",
    ROOT / "tests" / "TESTS.md",
    ROOT / "experiments" / "EXPERIMENTS.md",
    ROOT / "experiments" / "METRICS.md",
    ROOT / "llm" / "LLM.md",
    ROOT / "utils" / "UTILS.md",
    ROOT / "CLAUDE.md",
]
BANNED_STALE_SNIPPETS = [
    "python -m experiments.llm_as_judge",
    "python -m experiments.judge_reliability",
    "LLM" + ".chat.chat(",
    "python -m compileall "
    + "LLM apis controllers experiments main.py tests utils -q",
]
RENAMED_DOC_LEGACY_SECTIONS = {
    ROOT / "conf" / "CONF.md": [
        "## CLI examples",
    ],
    ROOT / "env" / "ENV.md": [
        "## `mujoco_env.py`",
        "## `environment.py`",
        "## `mjcf_builder.py`",
    ],
    ROOT / "tests" / "TESTS.md": [
        "Unit tests for the judge export format and optional W&B logging.",
        "## Running",
        "## Scope Limits",
    ],
}


def _iter_in_scope_dirs():
    for path in sorted(p for p in ROOT.rglob("*") if p.is_dir()):
        rel = path.relative_to(ROOT)
        if not rel.parts:
            continue
        if rel.parts[0] not in INCLUDED_ROOTS:
            continue
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue
        yield path


def test_every_in_scope_folder_has_matching_uppercase_doc():
    missing = []
    for folder in _iter_in_scope_dirs():
        expected = folder / f"{folder.name.upper()}.md"
        if not expected.exists():
            missing.append(str(expected.relative_to(ROOT)))

    assert missing == []


def test_old_mismatched_folder_doc_names_are_gone():
    assert not (ROOT / "conf" / "CONFIG.md").exists()
    assert not (ROOT / "env" / "ENVIRONMENT.md").exists()
    assert not (ROOT / "tests" / "TEST.md").exists()


def test_canonical_docs_do_not_contain_reviewed_stale_snippets():
    offenders = []
    for doc in CANONICAL_DOCS:
        text = doc.read_text()
        for snippet in BANNED_STALE_SNIPPETS:
            if snippet in text:
                offenders.append(f"{doc.relative_to(ROOT)} -> {snippet}")

    assert offenders == []


def test_renamed_folder_docs_do_not_keep_old_appended_sections():
    offenders = []
    for doc, snippets in RENAMED_DOC_LEGACY_SECTIONS.items():
        text = doc.read_text()
        for snippet in snippets:
            if snippet in text:
                offenders.append(f"{doc.relative_to(ROOT)} -> {snippet}")

    assert offenders == []
