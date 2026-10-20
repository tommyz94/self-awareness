from pathlib import Path
import re
import pytest


ROOT = Path(__file__).resolve().parent.parent
SKILLS_ROOT = ROOT / ".claude" / "skills"
pytestmark = pytest.mark.skipif(
    not SKILLS_ROOT.exists(), reason="Optional machine-local assistant skills are not installed.",
)
SKILL_NAMES = [
    "experiment-extension",
    "gather-repo-infos",
    "judge-pipeline",
    "output-contract-guard",
    "plan-before-code",
    "provider-adapter",
    "update-folder-docs",
]
DESCRIPTION_BANNED_SNIPPETS = [
    "Start with repo inspection and planning before edits.",
    "before edits.",
]


def _read(path: Path) -> str:
    return path.read_text()


def _skill_description(skill_name: str) -> str:
    text = _read(SKILLS_ROOT / skill_name / "SKILL.md")
    match = re.search(r"^description:\s*(.+)$", text, flags=re.MULTILINE)
    assert match is not None, f"Missing description for {skill_name}"
    return match.group(1).strip()


def test_all_repo_local_skills_have_openai_agent_manifest():
    missing = []
    for skill_name in SKILL_NAMES:
        agent_manifest = SKILLS_ROOT / skill_name / "agents" / "openai.yaml"
        if not agent_manifest.exists():
            missing.append(str(agent_manifest.relative_to(ROOT)))

    assert missing == []


def test_skill_descriptions_use_trigger_style_not_workflow_style():
    offenders = []
    for skill_name in SKILL_NAMES:
        description = _skill_description(skill_name)
        if not description.startswith("Use when"):
            offenders.append(f"{skill_name}: does not start with 'Use when'")
        for banned in DESCRIPTION_BANNED_SNIPPETS:
            if banned in description:
                offenders.append(f"{skill_name}: contains workflow snippet {banned!r}")

    assert offenders == []


def test_plan_before_code_agent_manifest_explicitly_mentions_skill_name():
    text = _read(SKILLS_ROOT / "plan-before-code" / "agents" / "openai.yaml")

    assert "$plan-before-code" in text


def test_update_folder_docs_skill_matches_uppercase_folder_doc_convention():
    text = _read(SKILLS_ROOT / "update-folder-docs" / "SKILL.md")

    assert "<FOLDERNAME_IN_CAPS>.md" in text
    assert "create a concise `README.md`" not in text
