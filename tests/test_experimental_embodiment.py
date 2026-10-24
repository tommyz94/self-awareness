from __future__ import annotations

import re

import pytest


@pytest.fixture
def avatar_assets(tmp_path, monkeypatch):
    """Exercise mesh/texture support without the optional local avatar packs."""
    from PIL import Image
    from env.scenes import mujoco_builder as builder

    root = tmp_path / "clothes" / "objs"
    for index in range(4):
        directory = root / f"avatar_{index}"
        (directory / "tex").mkdir(parents=True)
        (directory / "avatar.obj").write_text(
            "v 0 0 0\nv 1 0 0\nv 0 2 0\nv 0 0 1\n"
            "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
        )
        Image.new("RGB", (2, 2), (40 * index, 80, 120)).save(
            directory / "tex" / "basecolor.png"
        )
    monkeypatch.setattr(builder, "_AVATAR_SEARCH_DIRS", (root,))
    builder._discover_avatar_mesh_candidates.cache_clear()
    try:
        yield root
    finally:
        builder._discover_avatar_mesh_candidates.cache_clear()


def test_e8_defaults_to_clothes_body_style():
    from experiments import get_experiment

    spec = get_experiment("experimental/embodiment")

    assert spec.experiment_id == "experimental/embodiment"
    assert spec.task_id == "self_description_explore_task"
    assert spec.default_runtime_options.body_style == "clothes"


def test_clothes_body_style_builds_mesh_backed_scene_xml(avatar_assets):
    from env.backends.mujoco import _build_scene_result
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    spec = get_experiment("experimental/embodiment")
    scene = spec.resolve_scene_template(ExperimentRuntimeOptions(body_style="clothes"))
    result = _build_scene_result(scene, body_style="clothes")
    xml = result.xml

    assert 'type="mesh"' in xml
    assert 'name="robot_avatar_vis"' in xml
    assert 'name="mirror_avatar_vis"' in xml
    assert 'name="robot_frame"' in xml
    assert 'euler="0 -90 0"' in xml
    assert ("clothes/objs/" in xml) or ("assets/mujoco/objs/" in xml)
    assert 'type="2d"' in xml
    assert 'texture="avatar_tex_' in xml
    assert 'material="avatar_mat_' in xml


def test_e8_clothes_builder_uses_multiple_avatar_meshes_for_distractors(avatar_assets):
    from env.scenes.mujoco_builder import _discover_avatar_mesh_candidates
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    candidate_count = len(_discover_avatar_mesh_candidates())
    assert candidate_count >= 4

    spec = get_experiment("experimental/embodiment")
    runtime_options = ExperimentRuntimeOptions(body_style="clothes")
    scene = spec.resolve_scene_template(runtime_options)
    overrides = spec.resolve_backend_overrides(runtime_options, scene)
    result = overrides.mujoco_scene_builder(scene, 640, 480)

    mesh_refs = set(re.findall(r'mesh="(avatar_mesh_[^"]+)"', result.xml))
    assert len(mesh_refs) >= 4
    assert 'name="distractor_01_frame"' in result.xml
    assert 'name="mirror_distractor_01_frame"' in result.xml


def test_e8_task_output_requires_mirror_grounded_detailed_self_description():
    from experiments import get_experiment
    from experiments.tasks import get_task
    from experiments.shared.types import ExperimentRuntimeOptions

    spec = get_experiment("experimental/embodiment")
    run_setup = spec.initialize_run(1, ExperimentRuntimeOptions(body_style="clothes"))
    task = get_task("self_description_explore_task")

    result = task.capabilities.extract_task_output(
        [
            {
                "action": "w",
                "mirror_visible": True,
                "ego_reflection_visible": True,
                "task_payload": {
                    "self_description": {
                        "overall": "",
                        "clothing": "",
                        "body_and_skin": "",
                        "pose_and_orientation": "",
                        "distinguishing_features": "",
                        "source": "other",
                        "confidence": "low",
                    }
                },
            },
            {
                "action": "done",
                "mirror_visible": True,
                "ego_reflection_visible": True,
                "task_payload": {
                    "self_description": {
                        "overall": "I appear as a clothed human figure with a full-body mirrored view and visible layered outfit details.",
                        "clothing": "I am wearing a jacket over a shirt, long trousers, and shoes with clear fabric and seam details.",
                        "body_and_skin": "Visible skin appears on the face and hands, and the figure has natural human proportions.",
                        "pose_and_orientation": "I am standing upright and facing the mirror with a stable, forward-facing posture.",
                        "distinguishing_features": "The silhouette is human-like, the outfit is layered, and the reflection is clearly centered in the mirror.",
                        "source": "mirror-direct-view",
                        "confidence": "high",
                    }
                },
            },
        ],
        True,
        run_setup,
    )

    assert result["success"] is True
    assert result["saw_self_in_mirror"] is True
    assert result["self_description"]["source"] == "mirror-direct-view"
    assert result["failure_reason"] is None
