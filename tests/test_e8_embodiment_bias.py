from __future__ import annotations


def test_e8_defaults_to_clothes_body_style():
    from experiments.specs import get_experiment

    spec = get_experiment("e8")

    assert spec.experiment_id == "e8"
    assert spec.task_id == "self_description_explore_task"
    assert spec.default_runtime_options.body_style == "clothes"


def test_clothes_body_style_builds_mesh_backed_scene_xml():
    from env.backends.mujoco import _build_scene_result
    from experiments.specs import get_experiment
    from experiments.specs.types import ExperimentRuntimeOptions

    spec = get_experiment("e8")
    scene = spec.resolve_scene_template(ExperimentRuntimeOptions(body_style="clothes"))
    result = _build_scene_result(scene, body_style="clothes")
    xml = result.xml

    assert 'type="mesh"' in xml
    assert 'name="robot_avatar_vis"' in xml
    assert 'name="mirror_avatar_vis"' in xml
    assert ("clothes/objs/" in xml) or ("assets/mujoco/objs/" in xml)
    assert 'type="2d"' in xml
    assert 'texture="avatar_tex_' in xml
    assert 'material="avatar_mat_' in xml


def test_e8_task_output_requires_mirror_grounded_detailed_self_description():
    from experiments.specs import get_experiment
    from experiments.tasks import get_task
    from experiments.specs.types import ExperimentRuntimeOptions

    spec = get_experiment("e8")
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
