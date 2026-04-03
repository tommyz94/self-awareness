from __future__ import annotations

def test_mujoco_builder_scales_first_person_camera_height():
    from env.scenes.mujoco_builder import build_humanoid_scene

    fragments = build_humanoid_scene(
        mirror_present=True,
        spawn_pos=(-10.0, -1.0, 2.5),
        camera_height=0.5,
    )

    assert 'name="first_person"' in fragments.robot_body_xml
    assert 'pos="0.15 0.7250 0"' in fragments.robot_body_xml


def test_e5_mujoco_builder_respects_runtime_camera_height():
    from experiments.specs import get_experiment
    from experiments.specs.types import ExperimentRuntimeOptions

    spec = get_experiment("e5")
    runtime_options = ExperimentRuntimeOptions(distractors=2, camera_height=0.5)
    scene_template = spec.resolve_scene_template(runtime_options)
    overrides = spec.resolve_backend_overrides(runtime_options, scene_template)

    result = overrides.mujoco_scene_builder(scene_template, 640, 480)

    assert 'name="first_person"' in result.xml
    assert 'pos="0.15 0.7250 0"' in result.xml
