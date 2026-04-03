from __future__ import annotations

import pytest


def test_mujoco_arm_rest_pose_keeps_arms_down_and_lightly_flexed():
    from env.backends.rest_pose import MUJOCO_ARM_REST_POSE as _ARM_REST_POSE

    assert _ARM_REST_POSE["shoulder1_right"] == pytest.approx(_ARM_REST_POSE["shoulder1_left"])
    assert _ARM_REST_POSE["shoulder2_right"] == pytest.approx(_ARM_REST_POSE["shoulder2_left"])
    assert _ARM_REST_POSE["elbow_right"] == pytest.approx(_ARM_REST_POSE["elbow_left"])
    assert 0.75 <= _ARM_REST_POSE["shoulder1_right"] <= 1.0
    assert -1.25 <= _ARM_REST_POSE["shoulder2_right"] <= -0.85
    assert -0.75 <= _ARM_REST_POSE["elbow_right"] <= -0.40


def test_e5_mujoco_reset_applies_arm_rest_pose_to_distractor_prefixes():
    from experiments.specs.e5.experiment import _apply_mujoco_reset
    from experiments.specs.types import RunSetup, SceneObjectSpec, SceneSpec

    applied_prefixes: list[str] = []

    class _FakeEnv:
        _spawn_y = -1.0
        _mirror_plane_x = -0.1

        def _jnt_addr_if_exists(self, name: str):
            return name

        def _set_body_pose(self, *args, **kwargs):
            return None

        def _set_color(self, *args, **kwargs):
            return None

        def _geom_ids_with_prefix(self, prefix: str):
            return [prefix]

        def _apply_arm_rest_pose_for_prefix(self, prefix: str):
            applied_prefixes.append(prefix)

        def _is_in_mirror_aperture(self, z: float) -> bool:
            return True

        def _park_body(self, jnt_addr):
            return None

    run_setup = RunSetup(
        agent_color_name="red",
        scene_spec=SceneSpec(
            scene_id="mirror_room_robot_distractors",
            mirror_enabled=True,
            objects=[
                SceneObjectSpec(
                    object_id="distractor_01",
                    kind="distractor_robot",
                    position=(-3.8, -0.9),
                    color="blue",
                    metadata={"yaw": 0.0, "reflect_in_mirror": True},
                ),
                SceneObjectSpec(
                    object_id="distractor_02",
                    kind="distractor_robot",
                    position=(-3.8, 0.9),
                    color="blue",
                    metadata={"yaw": 0.0, "reflect_in_mirror": True},
                ),
            ],
        ),
    )

    _apply_mujoco_reset(_FakeEnv(), run_setup)

    assert applied_prefixes == [
        "distractor_01_",
        "mirror_distractor_01_",
        "distractor_02_",
        "mirror_distractor_02_",
    ]
