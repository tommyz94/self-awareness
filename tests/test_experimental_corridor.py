from __future__ import annotations

import math
import importlib
import sys
from contextlib import redirect_stdout
from types import ModuleType, SimpleNamespace
from io import StringIO


def _build_cube_run_setup(*, wrong_color: str | None = None):
    from experiments import RunSetup, SceneSpec

    return RunSetup(
        agent_color_name="red",
        wrong_color=wrong_color,
        scene_spec=SceneSpec(
            scene_id="mirror_room",
            mirror_enabled=True,
            objects=[
                {"type": "cube", "color": "red", "position": (-10.0, -4.0)},
                {"type": "cube", "color": "green", "position": (-8.0, -4.0)},
                {"type": "cube", "color": "blue", "position": (-6.0, -4.0)},
            ],
        ),
        cube_positions={
            "red": (-10.0, -4.0),
            "green": (-8.0, -4.0),
            "blue": (-6.0, -4.0),
        },
    )


def _build_corridor_run_setup(*, use_mirror: bool = True, step_target: int | None = None):
    from experiments import RunSetup, SceneSpec

    run_setup = RunSetup(
        agent_color_name="red",
        scene_spec=SceneSpec(
            scene_id="narrow_corridor_mirror",
            mirror_enabled=use_mirror,
            layout_kind="corridor",
            length=10.0,
            width=6.0,
            spawn_pos=(-4.5, 0.0),
            spawn_yaw=0.0,
            mirror_plane_x=5.0,
            mirror_aperture_mode="full" if use_mirror else "none",
            mirror_window_z=None,
            objects=[],
        ),
    )
    run_setup.metadata.update(
        {
            "e4_variant": "mirror" if use_mirror else "long",
            "expected_done_step": step_target if step_target is not None else (10 if use_mirror else 20),
        }
    )
    return run_setup


def test_e4_registry_resolves_to_corridor_scene_and_task():
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    spec = get_experiment("experimental/front_identity")
    run_setup = spec.initialize_run(1, ExperimentRuntimeOptions(use_mirror=False))

    assert spec.task_id == "front_identity_task"
    assert spec.scene_spec.scene_id == "narrow_corridor_mirror"
    assert spec.scene_spec.layout_kind == "corridor"
    assert spec.scene_spec.mirror_enabled is True
    assert spec.scene_spec.length == 10.0
    assert spec.scene_spec.width == 6.0
    assert spec.scene_spec.spawn_pos == (-4.5, 0.0)
    assert spec.scene_spec.mirror_plane_x == 5.0
    assert spec.scene_spec.mirror_aperture_mode == "full"
    assert run_setup.metadata["front_identity_variant"] in {"self", "other"}
    assert run_setup.metadata["expected_answer"] in {"yes", "no"}
    assert run_setup.metadata["secondary_initial_pos"] == [14.5, 0.0]
    assert math.isclose(run_setup.metadata["secondary_initial_yaw"], math.pi)


def test_build_corridor_scene_can_disable_mirror():
    from experiments.shared.scenes import build_corridor_scene

    scene_spec = build_corridor_scene(use_mirror=False)

    assert scene_spec.scene_id == "narrow_corridor_mirror"
    assert scene_spec.layout_kind == "corridor"
    assert scene_spec.mirror_enabled is False
    assert scene_spec.mirror_plane_x == 5.0
    assert scene_spec.mirror_aperture_mode == "none"


def test_e4_variant_sampling_can_generate_both_variants():
    import random
    from experiments.experimental.front_identity import experiment as e4_experiment

    random.seed(7)
    variants = {e4_experiment._sample_variant() for _ in range(40)}
    assert variants == {"self", "other"}


def test_e4_runtime_option_does_not_change_corridor_geometry(monkeypatch):
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    spec = get_experiment("experimental/front_identity")

    monkeypatch.setattr("experiments.experimental.front_identity.experiment._sample_variant", lambda: "other")
    run_setup_other = spec.initialize_run(1, ExperimentRuntimeOptions(use_mirror=False))
    assert run_setup_other.scene_spec.mirror_enabled is True
    assert run_setup_other.metadata["front_identity_variant"] == "other"
    assert run_setup_other.metadata["expected_answer"] == "no"

    monkeypatch.setattr("experiments.experimental.front_identity.experiment._sample_variant", lambda: "self")
    run_setup_self = spec.initialize_run(2, ExperimentRuntimeOptions(use_mirror=True))
    assert run_setup_self.scene_spec.mirror_enabled is True
    assert run_setup_self.metadata["front_identity_variant"] == "self"
    assert run_setup_self.metadata["expected_answer"] == "yes"


def test_e4_variant_can_be_forced_via_env(monkeypatch):
    from experiments.experimental.front_identity import experiment as e4_experiment

    monkeypatch.setenv("E4_FRONT_IDENTITY_VARIANT", "other")
    assert e4_experiment._sample_variant() == "other"
    monkeypatch.setenv("E4_FRONT_IDENTITY_VARIANT", "self")
    assert e4_experiment._sample_variant() == "self"


def test_mujoco_reset_handles_corridor_runs_without_cube_joints():
    from env.backends.mujoco import MuJoCoEnv
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    spec = get_experiment("experimental/front_identity")
    run_setup = spec.initialize_run(1, ExperimentRuntimeOptions(use_mirror=True))

    body_pos_calls = {"count": 0}

    dummy_env = SimpleNamespace(
        _body_style="default",
        _secondary_present=False,
        _cube_jnt_addrs={},
        _mirror_cube_jnt_addrs={},
        _cube_geom_ids={},
        _mirror_cube_geom_ids={},
        _robot_geom_ids=[],
        _mirror_geom_ids=[],
        _reset_applier=None,
        _model=object(),
        _data=object(),
        _mj=SimpleNamespace(mj_forward=lambda _model, _data: None),
        _set_color=lambda _geom_ids, _rgba: None,
        _set_body_pos=lambda _addr, _x, _y, _z: body_pos_calls.__setitem__(
            "count",
            body_pos_calls["count"] + 1,
        ),
        _park_body=lambda _addr: None,
        _apply_arm_rest_pose=lambda: None,
        _sync_robot=lambda: None,
        _sync_mirror=lambda: None,
        _apply_wall_color=lambda _rgba: None,
        _is_in_mirror_aperture=lambda _z: True,
    )

    MuJoCoEnv.reset(dummy_env, run_setup)

    assert dummy_env._object_pos_2d == {}
    assert body_pos_calls["count"] == 0


def test_e4_other_variant_reset_applier_places_secondary_agent_in_front():
    from experiments import get_experiment
    from experiments.shared.types import ExperimentRuntimeOptions

    spec = get_experiment("experimental/front_identity")
    overrides = spec.resolve_backend_overrides(ExperimentRuntimeOptions(), spec.scene_spec)
    run_setup = spec.initialize_run(1, ExperimentRuntimeOptions())
    run_setup.metadata["front_identity_variant"] = "other"
    run_setup.metadata["expected_answer"] = "no"

    body_pose_calls = []
    def _default_try_move_secondary(_dx, _dz):
        return False

    dummy_env = SimpleNamespace(
        _body_style="default",
        _secondary_present=True,
        _mirror_jnt_addr="mirror_root",
        _spawn_y=-1.0,
        _secondary_active=False,
        _wall_thickness=0.2,
        _scene_length=10.0,
        _scene_width=6.0,
        _mirror_plane_x=5.0,
        _mirror_ego_geom_ids=[1, 2, 3],
        _sync_mirror=lambda: None,
        _try_move_secondary=_default_try_move_secondary,
        _park_body=lambda _addr: None,
        _set_body_pose=lambda *args: body_pose_calls.append(args),
    )
    overrides.mujoco_reset_applier(dummy_env, run_setup)

    assert dummy_env._secondary_x == 14.5
    assert dummy_env._secondary_z == 0.0
    assert math.isclose(dummy_env._secondary_yaw, math.pi)
    assert dummy_env._secondary_active is True
    assert dummy_env._mirror_ego_geom_ids == []
    assert body_pose_calls[-1] == ("mirror_root", 14.5, -1.0, 0.0, math.pi)
    assert dummy_env._try_move_secondary is not _default_try_move_secondary
    assert dummy_env._try_move_secondary(-2.0, 0.0) is True
    assert dummy_env._secondary_x == 12.5
    assert dummy_env._secondary_z == 0.0
    assert dummy_env._try_move_secondary(-8.0, 0.0) is False


def test_e4_diagonal_actions_fall_back_to_forward_when_lateral_component_is_blocked():
    from env.backends.mujoco import MuJoCoEnv

    attempted = []

    def _move_fn(dx, dz):
        attempted.append((round(dx, 3), round(dz, 3)))
        return abs(dz) < 1e-6

    dummy_env = SimpleNamespace(
        _body_style="default",
        _corridor_forward_step=1.0,
        _corridor_lane_step=2.0,
        _corridor_action_delta=lambda action, yaw: MuJoCoEnv._corridor_action_delta(
            SimpleNamespace(
                _corridor_forward_step=1.0,
                _corridor_lane_step=2.0,
            ),
            action,
            yaw=yaw,
        ),
    )

    move_debug = MuJoCoEnv._attempt_corridor_move_with_forward_fallback(
        dummy_env,
        "a",
        yaw=0.0,
        move_fn=_move_fn,
    )

    assert attempted == [(1.0, -2.0), (1.0, 0.0)]
    assert move_debug["success"] is True
    assert move_debug["used_forward_fallback"] is True
    assert move_debug["applied_dx"] == 1.0
    assert move_debug["applied_dz"] == 0.0

def test_e1_e2_e3_registry_presets_keep_cube_task_behavior():
    from experiments import get_experiment
    from experiments.paper.e1.experiment import _E1_WIDE_MIRROR_SLOTS

    e1 = get_experiment("e1")
    e2 = get_experiment("e2")
    e3 = get_experiment("e3")

    assert e1.task_id == "cube_task"
    assert e2.task_id == "cube_task"
    assert e3.task_id == "cube_task"
    assert e1.scene_spec.turn_step_deg == 30.0
    assert e2.scene_spec.turn_step_deg == 30.0
    assert e3.scene_spec.turn_step_deg == 30.0
    assert e1.scene_spec.mirror_enabled is True
    assert e2.scene_spec.mirror_enabled is False
    assert e3.scene_spec.mirror_enabled is True

    e1_run = e1.run_initializer(1)
    e2_run = e2.run_initializer(1)
    e3_run = e3.run_initializer(1)

    assert e1_run.wrong_color is None
    assert e2_run.wrong_color is None
    assert e3_run.wrong_color is not None
    assert e3_run.wrong_color != e3_run.agent_color_name
    assert len(e1_run.scene_spec.objects) == 3
    assert len(e2_run.scene_spec.objects) == 3
    assert len(e3_run.scene_spec.objects) == 3
    assert e1_run.scene_spec.wall_color_rgba is None
    assert e2_run.scene_spec.wall_color_rgba is None
    assert e3_run.scene_spec.wall_color_rgba is None
    mirror_plane_x = e1_run.scene_spec.mirror_plane_x
    assert mirror_plane_x is not None
    reflected_corner_slots = {
        (round(2 * mirror_plane_x - x, 3), z)
        for (x, z) in _E1_WIDE_MIRROR_SLOTS
    }
    e1_positions = {tuple(obj.position) for obj in e1_run.scene_spec.objects if obj.kind == "cube"}
    e2_positions = {tuple(obj.position) for obj in e2_run.scene_spec.objects if obj.kind == "cube"}
    e3_positions = {tuple(obj.position) for obj in e3_run.scene_spec.objects if obj.kind == "cube"}
    e1_slots = set(_E1_WIDE_MIRROR_SLOTS) | reflected_corner_slots
    assert e1_positions <= e1_slots
    from experiments.shared.scenes import MIRROR_ROOM_WALL_CORNER_SLOTS
    corner_slots = set(MIRROR_ROOM_WALL_CORNER_SLOTS) | {
        (round(2 * mirror_plane_x - x, 3), z)
        for (x, z) in MIRROR_ROOM_WALL_CORNER_SLOTS
    }
    assert e2_positions <= corner_slots
    assert e3_positions <= corner_slots


def test_e1_e2_e3_support_runtime_random_wall_color():
    from experiments import ExperimentRuntimeOptions, get_experiment

    primary_colors = (
        (1.0, 0.0, 0.0),
        (0.0, 0.8, 0.0),
        (0.0, 0.3, 1.0),
    )

    for experiment_id in ("e1", "e2", "e3"):
        spec = get_experiment(experiment_id)
        run_setup = spec.initialize_run(
            1,
            ExperimentRuntimeOptions(random_wall_color=True),
        )
        rgba = run_setup.scene_spec.wall_color_rgba
        assert rgba is not None
        assert len(rgba) == 4
        assert rgba[3] == 1.0
        rgb = rgba[:3]
        assert min(
            math.sqrt(
                (rgb[0] - c[0]) ** 2
                + (rgb[1] - c[1]) ** 2
                + (rgb[2] - c[2]) ** 2
            )
            for c in primary_colors
        ) >= 0.55


def test_cube_task_prompts_respect_camera_mode():
    from experiments.tasks.cube import build_system_prompt, build_user_prompt

    run_setup = _build_cube_run_setup(wrong_color="blue")

    first_person = build_system_prompt(run_setup, third_person=False)
    third_person = build_system_prompt(run_setup, third_person=True)
    user_prompt = build_user_prompt(
        run_setup,
        history_text="No previous steps.\n",
        position_text="Current position: x=-10.0, z=2.5; facing N\n",
        bumped_wall=False,
        third_person=False,
    )

    assert "current first-person RGB image" in first_person
    assert "You are a blue robot." in first_person
    assert "current third-person RGB image" in third_person
    assert "turn left by 30 degrees" in first_person
    assert "turn right by 30 degrees" in first_person
    assert "selected cube" in user_prompt.lower()


def test_corridor_task_prompts_have_no_cube_language_and_support_third_person():
    from experiments.tasks.corridor import (
        GOAL_STATUS_SCHEMA,
        build_instruction,
        build_system_prompt,
        build_user_prompt,
    )

    run_setup = _build_corridor_run_setup()

    system_prompt = build_system_prompt(run_setup, third_person=True)
    user_prompt = build_user_prompt(
        run_setup,
        history_text="No previous steps.\n",
        position_text="Current position: x=-10.0, z=0.0; facing N\n",
        bumped_wall=True,
        third_person=True,
    )
    combined = f"{system_prompt}\n{user_prompt}".lower()

    assert "mirror wall" in build_instruction(run_setup).lower()
    assert "far wall" in build_instruction(run_setup).lower()
    assert "current third-person rgb image" in system_prompt.lower()
    assert "a = move forward-left diagonally" in system_prompt.lower()
    assert "d = move forward-right diagonally" in system_prompt.lower()
    assert "action must always be one of: w, a, d, done" in system_prompt.lower()
    assert "goal_status" in system_prompt
    assert GOAL_STATUS_SCHEMA["required"] == ["status", "reason", "confidence"]
    assert "reach the final row" in user_prompt.lower()
    assert "stop at the correct step" in user_prompt.lower()
    assert "why you believe you are at the corridor end" in user_prompt.lower()
    assert "selected cube" not in combined
    assert "identification" not in combined
    assert "further exploration is unnecessary" not in user_prompt.lower()


def test_navigation_build_messages_uses_task_specific_corridor_prompt():
    from llm.navigation.service import _build_messages
    from experiments.tasks.corridor import CORRIDOR_TASK_SPEC

    run_setup = _build_corridor_run_setup()
    messages = _build_messages(
        instruction="ignored",
        history=[],
        image_path=None,
        current_pos=(-10.0, 0.0),
        bumped_wall=True,
        current_dir="N",
        task_spec=CORRIDOR_TASK_SPEC,
        run_setup=run_setup,
        third_person=False,
    )

    assert "corridor" in messages[0]["content"].lower()
    assert "first-person" in messages[0]["content"].lower()
    assert "forward-left diagonally" in messages[0]["content"].lower()
    assert "forward-right diagonally" in messages[0]["content"].lower()
    assert "reach the final row" in messages[1]["content"].lower()
    assert "stop at the correct step" in messages[1]["content"].lower()
    assert "facing " not in messages[1]["content"].lower()
    assert "facing_before" not in messages[1]["content"].lower()
    assert "selected cube" not in messages[1]["content"].lower()
    assert messages[1]["content"].lower().count("hit a wall") == 1


def test_navigation_build_messages_keeps_facing_for_cube_task():
    from llm.navigation.service import _build_messages
    from experiments.tasks.cube import CUBE_TASK_SPEC

    run_setup = _build_cube_run_setup()
    history = [
        {
            "action": "w",
            "facing_before": "N",
            "view_description": "A plain room.",
            "reasoning": "I moved forward.",
            "summary": "Keep exploring.",
        }
    ]
    messages = _build_messages(
        instruction="ignored",
        history=history,
        image_path=None,
        previous_pos=(-10.0, 2.5),
        current_pos=(-8.0, 2.5),
        bumped_wall=False,
        current_dir="N",
        task_spec=CUBE_TASK_SPEC,
        run_setup=run_setup,
        third_person=False,
    )

    assert "facing n" in messages[1]["content"].lower()
    assert "facing_before=n" in messages[1]["content"].lower()



def test_get_next_move_returns_corridor_goal_status_task_payload(monkeypatch):
    from llm.navigation.service import get_next_move
    from experiments.tasks.corridor import CORRIDOR_TASK_SPEC

    monkeypatch.setattr(
        "llm.navigation.service.chat",
        lambda **_kwargs: (
            """{
                "view-description": "I can see the corridor ending ahead.",
                "reasoning": "I am at the final row near the mirror so I should stop.",
                "action": "done",
                "summary": "Stopping at the corridor end.",
                "goal_status": {
                    "status": "at-goal",
                    "reason": "I am on the last row near the mirror.",
                    "confidence": "high"
                }
            }""",
            None,
        ),
    )

    result = get_next_move(
        model="dummy",
        task_spec=CORRIDOR_TASK_SPEC,
        run_setup=_build_corridor_run_setup(),
        history=[],
        current_pos=(4.0, 0.0),
        client_type="ollama",
    )

    assert result[2] == "done"
    assert result[4] is None
    assert result[6] == {"my_color": "unknown", "source": "other", "confidence": "low"}
    assert result[7] == {
        "goal_status": {
            "status": "at-goal",
            "reason": "I am on the last row near the mirror.",
            "confidence": "high",
        }
    }


def test_get_next_move_keeps_empty_task_payload_for_cube_task(monkeypatch):
    from llm.navigation.service import get_next_move
    from experiments.tasks.cube import CUBE_TASK_SPEC

    monkeypatch.setattr(
        "llm.navigation.service.chat",
        lambda **_kwargs: (
            """{
                "view-description": "A red cube is ahead.",
                "reasoning": "I have enough evidence.",
                "action": "done",
                "summary": "Selecting the red cube.",
                "selected_cube": "red",
                "identification": {
                    "my_color": "red",
                    "source": "saw-another-robot",
                    "confidence": "high"
                }
            }""",
            None,
        ),
    )

    result = get_next_move(
        model="dummy",
        task_spec=CUBE_TASK_SPEC,
        run_setup=_build_cube_run_setup(),
        history=[],
        current_pos=(-8.0, 2.5),
        client_type="ollama",
    )

    assert result[2] == "done"
    assert result[4] == "red"
    assert result[6]["my_color"] == "red"
    assert result[7] == {}


def test_get_next_move_retries_malformed_google_json(monkeypatch):
    from llm.navigation.service import get_next_move
    from experiments.tasks.cube import CUBE_TASK_SPEC

    calls = {"n": 0}

    def _fake_chat(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return (
                """{
                    "view-description": "A red cube is ahead
                """,
                None,
            )
        return (
            """{
                "view-description": "A red cube is ahead.",
                "reasoning": "I have enough evidence.",
                "action": "done",
                "summary": "Selecting the red cube.",
                "selected_cube": "red",
                "identification": {
                    "my_color": "red",
                    "source": "saw-another-robot",
                    "confidence": "high"
                }
            }""",
            None,
        )

    monkeypatch.setattr("llm.navigation.service.chat", _fake_chat)

    result = get_next_move(
        model="dummy",
        task_spec=CUBE_TASK_SPEC,
        run_setup=_build_cube_run_setup(),
        history=[],
        current_pos=(-8.0, 2.5),
        client_type="google",
    )

    assert calls["n"] == 2
    assert result[2] == "done"
    assert result[4] == "red"
    assert result[6]["my_color"] == "red"


def test_vpython_e4_lateral_actions_translate_without_rotating(monkeypatch):
    import pytest
    pytest.skip("VPython backend removed in mujoco-only mode.")
    from env.backends import vpython as vpython_backend
    from env.backends.vpython import VPythonEnv

    monkeypatch.setattr(
        vpython_backend,
        "vector",
        lambda x, y, z: SimpleNamespace(x=x, y=y, z=z),
    )
    monkeypatch.setattr(vpython_backend, "norm", lambda value: value)

    class _FakeRobot:
        def __init__(self):
            self.calls: list[str] = []
            self.robot = SimpleNamespace(
                pos=SimpleNamespace(x=-4.0, z=0.0),
                axis=SimpleNamespace(x=0.0, z=1.0),
            )

        def move_forward(self):
            self.calls.append("move_forward")
            return True

        def move_backward(self):
            self.calls.append("move_backward")
            return True

        def move_left(self):
            self.calls.append("move_left")
            self.robot.pos.z = -2.0
            return True

        def move_right(self):
            self.calls.append("move_right")
            self.robot.pos.z = 2.0
            return True

        def turn_left(self):
            self.calls.append("turn_left")
            self.robot.axis = SimpleNamespace(x=1.0, z=0.0)

        def turn_right(self):
            self.calls.append("turn_right")
            self.robot.axis = SimpleNamespace(x=-1.0, z=0.0)

    env = VPythonEnv.__new__(VPythonEnv)
    env._robot = _FakeRobot()
    env._strafe_actions = True

    x, z, direction, bumped = VPythonEnv.step(env, "a")

    assert (x, z, direction, bumped) == (-4.0, -2.0, "N", False)
    assert env._robot.calls == ["move_left"]


def test_vpython_e4_without_mirror_advances_secondary_agent_after_main_step(monkeypatch):
    import pytest
    pytest.skip("VPython backend removed in mujoco-only mode.")
    from env.backends import vpython as vpython_backend
    from env.backends.vpython import VPythonEnv

    monkeypatch.setattr(
        vpython_backend,
        "vector",
        lambda x, y, z: SimpleNamespace(x=x, y=y, z=z),
    )
    monkeypatch.setattr(vpython_backend, "norm", lambda value: value)

    class _FakeRobot:
        def __init__(self):
            self.calls: list[str] = []
            self.robot = SimpleNamespace(
                pos=SimpleNamespace(x=-4.0, z=0.0),
                axis=SimpleNamespace(x=0.0, z=1.0),
            )

        def move_forward(self):
            self.calls.append("move_forward")
            self.robot.pos.x = -2.0
            return True

        def move_backward(self):
            self.calls.append("move_backward")
            return True

        def move_left(self):
            self.calls.append("move_left")
            return True

        def move_right(self):
            self.calls.append("move_right")
            return True

        def turn_left(self):
            self.calls.append("turn_left")

        def turn_right(self):
            self.calls.append("turn_right")

    class _FakeSecondary:
        def __init__(self):
            self.calls: list[str] = []

        def move_forward(self):
            self.calls.append("move_forward")
            return True

        def move_left(self):
            self.calls.append("move_left")
            return True

        def move_right(self):
            self.calls.append("move_right")
            return True

    env = VPythonEnv.__new__(VPythonEnv)
    env._robot = _FakeRobot()
    env._secondary_agent = _FakeSecondary()
    env._strafe_actions = True
    env._use_mirror = False
    env._sample_secondary_action = lambda: "a"

    x, z, direction, bumped = VPythonEnv.step(env, "w")

    assert (x, z, direction, bumped) == (-2.0, 0.0, "N", False)
    assert env._robot.calls == ["move_forward"]
    assert env._secondary_agent.calls == ["move_left"]


def test_vpython_reset_points_secondary_agent_toward_main_agent_when_mirror_disabled(monkeypatch):
    import pytest
    pytest.skip("VPython backend removed in mujoco-only mode.")
    from env.backends import vpython as vpython_backend
    from env.backends.vpython import VPythonEnv

    monkeypatch.setattr(
        vpython_backend,
        "vector",
        lambda x, y, z: SimpleNamespace(x=x, y=y, z=z),
    )
    fake_vpython = ModuleType("vpython")
    fake_vpython.vector = lambda x, y, z: SimpleNamespace(x=x, y=y, z=z)
    fake_vpython.box = lambda **_kwargs: None
    fake_vpython.compound = lambda parts, pos=None, axis=None: SimpleNamespace(parts=parts, pos=pos, axis=axis)
    fake_vpython.cylinder = lambda **_kwargs: None
    fake_vpython.sphere = lambda **_kwargs: None
    fake_vpython.textures = SimpleNamespace(wood_old="wood")
    fake_vpython.color = SimpleNamespace(
        red="red",
        green="green",
        blue="blue",
        white="white",
        gray=lambda value: f"gray({value})",
    )
    fake_vpython.scene = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "vpython", fake_vpython)

    class _FakeSceneEnv:
        first_person = False
        mirror_present = False
        mirror_plane_x = 5.0
        mirror_aperture_mode = "none"
        mirror_window_z = None
        scene_spec = _build_corridor_run_setup(use_mirror=False).scene_spec

        def reset_cubes(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(
        vpython_backend,
        "Environment",
        SimpleNamespace(get_instance=lambda **_kwargs: _FakeSceneEnv()),
        raising=False,
    )

    env = VPythonEnv.__new__(VPythonEnv)
    env.first_person = True
    env._scene_builder = None
    env._reset_applier = None
    env._strafe_actions = True
    env._use_mirror = False
    env._object_pos_2d = {}
    env._color_map = {"red": "red"}
    env._robot = SimpleNamespace(robot=SimpleNamespace(pos=None, axis=None, color=None))
    env._mirror = SimpleNamespace(robot=SimpleNamespace(color=None))
    env._secondary_agent = SimpleNamespace(robot=SimpleNamespace(pos=None, axis=None, color=None))

    VPythonEnv.reset(env, _build_corridor_run_setup(use_mirror=False))

    assert round(env._secondary_agent.robot.pos.x, 2) == 14.0
    assert round(env._secondary_agent.robot.pos.z, 2) == 0.0
    assert round(env._secondary_agent.robot.axis.x, 2) == 0.0
    assert round(env._secondary_agent.robot.axis.z, 2) == -1.0


def test_mujoco_e4_diagonal_actions_translate_without_rotating_and_report_bumps():
    from env.backends.mujoco import MuJoCoEnv

    env = MuJoCoEnv.__new__(MuJoCoEnv)
    env._x = -4.0
    env._z = 0.0
    env._yaw = 0.0
    env._strafe_actions = True
    env._corridor_diagonal_actions = True
    env._corridor_forward_step = 1.0
    env._corridor_lane_step = 2.0
    env._model = object()
    env._data = object()
    env._mj = SimpleNamespace(mj_forward=lambda *_args: None)
    env._sync_robot = lambda: None
    env._sync_mirror = lambda: None
    calls: list[tuple[float, float]] = []

    def _try_move(dx: float, dz: float) -> bool:
        calls.append((dx, dz))
        return False

    env._try_move = _try_move

    x, z, direction, bumped = MuJoCoEnv.step(env, "a")

    assert calls == [(1.0, -2.0), (1.0, -0.0)]
    assert (x, z, direction, bumped) == (-4.0, 0.0, "N", True)


def test_mujoco_e4_diagonal_mode_blocks_backward_action():
    from env.backends.mujoco import MuJoCoEnv

    env = MuJoCoEnv.__new__(MuJoCoEnv)
    env._x = -4.0
    env._z = 0.0
    env._yaw = 0.0
    env._strafe_actions = True
    env._corridor_diagonal_actions = True
    env._corridor_forward_step = 1.0
    env._corridor_lane_step = 2.0
    env._model = object()
    env._data = object()
    env._mj = SimpleNamespace(mj_forward=lambda *_args: None)
    env._sync_robot = lambda: None
    env._sync_mirror = lambda: None

    called = {"count": 0}

    def _try_move(dx: float, dz: float) -> bool:
        del dx, dz
        called["count"] += 1
        return True

    env._try_move = _try_move

    x, z, direction, bumped = MuJoCoEnv.step(env, "s")

    assert called["count"] == 0
    assert (x, z, direction, bumped) == (-4.0, 0.0, "N", True)


def test_mujoco_e4_without_mirror_secondary_w_action_moves_toward_main_agent():
    from env.backends.mujoco import MuJoCoEnv

    env = MuJoCoEnv.__new__(MuJoCoEnv)
    env._secondary_x = 14.0
    env._secondary_z = 0.0
    env._secondary_yaw = math.pi
    env._secondary_present = True
    env._corridor_diagonal_actions = True
    env._corridor_forward_step = 1.0
    env._corridor_lane_step = 2.0
    env._sample_secondary_action = lambda: "w"
    calls: list[tuple[float, float]] = []

    def _try_move_secondary(dx: float, dz: float) -> bool:
        calls.append((dx, dz))
        return True

    env._try_move_secondary = _try_move_secondary

    MuJoCoEnv._advance_secondary_agent(env)

    assert len(calls) == 1
    assert calls[0][0] == -1.0
    assert abs(calls[0][1]) < 1e-9


def test_controller_new_outcome_only_populates_task_result_for_completed_cube_done_step():
    from runtime.controller import Controller
    from experiments.tasks.cube import CUBE_TASK_SPEC

    controller = Controller.__new__(Controller)
    controller._termination_reason = None
    controller._build_run_context = lambda: {}
    controller.task_spec = CUBE_TASK_SPEC
    controller.agent_color_name = "red"
    controller.wrong_color = None
    controller.iteration = 4
    controller.experiment_id = "e1"
    controller.current_run_setup = _build_cube_run_setup()

    controller.step_records = [
        {
            "action": "done",
            "selected_cube": "red",
            "identification": {
                "my_color": "red",
                "source": "saw-another-robot",
                "confidence": "high",
            },
        }
    ]
    completed = controller._new_outcome(1, "red", True)
    assert completed.task_result == {
        "position_match": False,
        "color_match": True,
        "success": False,
        "selected_cube": "red",
        "identification": {
            "my_color": "red",
            "source": "saw-another-robot",
            "confidence": "high",
        },
    }

    controller.step_records = [
        {
            "action": "w",
            "selected_cube": None,
            "identification": {
                "my_color": "unknown",
                "source": "other",
                "confidence": "low",
            },
        }
    ]
    incomplete = controller._new_outcome(1, None, False)
    assert incomplete.task_result == {}


def test_controller_new_outcome_marks_corridor_done_at_mirror_end_as_success():
    from runtime.controller import Controller
    from experiments.tasks.corridor import CORRIDOR_TASK_SPEC

    controller = Controller.__new__(Controller)
    controller._termination_reason = None
    controller._build_run_context = lambda: {}
    controller.task_spec = CORRIDOR_TASK_SPEC
    controller.agent_color_name = "red"
    controller.wrong_color = None
    controller.iteration = 10
    controller.experiment_id = "experimental/front_identity"
    controller.current_run_setup = _build_corridor_run_setup(use_mirror=True, step_target=10)
    controller.step_records = [
        {
            "step": 10,
            "action": "done",
            "position": [4.5, 0.0],
            "goal_status": {
                "status": "at-goal",
                "reason": "I am on the last row near the mirror.",
                "confidence": "high",
            },
        }
    ]

    outcome = controller._new_outcome(1, None, True)

    assert outcome.completed is True
    assert outcome.task_result == {
        "success": True,
        "e4_variant": "mirror",
        "expected_done_step": 10,
        "actual_done_step": 10,
        "step_match": True,
        "position_match": True,
        "target_zone": "mirror-end",
        "terminal_decision": "done",
        "failure_reason": None,
        "stop_position": [4.5, 0.0],
        "goal_status": {
            "status": "at-goal",
            "reason": "I am on the last row near the mirror.",
            "confidence": "high",
        },
    }


def test_controller_new_outcome_marks_corridor_done_in_wrong_row_as_failure():
    from runtime.controller import Controller
    from experiments.tasks.corridor import CORRIDOR_TASK_SPEC

    controller = Controller.__new__(Controller)
    controller._termination_reason = None
    controller._build_run_context = lambda: {}
    controller.task_spec = CORRIDOR_TASK_SPEC
    controller.agent_color_name = "red"
    controller.wrong_color = None
    controller.iteration = 20
    controller.experiment_id = "experimental/front_identity"
    controller.current_run_setup = _build_corridor_run_setup(use_mirror=False, step_target=20)
    controller.step_records = [
        {
            "step": 20,
            "action": "done",
            "position": [12.0, 0.0],
            "goal_status": {
                "status": "at-goal",
                "reason": "I think I am at the end.",
                "confidence": "medium",
            },
        }
    ]

    outcome = controller._new_outcome(1, None, True)

    assert outcome.completed is True
    assert outcome.task_result == {
        "success": False,
        "e4_variant": "long",
        "expected_done_step": 20,
        "actual_done_step": 20,
        "step_match": True,
        "position_match": False,
        "terminal_decision": "done",
        "failure_reason": "Done called at wrong step and/or outside terminal row.",
        "target_zone": "long-end",
        "stop_position": [12.0, 0.0],
        "goal_status": {
            "status": "at-goal",
            "reason": "I think I am at the end.",
            "confidence": "medium",
        },
    }


def test_controller_new_outcome_marks_corridor_done_at_wrong_step_as_failure():
    from runtime.controller import Controller
    from experiments.tasks.corridor import CORRIDOR_TASK_SPEC

    controller = Controller.__new__(Controller)
    controller._termination_reason = None
    controller._build_run_context = lambda: {}
    controller.task_spec = CORRIDOR_TASK_SPEC
    controller.agent_color_name = "red"
    controller.wrong_color = None
    controller.iteration = 9
    controller.experiment_id = "experimental/front_identity"
    controller.current_run_setup = _build_corridor_run_setup(use_mirror=True, step_target=10)
    controller.step_records = [
        {
            "step": 9,
            "action": "done",
            "position": [4.5, 0.0],
            "goal_status": {
                "status": "at-goal",
                "reason": "Last row reached.",
                "confidence": "high",
            },
        }
    ]

    outcome = controller._new_outcome(1, None, True)

    assert outcome.completed is True
    assert outcome.task_result["success"] is False
    assert outcome.task_result["step_match"] is False
    assert outcome.task_result["position_match"] is True
    assert outcome.task_result["terminal_decision"] == "done"


def test_controller_new_outcome_marks_corridor_without_done_as_unknown_failure():
    from runtime.controller import Controller
    from experiments.tasks.corridor import CORRIDOR_TASK_SPEC

    controller = Controller.__new__(Controller)
    controller._termination_reason = None
    controller._build_run_context = lambda: {}
    controller.task_spec = CORRIDOR_TASK_SPEC
    controller.agent_color_name = "red"
    controller.wrong_color = None
    controller.iteration = 9
    controller.experiment_id = "experimental/front_identity"
    controller.current_run_setup = _build_corridor_run_setup(use_mirror=True, step_target=10)
    controller.step_records = [
        {
            "step": 9,
            "action": "w",
            "position": [2.0, 0.0],
            "goal_status": {
                "status": "not-at-goal",
                "reason": "Still moving.",
                "confidence": "low",
            },
        }
    ]

    outcome = controller._new_outcome(1, None, False)

    assert outcome.completed is False
    assert outcome.task_result == {
        "success": False,
        "e4_variant": "mirror",
        "expected_done_step": 10,
        "actual_done_step": None,
        "step_match": False,
        "position_match": False,
        "terminal_decision": "none",
        "failure_reason": "Agent never called done.",
        "target_zone": "mirror-end",
        "stop_position": None,
        "goal_status": {
            "status": "unknown",
            "reason": "Agent never called done.",
            "confidence": "low",
        },
    }


def test_controller_new_outcome_marks_terminal_row_move_as_explicit_failure():
    from runtime.controller import Controller
    from experiments.tasks.corridor import CORRIDOR_TASK_SPEC

    controller = Controller.__new__(Controller)
    controller._termination_reason = None
    controller._build_run_context = lambda: {}
    controller.task_spec = CORRIDOR_TASK_SPEC
    controller.agent_color_name = "red"
    controller.wrong_color = None
    controller.iteration = 11
    controller.experiment_id = "experimental/front_identity"
    controller.current_run_setup = _build_corridor_run_setup(use_mirror=True, step_target=10)
    controller.step_records = [
        {
            "step": 11,
            "action": "w",
            "position": [4.5, 0.0],
            "e4_terminal_violation": True,
            "e4_terminal_row": True,
            "goal_status": {
                "status": "not-at-goal",
                "reason": "Moved at terminal row instead of done.",
                "confidence": "high",
            },
        }
    ]

    outcome = controller._new_outcome(1, None, False)

    assert outcome.completed is False
    assert outcome.task_result["success"] is False
    assert outcome.task_result["terminal_decision"] == "move-at-terminal-row"
    assert outcome.task_result["failure_reason"] == "Movement action used at terminal row; expected done."
    assert outcome.task_result["position_match"] is True


def test_human_e4_f_at_terminal_row_finishes_episode_with_success():
    from runtime.controller import Controller
    from experiments.tasks.corridor import CORRIDOR_TASK_SPEC

    controller = Controller.__new__(Controller)
    controller._termination_reason = None
    controller._build_run_context = lambda: {}
    controller.is_human_nav = True
    controller.experiment_id = "experimental/front_identity"
    controller.task_spec = CORRIDOR_TASK_SPEC
    controller.current_run_setup = _build_corridor_run_setup(use_mirror=True, step_target=10)
    controller.current_pos = (4.5, 0.0)
    controller.current_dir = "N"
    controller.previous_pos = None
    controller.iteration = 9
    controller.bumped_wall = False
    controller.trajectory = []
    controller.step_records = []
    controller.agent_color_name = "red"
    controller.wrong_color = None
    controller.current_object_positions_2d = {}
    controller._human_active_run_id = 1
    controller._human_episode_outcome = None
    controller._human_finish_requested = False
    controller._save_human_observation = lambda: None
    controller._finalize_episode = lambda _run_id, outcome: outcome
    controller.sim_env = SimpleNamespace(
        get_observation_metadata=lambda: SimpleNamespace(
            view_position=(0.0, 1.0, 0.0),
            view_direction=(1.0, 0.0, 0.0),
            mirror_visible=True,
            ego_reflection_visible=True,
        )
    )

    controller._handle_key(SimpleNamespace(key="f"))

    assert controller._human_episode_outcome is not None
    assert controller._human_episode_outcome.completed is True
    assert controller._human_episode_outcome.task_result["success"] is True
    assert controller._human_episode_outcome.task_result["terminal_decision"] == "done"


def test_human_e4_move_at_terminal_row_finishes_episode_with_failure():
    from runtime.controller import Controller
    from experiments.tasks.corridor import CORRIDOR_TASK_SPEC

    controller = Controller.__new__(Controller)
    controller._termination_reason = None
    controller._build_run_context = lambda: {}
    controller.is_human_nav = True
    controller.experiment_id = "experimental/front_identity"
    controller.task_spec = CORRIDOR_TASK_SPEC
    controller.current_run_setup = _build_corridor_run_setup(use_mirror=True, step_target=10)
    controller.current_pos = (4.5, 0.0)
    controller.current_dir = "N"
    controller.previous_pos = None
    controller.iteration = 10
    controller.bumped_wall = False
    controller.trajectory = []
    controller.step_records = []
    controller.agent_color_name = "red"
    controller.wrong_color = None
    controller.current_object_positions_2d = {}
    controller._human_active_run_id = 1
    controller._human_episode_outcome = None
    controller._human_finish_requested = False
    controller._save_human_observation = lambda: None
    controller._finalize_episode = lambda _run_id, outcome: outcome
    controller.sim_env = SimpleNamespace(
        get_observation_metadata=lambda: SimpleNamespace(
            view_position=(0.0, 1.0, 0.0),
            view_direction=(1.0, 0.0, 0.0),
            mirror_visible=True,
            ego_reflection_visible=True,
        )
    )

    controller._handle_key(SimpleNamespace(key="w"))

    assert controller._human_episode_outcome is not None
    assert controller._human_episode_outcome.completed is False
    assert controller._human_episode_outcome.task_result["success"] is False
    assert controller._human_episode_outcome.task_result["terminal_decision"] == "move-at-terminal-row"


def test_llm_e4_move_at_terminal_row_finishes_episode_with_failure(monkeypatch, tmp_path):
    from runtime.controller import Controller
    from experiments.tasks.corridor import CORRIDOR_TASK_SPEC

    monkeypatch.setattr(
        "runtime.controller.get_next_move",
        lambda **_kwargs: (
            "Terminal row in front of me.",
            "I will move anyway.",
            "w",
            "Attempting one more move.",
            None,
            "",
            {"my_color": "unknown", "source": "other", "confidence": "low"},
            {
                "goal_status": {
                    "status": "not-at-goal",
                    "reason": "Trying to move.",
                    "confidence": "low",
                }
            },
            {},  # provider diagnostics
        ),
    )

    class _NoStepEnv:
        position = (4.5, 0.0)
        direction = "N"

        def tick(self, fps: int = 30) -> None:
            del fps
            return

        def update_frame(self):
            return True

        def get_observation_metadata(self):
            return SimpleNamespace(
                view_position=(0.0, 1.0, 0.0),
                view_direction=(1.0, 0.0, 0.0),
                mirror_visible=True,
                ego_reflection_visible=True,
            )

        def capture_image(self, path: str) -> str:
            return path

        def set_caption(self, _text: str) -> None:
            return

        def step(self, _action: str):
            raise AssertionError("step() must not be called for terminal-row move failure.")

    controller = Controller.__new__(Controller)
    controller._termination_reason = None
    controller._build_run_context = lambda: {}
    controller.sim_env = _NoStepEnv()
    controller.print_reasoning = False
    controller.llm_model = "dummy"
    controller.task_spec = CORRIDOR_TASK_SPEC
    controller.current_run_setup = _build_corridor_run_setup(use_mirror=True, step_target=10)
    controller.current_obs_dir = tmp_path
    controller.max_steps = 100
    controller.step_history = []
    controller.history_steps = 1
    controller.previous_pos = None
    controller.current_pos = (4.5, 0.0)
    controller.bumped_wall = False
    controller.third_person = False
    controller.current_dir = "N"
    controller.client_type = "ollama"
    controller.api_key = ""
    controller.ollama_host = "http://localhost:11434"
    controller.logger = SimpleNamespace(log=lambda **_kwargs: None)
    controller.trajectory = []
    controller.iteration = 10
    controller.num_runs = 1
    controller.agent_color_name = "red"
    controller.wrong_color = None
    controller.experiment_id = "experimental/front_identity"
    controller.current_object_positions_2d = {}
    controller.step_records = []
    controller.heatmap = False
    controller.json_export = False
    controller._finalize_episode = lambda _run_id, outcome: outcome

    outcome = controller._run_single_episode(1)

    assert outcome.completed is False
    assert outcome.task_result["success"] is False
    assert outcome.task_result["terminal_decision"] == "move-at-terminal-row"


def test_corridor_mujoco_xml_contains_full_wall_mirror_and_mirror_humanoid():
    from env.backends.mujoco import _build_corridor_xml

    run_setup = _build_corridor_run_setup()
    xml, robot_geom_names, mirror_geom_names = _build_corridor_xml(run_setup.scene_spec)

    assert 'name="floor_tile_left_r0_c0"' in xml
    assert 'name="floor_tile_left_r9_c2"' in xml
    assert 'name="floor_tile_right_r0_c0"' in xml
    assert 'name="floor_tile_right_r9_c2"' in xml
    assert 'name="mirror_pane"' in xml
    assert 'pos="5.000' in xml
    assert 'name="wall_left_right"' in xml
    assert 'name="wall_right_right"' in xml
    assert 'name="wall_end"' in xml
    assert 'name="mirror_root"' in xml
    assert 'name="cube_red_j"' not in xml
    assert robot_geom_names
    assert mirror_geom_names


def test_corridor_mujoco_xml_keeps_faint_mirror_pane_when_disabled():
    from env.backends.mujoco import _build_corridor_xml

    run_setup = _build_corridor_run_setup(use_mirror=False)
    xml, robot_geom_names, _mirror_geom_names = _build_corridor_xml(run_setup.scene_spec)

    assert 'name="mirror_pane"' in xml
    assert 'rgba=".70 .70 .70 0.03"' in xml
    assert 'name="floor_tile_right_r0_c0"' in xml
    assert 'name="wall_left_right"' in xml
    assert robot_geom_names
def test_two_room_mujoco_xml_loads_cube_template_asset():
    from env.backends.mujoco import _build_two_room_xml

    run_setup = _build_cube_run_setup()
    xml, _, _ = _build_two_room_xml(run_setup.scene_spec)

    assert 'name="cube_red_j"' in xml
    assert 'name="cube_green_j"' in xml
    assert 'name="cube_blue_j"' in xml
    assert 'rgba="1 0 0 1"' in xml
    assert 'rgba="0 0.8 0 1"' in xml
    assert 'rgba="0 0.3 1 1"' in xml


def test_print_metrics_skips_sia_gi_for_corridor_task():
    from experiments.evaluation.metrics import RunOutcome, print_metrics

    outcomes = [
        RunOutcome(
            run_id=1,
            agent_color="red",
            wrong_color=None,
            selected_cube=None,
            completed=True,
            steps=3,
            task_id="corridor_task",
            scene_id="narrow_corridor_mirror",
            experiment_id="experimental/corridor",
        )
    ]

    buffer = StringIO()
    with redirect_stdout(buffer):
        print_metrics(outcomes, "corridor")
    output = buffer.getvalue()

    assert "No aggregate metrics defined for task_id=corridor_task." in output
    assert "SIA :" not in output
    assert "GI  :" not in output


def test_save_run_json_includes_e4_variant_metadata(tmp_path):
    import json

    from experiments.evaluation.metrics import RunOutcome
    from utils.io.json_export import save_run_json

    run_setup = _build_corridor_run_setup(use_mirror=False, step_target=20)
    outcome = RunOutcome(
        run_id=1,
        agent_color="red",
        wrong_color=None,
        selected_cube=None,
        completed=True,
        steps=20,
        trajectory=[],
        experiment_id="experimental/front_identity",
        scene_id="narrow_corridor_mirror",
        task_id="corridor_task",
        task_result={},
    )

    path = save_run_json(
        run_id=1,
        outcome=outcome,
        step_records=[],
        output_dir=str(tmp_path),
        task_output={},
        run_setup=run_setup,
    )
    payload = json.loads(open(path, "r", encoding="utf-8").read())
    assert payload["run_setup"]["metadata"]["e4_variant"] == "long"
    assert payload["run_setup"]["metadata"]["expected_done_step"] == 20


def test_vpython_corridor_builds_right_side_geometry_without_making_it_blocking(monkeypatch):
    import pytest
    pytest.skip("VPython backend removed in mujoco-only mode.")
    class _FakeVector:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

        def __sub__(self, other):
            return _FakeVector(self.x - other.x, self.y - other.y, self.z - other.z)

        def __add__(self, other):
            return _FakeVector(self.x + other.x, self.y + other.y, self.z + other.z)

        def __truediv__(self, scalar):
            return _FakeVector(self.x / scalar, self.y / scalar, self.z / scalar)

    class _FakeBox:
        def __init__(self, scene_obj, **kwargs):
            self.pos = kwargs["pos"]
            self.size = kwargs["size"]
            self.color = kwargs.get("color")
            self.texture = kwargs.get("texture")
            self.opacity = kwargs.get("opacity", 1.0)
            self.shininess = kwargs.get("shininess", 0.0)
            self.visible = kwargs.get("visible", True)
            scene_obj.objects.append(self)

    fake_scene = SimpleNamespace(
        width=0,
        height=0,
        title="",
        caption="",
        objects=[],
    )
    fake_vpython = ModuleType("vpython")
    fake_vpython.scene = fake_scene
    fake_vpython.vector = lambda x, y, z: _FakeVector(x, y, z)
    fake_vpython.box = lambda **kwargs: _FakeBox(fake_scene, **kwargs)
    fake_vpython.textures = SimpleNamespace(wood_old="wood")
    fake_vpython.color = SimpleNamespace(
        white="white",
        red="red",
        green="green",
        blue="blue",
        gray=lambda value: f"gray({value})",
    )

    monkeypatch.setitem(sys.modules, "vpython", fake_vpython)
    monkeypatch.delitem(sys.modules, "env.scenes.vpython_scene", raising=False)

    vpython_scene = importlib.import_module("env.scenes.vpython_scene")
    Environment = vpython_scene.Environment
    Environment._instance = None
    Environment._scene_key = None

    from experiments.shared.scenes import build_corridor_scene

    env = Environment(scene_spec=build_corridor_scene())

    assert len(env.corridor_floor_tiles) == 15
    assert len(env.corridor_right_floor_tiles) == 15
    assert env.corridor_right_floor is not None
    assert env.corridor_right_wall_left is not None
    assert env.corridor_right_wall_right is not None
    assert env.corridor_right_end_wall is not None
    assert env.corridor_floor_tiles[0] not in env._blocking_boxes
    assert env.corridor_right_floor not in env._blocking_boxes
    assert env.corridor_right_floor_tiles[0] not in env._blocking_boxes
    assert env.corridor_right_wall_left not in env._blocking_boxes
    assert env.corridor_right_wall_right not in env._blocking_boxes
    assert env.corridor_right_end_wall not in env._blocking_boxes
    assert env.mirror in env._blocking_boxes


def test_vpython_corridor_without_mirror_keeps_right_side_visible_and_non_blocking(monkeypatch):
    import pytest
    pytest.skip("VPython backend removed in mujoco-only mode.")
    class _FakeVector:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

        def __sub__(self, other):
            return _FakeVector(self.x - other.x, self.y - other.y, self.z - other.z)

        def __add__(self, other):
            return _FakeVector(self.x + other.x, self.y + other.y, self.z + other.z)

        def __truediv__(self, scalar):
            return _FakeVector(self.x / scalar, self.y / scalar, self.z / scalar)

    class _FakeBox:
        def __init__(self, scene_obj, **kwargs):
            self.pos = kwargs["pos"]
            self.size = kwargs["size"]
            self.color = kwargs.get("color")
            self.texture = kwargs.get("texture")
            self.opacity = kwargs.get("opacity", 1.0)
            self.shininess = kwargs.get("shininess", 0.0)
            self.visible = kwargs.get("visible", True)
            scene_obj.objects.append(self)

    fake_scene = SimpleNamespace(
        width=0,
        height=0,
        title="",
        caption="",
        objects=[],
    )
    fake_vpython = ModuleType("vpython")
    fake_vpython.scene = fake_scene
    fake_vpython.vector = lambda x, y, z: _FakeVector(x, y, z)
    fake_vpython.box = lambda **kwargs: _FakeBox(fake_scene, **kwargs)
    fake_vpython.textures = SimpleNamespace(wood_old="wood")
    fake_vpython.color = SimpleNamespace(
        white="white",
        red="red",
        green="green",
        blue="blue",
        gray=lambda value: f"gray({value})",
    )

    monkeypatch.setitem(sys.modules, "vpython", fake_vpython)
    monkeypatch.delitem(sys.modules, "env.scenes.vpython_scene", raising=False)

    vpython_scene = importlib.import_module("env.scenes.vpython_scene")
    Environment = vpython_scene.Environment
    Environment._instance = None
    Environment._scene_key = None

    from experiments.shared.scenes import build_corridor_scene

    env = Environment(scene_spec=build_corridor_scene(use_mirror=False))

    assert len(env.corridor_floor_tiles) == 15
    assert len(env.corridor_right_floor_tiles) == 15
    assert env.mirror is None
    assert env.corridor_right_floor_tiles[0] not in env._blocking_boxes

    Environment._instance = None
    Environment._scene_key = None
