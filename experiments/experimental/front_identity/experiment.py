from __future__ import annotations

import math as _math
import os as _os
import random as _random

from experiments.shared.scenes import (
    build_corridor_run_setup as _build_corridor_run_setup,
    build_corridor_scene as _build_corridor_scene,
)
from experiments.shared.types import (
    BackendOverrides as _BackendOverrides,
    ExperimentRuntimeOptions as _ExperimentRuntimeOptions,
    ExperimentSpec as _ExperimentSpec,
    RunSetup as _RunSetup,
    SceneSpec as _SceneSpec,
)

_VARIANT_SELF = "self"
_VARIANT_OTHER = "other"
_VARIANT_CHOICES = (_VARIANT_SELF, _VARIANT_OTHER)


def _sample_variant() -> str:
    forced = str(_os.getenv("E4_FRONT_IDENTITY_VARIANT", "")).strip().lower()
    if forced in _VARIANT_CHOICES:
        return forced
    return _random.choice(_VARIANT_CHOICES)


def _build_scene_template(options: _ExperimentRuntimeOptions) -> _SceneSpec:
    del options
    return _build_corridor_scene(use_mirror=True)


def _secondary_initial_pose(scene_spec: _SceneSpec) -> tuple[float, float, float]:
    mirror_plane_x = float(scene_spec.mirror_plane_x or (scene_spec.length / 2))
    spawn_x, spawn_z = scene_spec.spawn_pos
    secondary_x = round(2 * mirror_plane_x - float(spawn_x), 2)
    secondary_z = round(float(spawn_z), 2)
    secondary_yaw = float(scene_spec.spawn_yaw) + _math.pi
    return secondary_x, secondary_z, secondary_yaw


def _build_run_setup(
    run_id: int,
    _options: _ExperimentRuntimeOptions,
    _scene_template: _SceneSpec,
) -> _RunSetup:
    del run_id, _options
    scene_spec = _build_corridor_scene(use_mirror=True)
    run_setup = _build_corridor_run_setup(scene_spec)
    variant = _sample_variant()
    secondary_x, secondary_z, secondary_yaw = _secondary_initial_pose(scene_spec)
    run_setup.metadata.update(
        {
            "front_identity_variant": variant,
            "expected_answer": "yes" if variant == _VARIANT_SELF else "no",
            "secondary_initial_pos": [secondary_x, secondary_z],
            "secondary_initial_yaw": secondary_yaw,
        }
    )
    return run_setup


def _patch_sync_for_other_variant(env) -> None:
    if not hasattr(env, "_e4_default_sync_mirror"):
        env._e4_default_sync_mirror = env._sync_mirror
        env._e4_default_mirror_ego_geom_ids = list(getattr(env, "_mirror_ego_geom_ids", []))
        env._e4_default_try_move_secondary = env._try_move_secondary

    def _sync_secondary_agent() -> None:
        if not env._secondary_present:
            return
        env._set_body_pose(
            env._mirror_jnt_addr,
            env._secondary_x,
            env._spawn_y,
            env._secondary_z,
            env._secondary_yaw,
        )

    def _try_move_secondary_mirrored(dx: float, dz: float) -> bool:
        distance = _math.hypot(dx, dz)
        steps = max(1, int(distance / 0.1))
        origin_x, origin_z = env._secondary_x, env._secondary_z

        mirror_plane_x = float(getattr(env, "_mirror_plane_x", 0.0) or 0.0)
        min_x = mirror_plane_x + max(float(getattr(env, "_wall_thickness", 0.2)) / 2.0, 0.05)
        max_x = mirror_plane_x + float(getattr(env, "_scene_length", 10.0))
        min_z = -float(getattr(env, "_scene_width", 6.0)) / 2.0
        max_z = float(getattr(env, "_scene_width", 6.0)) / 2.0

        for i in range(1, steps + 1):
            nx = origin_x + dx * (i / steps)
            nz = origin_z + dz * (i / steps)
            if nx < min_x or nx > max_x or nz < min_z or nz > max_z:
                return False

        env._secondary_x = round(origin_x + dx, 2)
        env._secondary_z = round(origin_z + dz, 2)
        return True

    env._sync_mirror = _sync_secondary_agent
    env._try_move_secondary = _try_move_secondary_mirrored
    env._secondary_active = True
    env._mirror_ego_geom_ids = []


def _restore_sync_for_self_variant(env) -> None:
    default_sync = getattr(env, "_e4_default_sync_mirror", None)
    if default_sync is not None:
        env._sync_mirror = default_sync
    env._secondary_active = False
    default_geom_ids = getattr(env, "_e4_default_mirror_ego_geom_ids", None)
    if default_geom_ids is not None:
        env._mirror_ego_geom_ids = list(default_geom_ids)
    default_try_move_secondary = getattr(env, "_e4_default_try_move_secondary", None)
    if default_try_move_secondary is not None:
        env._try_move_secondary = default_try_move_secondary


def _apply_mujoco_reset(env, run_setup: _RunSetup) -> None:
    variant = str(run_setup.metadata.get("front_identity_variant", _VARIANT_SELF)).strip().lower()
    secondary_x, secondary_z, secondary_yaw = _secondary_initial_pose(run_setup.scene_spec)
    env._secondary_x = secondary_x
    env._secondary_z = secondary_z
    env._secondary_yaw = secondary_yaw

    if variant == _VARIANT_OTHER:
        _patch_sync_for_other_variant(env)
    else:
        _restore_sync_for_self_variant(env)

    env._sync_mirror()


def _resolve_backend_overrides(
    runtime_options: _ExperimentRuntimeOptions,
    scene_template: _SceneSpec,
) -> _BackendOverrides:
    del runtime_options, scene_template
    return _BackendOverrides(
        mujoco_reset_applier=_apply_mujoco_reset,
    )


EXPERIMENT = _ExperimentSpec(
    experiment_id="experimental/front_identity",
    scene_id=_build_corridor_scene(use_mirror=True).scene_id,
    task_id="front_identity_task",
    num_runs_default=10,
    scene_template_factory=_build_scene_template,
    run_setup_factory=_build_run_setup,
    backend_overrides_factory=_resolve_backend_overrides,
)
