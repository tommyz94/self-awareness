import os
import re
import time
import math
import json
from pathlib import Path
from typing import Any, List, Optional

from llm import get_next_move
from llm.dispatch import canonicalize_client_type
from providers.ollama import list_models
from env.base import CUBE_SLOTS, BaseEnv
from experiments.evaluation.metrics import (
    RunOutcome,
    extract_mirror_exposure_metrics,
    print_metrics,
)
from experiments.specs.builtin_helpers import build_cube_objects
from experiments.naming import resolve_experiment_selection
from experiments.specs import (
    BackendOverrides,
    ExperimentRuntimeOptions,
    RunSetup,
    SceneSpec,
    get_experiment,
    scene_geometry_signature,
)
from runtime.human_finish_dialog import prompt_human_cube_selection
from utils.colors import COLOR_POOL as _COLOR_POOL
from utils.io.json_export import save_experiment_json, save_run_json
from utils.io.logger import Logger

_DEFAULT_SINGLE_RUN_COLOR = "red"
_VALID_CUBE_SELECTIONS = set(_COLOR_POOL)
_VALID_CLIENT_TYPES = {"ollama", "openai", "openrouter", "google", "anthropic", "claude"}
_CANONICAL_CLIENT_TYPES = {"ollama", "openai", "openrouter", "google", "anthropic"}
_EXPERIMENT_SCOPED_OUTPUT_ROOTS = {"e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "corridor"}
_AUTO_NAV_NAME = "auto"
_E6_AUTOPILOT_SLEEP_SEC = 0.5
_E6_AUTOPILOT_TARGET_REACHED_EPS = 1.0
_DATASET_DIR_NAME = "Mirror_Video_Dataset"
_AUTO_NAV_EXPERIMENTS = {"e6", "e7"}


def _create_env(
    env_type: str,
    scene_spec: SceneSpec,
    backend_overrides: Optional[BackendOverrides],
    third_person: bool,
    camera_height: float,
    body_style: str,
    mujoco_headless: bool = True,
    mujoco_render_width: int = 640,
    mujoco_render_height: int = 480,
) -> BaseEnv:
    if env_type != "mujoco":
        raise ValueError(
            f"Unsupported env={env_type!r}. This repository is configured for MuJoCo only."
        )

    from env.backends.mujoco import MuJoCoEnv
    return MuJoCoEnv(
        scene_spec=scene_spec,
        backend_overrides=backend_overrides,
        third_person=third_person,
        camera_height=camera_height,
        body_style=body_style,
        headless=mujoco_headless,
        render_width=mujoco_render_width,
        render_height=mujoco_render_height,
    )


def _heatmap_supported(
    *,
    run_setup: Optional[RunSetup],
    task_spec: Optional[object],
) -> bool:
    if run_setup is None:
        return False

    if run_setup.scene_spec.layout_kind != "two_room":
        return False

    if task_spec is None:
        return True

    capabilities = getattr(task_spec, "capabilities", None)
    if capabilities is not None:
        return bool(getattr(capabilities, "supports_heatmap", False))

    return getattr(task_spec, "task_id", None) == "cube_task"


def _save_heatmap_if_supported(
    *,
    run_id: int,
    trajectory: list,
    agent_color: str,
    object_positions_2d: dict,
    output_dir: str,
    run_setup: Optional[RunSetup],
    task_spec: Optional[object],
) -> Optional[str]:
    if not _heatmap_supported(run_setup=run_setup, task_spec=task_spec):
        task_id = getattr(task_spec, "task_id", None) or "free_nav"
        scene_id = run_setup.scene_spec.scene_id if run_setup is not None else "unknown"
        print(
            f"Skipping heatmap for task_id={task_id}, scene_id={scene_id}: "
            "unsupported layout."
        )
        return None

    from utils.viz.heatmap import save_trajectory_heatmap

    return save_trajectory_heatmap(
        run_id,
        trajectory,
        agent_color,
        object_positions_2d,
        output_dir,
    )


def _slugify_path_token(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip().lower()).strip("_")


def _experiment_variant_tag(
    experiment_id: Optional[str],
    *,
    body_style: str,
    self_perception_mode: str,
) -> Optional[str]:
    if experiment_id == "e0":
        return "__".join(
            token
            for token in (
                _slugify_path_token(body_style or "default"),
                _slugify_path_token(self_perception_mode or "color"),
            )
            if token
        )
    if str(body_style or "default").strip().lower() != "default":
        token = _slugify_path_token(body_style)
        return token or None
    return None


def _should_display_agent_color(
    *,
    task_spec: Optional[object],
    run_setup: Optional[RunSetup],
) -> bool:
    task_id = getattr(task_spec, "task_id", None)
    if task_id == "cube_task":
        return True
    if task_id == "self_perception_task":
        mode = ""
        if run_setup is not None:
            mode = str(run_setup.metadata.get("self_perception_mode", "color")).strip().lower()
        return mode == "color"
    return False


class Controller:
    def __init__(
        self,
        nav_mode: str = "human",
        client_type: str = "ollama",
        print_reasoning: bool = False,
        num_distractors: int = 0,
        max_steps: int = 20,
        third_person: bool = False,
        camera_height: float = 1.0,
        use_mirror: bool = True,
        random_wall_color: bool = False,
        distractor_color_mode: str = "allow_duplicates",
        e7_random_cube_heights: bool = False,
        body_style: str = "default",
        self_perception_mode: str = "color",
        experiment_id: Optional[str] = None,
        num_runs: int = 10,
        history_steps: int = 1,
        ollama_host: str = "http://localhost:11434",
        openai_api_key: str = "",
        openrouter_api_key: str = "",
        openrouter_base_url: str = "https://openrouter.ai/api/v1",
        openrouter_http_referer: str = "",
        openrouter_app_title: str = "",
        openrouter_require_parameters: bool = True,
        openrouter_min_request_interval_sec: float = 0.0,
        google_api_key: str = "",
        anthropic_api_key: str = "",
        output_dir: str = "outputs",
        log_file: str = "llm_history.txt",
        json_export: bool = True,
        heatmap: bool = True,
        save_obs: bool = False,
        save_dataset: bool = False,
        env_type: str = "mujoco",
        mujoco_headless: bool = False,
        mujoco_render_width: int = 640,
        mujoco_render_height: int = 480,
        paper_experiment_id: Optional[str] = None,
    ):
        experiment_id, paper_experiment_id = resolve_experiment_selection(
            experiment_id, paper_experiment_id
        )
        canonical_client_type = canonicalize_client_type(client_type)
        if canonical_client_type not in _CANONICAL_CLIENT_TYPES:
            raise ValueError(
                f"Unknown client={client_type!r}. "
                f"Valid options: {sorted(_VALID_CLIENT_TYPES)}"
            )

        self.print_reasoning = print_reasoning
        self.history_steps = history_steps
        self.nav_mode = nav_mode.strip()
        self.is_human_nav = self.nav_mode.lower() == "human"
        self.is_auto_nav = self.nav_mode.lower() == _AUTO_NAV_NAME
        self.llm_model = None if (self.is_human_nav or self.is_auto_nav) else self.nav_mode
        self.env_type = env_type
        self.mujoco_headless = bool(mujoco_headless)
        self.client_type = canonical_client_type
        self.ollama_host = ollama_host
        if self.client_type == "openai":
            self.api_key = openai_api_key
        elif self.client_type == "openrouter":
            self.api_key = openrouter_api_key
        elif self.client_type == "google":
            self.api_key = google_api_key
        elif self.client_type == "anthropic":
            self.api_key = anthropic_api_key
        else:
            self.api_key = ""
        self.provider_base_url = (
            openrouter_base_url if self.client_type == "openrouter" else None
        )
        self.provider_http_referer = (
            openrouter_http_referer if self.client_type == "openrouter" else None
        )
        self.provider_app_title = (
            openrouter_app_title if self.client_type == "openrouter" else None
        )
        self.provider_require_parameters = (
            bool(openrouter_require_parameters)
            if self.client_type == "openrouter"
            else None
        )
        self.provider_min_request_interval_sec = (
            max(0.0, float(openrouter_min_request_interval_sec))
            if self.client_type == "openrouter"
            else None
        )
        self.third_person = third_person
        if not 0.0 <= camera_height <= 1.0:
            raise ValueError(
                f"camera_height must be in [0.0, 1.0], got {camera_height}."
            )
        self.camera_height = camera_height
        self.body_style = str(body_style or "default")
        self.self_perception_mode = str(self_perception_mode or "color")
        self.experiment_id = experiment_id
        self.paper_experiment_id = paper_experiment_id
        self.experiment_spec = get_experiment(experiment_id) if experiment_id else None
        self.experiment_runtime_options = ExperimentRuntimeOptions(
            distractors=num_distractors,
            camera_height=camera_height,
            use_mirror=use_mirror,
            random_wall_color=random_wall_color,
            distractor_color_mode=distractor_color_mode,
            e7_random_cube_heights=e7_random_cube_heights,
            body_style=self.body_style,
            self_perception_mode=self.self_perception_mode,
        )
        self.task_spec = self.experiment_spec.resolve_task() if self.experiment_spec else None
        self.scene_template = (
            self.experiment_spec.resolve_scene_template(self.experiment_runtime_options)
            if self.experiment_spec
            else SceneSpec(scene_id="mirror_room", mirror_enabled=True)
        )
        self.backend_overrides = (
            self.experiment_spec.resolve_backend_overrides(
                self.experiment_runtime_options,
                self.scene_template,
            )
            if self.experiment_spec else None
        )
        self.num_runs = num_runs
        self.max_steps = max_steps
        self.output_dir = output_dir
        self.log_file = log_file
        self.json_export = json_export
        self.heatmap = heatmap
        self.save_obs = save_obs
        self.save_dataset = bool(save_dataset)
        self._obs_counter = 0
        self.sim_env: Optional[BaseEnv] = None
        self.dataset_root: Optional[Path] = None
        self.dataset_episodes_dir: Optional[Path] = None
        self.dataset_manifest_path: Optional[Path] = None
        self._dataset_enabled = bool(
            self.save_dataset
            and self.experiment_id in _AUTO_NAV_EXPERIMENTS
            and self.is_auto_nav
        )
        self._autopilot_sleep_sec = (
            0.0
            if (
                self.experiment_id in _AUTO_NAV_EXPERIMENTS
                and self.is_auto_nav
                and self.env_type == "mujoco"
                and self.mujoco_headless
            )
            else _E6_AUTOPILOT_SLEEP_SEC
        )

        llm_slug = re.sub(r'[:/\\]', '_', nav_mode)   # e.g. "gemma3_4b", "human"
        session_id = time.strftime("%Y%m%d_%H%M%S")
        
        import os
        msr_level_str = os.getenv("MSR_LEVEL", "2")
        msr_tag = f"level_{msr_level_str}"
        variant_tag = _experiment_variant_tag(
            experiment_id,
            body_style=self.body_style,
            self_perception_mode=self.self_perception_mode,
        )

        if experiment_id in _EXPERIMENT_SCOPED_OUTPUT_ROOTS:
            experiment_root = Path(output_dir) / env_type / experiment_id / self.client_type / llm_slug
            if variant_tag:
                experiment_root = experiment_root / variant_tag
            self.output_dir = str(experiment_root / msr_tag / session_id)
        else:
            self.output_dir = str(Path(output_dir) / env_type / self.client_type / llm_slug / msr_tag / session_id)

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.current_run_dir: Optional[Path] = None
        self.current_obs_dir: Optional[Path] = None
        if self._dataset_enabled:
            self._initialize_dataset_layout()
        elif self.save_dataset:
            print("save_dataset=true is currently supported for experiment=e6/e7 with nav=auto. Ignoring.")

        if env_type == "mujoco" and self.is_human_nav and mujoco_headless:
            raise ValueError(
                "env=mujoco nav=human requires a display window. "
                "Add mujoco.headless=false to open the live viewer."
            )

        try:
            self.sim_env = _create_env(
                env_type=env_type,
                scene_spec=self.scene_template,
                backend_overrides=self.backend_overrides,
                third_person=third_person,
                camera_height=camera_height,
                body_style=self.body_style,
                mujoco_headless=mujoco_headless,
                mujoco_render_width=mujoco_render_width,
                mujoco_render_height=mujoco_render_height,
            )

            if self.is_human_nav:
                self.sim_env.bind_keys(self._handle_key)

            self.base_instruction = "Select the cube that matches your color."  # free-nav only
            self.agent_color_name: Optional[str] = None
            self.wrong_color: Optional[str] = None
            self.current_run_setup: Optional[RunSetup] = None
            self.current_object_positions_2d: dict = {}
            self._reset_state()

            self.logger: Optional[Logger] = None
        except Exception:
            self._close_resources()
            raise

    def _close_resources(self) -> None:
        if self.sim_env is not None:
            self.sim_env.close()
            self.sim_env = None

    # ------------------------------------------------------------------
    # Human navigation key callback
    # ------------------------------------------------------------------

    def _handle_key(self, evt) -> None:
        key = evt.key.lower()
        if self._is_e4_corridor_run() and self._human_active_run_id is not None:
            run_id = int(self._human_active_run_id)
            if key == "f":
                observation_metadata = self.sim_env.get_observation_metadata()
                self.iteration += 1
                self.bumped_wall = False
                goal_status = {
                    "status": "at-goal" if self._e4_is_terminal_row(self.current_pos) else "unknown",
                    "reason": (
                        "Human ended the episode at terminal row."
                        if self._e4_is_terminal_row(self.current_pos)
                        else "Human requested done before terminal row."
                    ),
                    "confidence": "high" if self._e4_is_terminal_row(self.current_pos) else "low",
                }
                self._record_step(
                    move="done",
                    view_description="",
                    reasoning=goal_status["reason"],
                    summary="",
                    identification=self._human_identification(),
                    goal_status=goal_status,
                    bumped_wall=False,
                    view_position=observation_metadata.view_position,
                    view_direction=observation_metadata.view_direction,
                    mirror_visible=observation_metadata.mirror_visible,
                    ego_reflection_visible=observation_metadata.ego_reflection_visible,
                )
                outcome = self._finalize_episode(run_id, self._new_outcome(run_id, None, True))
                self._human_episode_outcome = outcome
                self._print_e4_run_result(run_id, outcome)
                self._save_human_observation()
                return

            if key in ("w", "a", "d"):
                observation_metadata = self.sim_env.get_observation_metadata()
                at_terminal_row = self._e4_is_terminal_row(self.current_pos)
                self.iteration += 1
                if at_terminal_row:
                    self.bumped_wall = True
                    self._record_step(
                        move=key,
                        view_description="",
                        reasoning="Movement at terminal row is invalid for e4; use done/f.",
                        summary="",
                        identification=self._human_identification(),
                        goal_status={
                            "status": "not-at-goal",
                            "reason": "Moved at terminal row instead of ending with done.",
                            "confidence": "high",
                        },
                        bumped_wall=True,
                        view_position=observation_metadata.view_position,
                        view_direction=observation_metadata.view_direction,
                        mirror_visible=observation_metadata.mirror_visible,
                        ego_reflection_visible=observation_metadata.ego_reflection_visible,
                    )
                    self.step_records[-1]["e4_terminal_violation"] = True
                    self.step_records[-1]["e4_terminal_row"] = True
                    outcome = self._finalize_episode(run_id, self._new_outcome(run_id, None, False))
                    self._human_episode_outcome = outcome
                    self._print_e4_run_result(run_id, outcome)
                    self._save_human_observation()
                    return

                previous_pos = self.current_pos
                x, z, direction, bumped = self.sim_env.step(key)
                self.bumped_wall = bumped
                self.previous_pos = previous_pos
                self.current_pos = (x, z)
                self.current_dir = direction
                self.trajectory.append(self.current_pos)
                self._record_step(
                    move=key,
                    view_description="",
                    reasoning="",
                    summary="",
                    identification=self._human_identification(),
                    goal_status={
                        "status": "not-at-goal",
                        "reason": "Still navigating toward terminal row.",
                        "confidence": "low",
                    },
                    bumped_wall=bumped,
                    view_position=observation_metadata.view_position,
                    view_direction=observation_metadata.view_direction,
                    mirror_visible=observation_metadata.mirror_visible,
                    ego_reflection_visible=observation_metadata.ego_reflection_visible,
                )
                self._save_human_observation()
                return

            if key in ("s", " "):
                return

        if self._human_finish_requested and key != "f":
            return
        if key == "f" and self._human_cube_task_enabled():
            if not self._human_finish_requested:
                self._human_finish_requested = True
                self._human_finish_observation = self.sim_env.get_observation_metadata()
                self._save_human_observation()
            return

        if key in ("w", "s", "a", "d"):
            observation_metadata = (
                self.sim_env.get_observation_metadata()
                if self._human_cube_task_enabled()
                else None
            )
            previous_pos = self.current_pos
            x, z, direction, bumped = self.sim_env.step(key)
            self.bumped_wall = bumped
            self.previous_pos = previous_pos
            self.current_pos = (x, z)
            self.current_dir = direction
            self.trajectory.append(self.current_pos)
            if observation_metadata is not None:
                self.iteration += 1
                self._record_step(
                    move=key,
                    view_description="",
                    reasoning="",
                    summary="",
                    identification=self._human_identification(),
                    bumped_wall=bumped,
                    view_position=observation_metadata.view_position,
                    view_direction=observation_metadata.view_direction,
                    mirror_visible=observation_metadata.mirror_visible,
                    ego_reflection_visible=observation_metadata.ego_reflection_visible,
                )
        elif key == " ":
            self.sim_env.jump()
        else:
            return
        self._save_human_observation()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self.iteration = 0
        self.step_history: list = []
        self.previous_pos = None
        self.current_pos = self.sim_env.position
        self.bumped_wall = False
        self.current_dir: str = self.sim_env.direction
        self.trajectory: list = []
        self.step_records: list = []
        self._human_finish_requested = False
        self._human_finish_observation = None
        self._human_episode_outcome = None
        self._human_active_run_id = None

    def _human_cube_task_enabled(self) -> bool:
        if not self.is_human_nav:
            return False
        if self.task_spec is None:
            return True
        return self.task_spec.task_id == "cube_task"

    def _human_identification(self, selected_cube: Optional[str] = None) -> dict:
        if selected_cube is None:
            return {
                "my_color": "unknown",
                "source": "other",
                "confidence": "low",
            }
        return {
            "my_color": selected_cube,
            "source": "other",
            "confidence": "high",
        }

    def _save_human_observation(self) -> None:
        if not self.save_obs or self.current_obs_dir is None:
            return
        self._obs_counter += 1
        obs_path = self.current_obs_dir / f"{self._obs_counter:06d}.jpg"
        self.sim_env.capture_image(str(obs_path))

    def _build_run_context(self) -> dict:
        paper_id = getattr(self, "paper_experiment_id", None)
        if paper_id is None:
            return {}
        return {
            "experiment_suite": "paper",
            "experiment_preset": f"paper/{paper_id}",
            "paper_experiment_id": paper_id,
            "implementation_experiment_id": self.experiment_id,
        }

    def _is_e4_corridor_run(self) -> bool:
        return bool(
            self.task_spec is not None
            and self.task_spec.task_id == "corridor_task"
            and self.current_run_setup is not None
            and self.current_run_setup.scene_spec.layout_kind == "corridor"
        )

    def _e4_expected_done_step(self) -> int:
        if self.current_run_setup is None:
            return 10
        raw = self.current_run_setup.metadata.get("expected_done_step")
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return 10 if self.current_run_setup.scene_spec.mirror_enabled else 20
        return value if value in {10, 20} else (10 if self.current_run_setup.scene_spec.mirror_enabled else 20)

    def _e4_terminal_target_x(self) -> Optional[float]:
        if not self._is_e4_corridor_run():
            return None
        scene_spec = self.current_run_setup.scene_spec
        tile_len = float(scene_spec.length) / 10.0
        if scene_spec.mirror_enabled:
            return float(scene_spec.length) / 2.0 - tile_len / 2.0
        mirror_plane_x = float(scene_spec.mirror_plane_x or 0.0)
        return mirror_plane_x + float(scene_spec.length) - tile_len / 2.0

    def _e4_is_terminal_row(self, position: Optional[tuple[float, float]] = None) -> bool:
        if not self._is_e4_corridor_run():
            return False
        pos = position if position is not None else self.current_pos
        target_x = self._e4_terminal_target_x()
        if target_x is None:
            return False
        return abs(float(pos[0]) - float(target_x)) <= 0.25

    def _print_e4_run_result(self, run_id: int, outcome: RunOutcome) -> None:
        task_result = outcome.task_result or {}
        success = bool(task_result.get("success"))
        status = "SUCCESS" if success else "FAILURE"
        variant = task_result.get("e4_variant", "unknown")
        expected_done = task_result.get("expected_done_step")
        actual_done = task_result.get("actual_done_step")
        reason = (
            task_result.get("failure_reason")
            or (task_result.get("goal_status") or {}).get("reason")
            or "n/a"
        )
        print(
            f"[E4] Run {run_id}: {status} | variant={variant} | "
            f"done_step={actual_done}/{expected_done} | reason={reason}"
        )

    def _initialize_dataset_layout(self) -> None:
        self.dataset_root = Path(_DATASET_DIR_NAME)
        self.dataset_episodes_dir = self.dataset_root / "episodes"
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.dataset_episodes_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_manifest_path = self.dataset_root / "dataset_manifest.json"
        manifest = {
            "dataset_name": _DATASET_DIR_NAME,
            "experiment_id": self.experiment_id,
            "nav_mode": self.nav_mode,
            "num_episodes_target": int(self.num_runs),
            "max_steps_per_episode": int(self.max_steps),
            "autopilot_sleep_sec": float(self._autopilot_sleep_sec),
            "created_unix": int(time.time()),
        }
        self.dataset_manifest_path.write_text(json.dumps(manifest, indent=2))

    def _dataset_episode_dir(self, run_id: int) -> Path:
        if self.dataset_episodes_dir is None:
            raise RuntimeError("Dataset layout is not initialized.")
        episode_dir = self.dataset_episodes_dir / f"episode_{run_id:04d}"
        (episode_dir / "rgb").mkdir(parents=True, exist_ok=True)
        return episode_dir

    def _capture_dataset_frame(self, run_id: int, frame_index: int) -> Optional[str]:
        if not self._dataset_enabled:
            return None
        try:
            episode_dir = self._dataset_episode_dir(run_id)
            frame_name = f"{frame_index:06d}.jpg"
            frame_path = episode_dir / "rgb" / frame_name
            self.sim_env.capture_image(str(frame_path))
            return f"rgb/{frame_name}"
        except Exception as exc:
            print(f"Dataset frame capture failed for run={run_id}, frame={frame_index}: {exc}")
            return None

    def _snapshot_distractor_states(self) -> list[dict[str, Any]]:
        states = getattr(self.sim_env, "_e5_distractor_nav", None)
        if not states:
            return []
        snapshots: list[dict[str, Any]] = []
        color_by_id: dict[str, str] = {}
        if self.current_run_setup is not None:
            color_by_id = {
                obj.object_id: str(obj.color)
                for obj in self.current_run_setup.scene_spec.objects
                if obj.kind == "distractor_robot" and obj.color is not None
            }
        for state in states:
            object_id = str(state.get("object_id", ""))
            snapshots.append({
                "object_id": object_id,
                "color": color_by_id.get(object_id),
                "x": float(state.get("x", 0.0)),
                "z": float(state.get("z", 0.0)),
                "yaw": float(state.get("yaw", 0.0)),
                "target_x": float(state.get("target_x", 0.0)),
                "target_z": float(state.get("target_z", 0.0)),
                "last_action": str(state.get("last_action", "none")),
            })
        return snapshots

    def _export_dataset_episode(
        self,
        run_id: int,
        *,
        outcome: RunOutcome,
        step_details: list[dict[str, Any]],
        initial_frame_relpath: Optional[str],
    ) -> None:
        if not self._dataset_enabled:
            return
        episode_dir = self._dataset_episode_dir(run_id)
        scene_spec = self.current_run_setup.scene_spec if self.current_run_setup is not None else None
        distractor_objects = [
            obj for obj in (scene_spec.objects if scene_spec is not None else [])
            if obj.kind == "distractor_robot"
        ]
        dataset_payload = {
            "episode_id": run_id,
            "experiment_id": self.experiment_id,
            "scene_id": scene_spec.scene_id if scene_spec is not None else None,
            "task_id": self.task_spec.task_id if self.task_spec is not None else None,
            "nav_mode": self.nav_mode,
            "completed": bool(outcome.completed),
            "steps_executed": int(self.iteration),
            "max_steps": int(self.max_steps),
            "autopilot_sleep_sec": float(self._autopilot_sleep_sec),
            "agent": {
                "color": self.agent_color_name,
                "spawn_pos": list(scene_spec.spawn_pos) if scene_spec is not None else None,
            },
            "distractors": {
                "count": len(distractor_objects),
                "objects": [
                    {
                        "object_id": obj.object_id,
                        "color": obj.color,
                        "spawn_position": [float(obj.position[0]), float(obj.position[1])],
                        "yaw": float(obj.metadata.get("yaw", 0.0)),
                    }
                    for obj in distractor_objects
                ],
            },
            "num_agents_total": 1 + len(distractor_objects),
            "environment": {
                "layout_kind": scene_spec.layout_kind if scene_spec is not None else None,
                "mirror_enabled": bool(scene_spec.mirror_enabled) if scene_spec is not None else None,
                "mirror_window_z": list(scene_spec.mirror_window_z) if scene_spec and scene_spec.mirror_window_z is not None else None,
                "turn_step_deg": float(scene_spec.turn_step_deg) if scene_spec is not None else None,
            },
            "ego_reflection_seen": any(
                record.get("ego_reflection_visible") is True
                for record in self.step_records
            ),
            "trajectory": [list(pos) for pos in self.trajectory],
            "actions": [record.get("action") for record in self.step_records],
            "initial_frame": initial_frame_relpath,
            "steps": step_details,
            "step_records": self.step_records,
        }
        episode_json = episode_dir / "episode.json"
        episode_json.write_text(json.dumps(dataset_payload, indent=2))

    def _normalize_human_finish_reply(self, reply: Optional[dict]) -> Optional[dict[str, str]]:
        if reply is None:
            return None
        selected_cube = str(reply.get("selected_cube", "")).strip().lower()
        reasoning = str(reply.get("reasoning", "")).strip()
        if selected_cube not in _VALID_CUBE_SELECTIONS:
            raise ValueError(f"Unsupported human selected_cube={selected_cube!r}.")
        if not reasoning:
            raise ValueError("Human finish reasoning must be non-empty.")
        return {
            "selected_cube": selected_cube,
            "reasoning": reasoning,
        }

    def _human_finish_ui_backend(self) -> str:
        return "opencv"

    def _human_cube_choices(self) -> tuple[str, ...]:
        if self.current_run_setup is None:
            return tuple(_COLOR_POOL[:3])
        colors = [
            str(obj.color).strip().lower()
            for obj in self.current_run_setup.scene_spec.objects
            if obj.kind == "cube" and obj.color is not None
        ]
        unique: list[str] = []
        for color in colors:
            if color and color not in unique:
                unique.append(color)
        if unique:
            return tuple(unique)
        return tuple(_COLOR_POOL[:3])

    def _default_single_run_cubes(self) -> dict:
        return {
            "red": CUBE_SLOTS[0],
            "green": CUBE_SLOTS[1],
            "blue": CUBE_SLOTS[2],
        }

    def _initialize_single_run(self) -> None:
        cube_positions = self._default_single_run_cubes()
        run_setup = RunSetup(
            agent_color_name=_DEFAULT_SINGLE_RUN_COLOR,
            scene_spec=SceneSpec(
                scene_id="mirror_room",
                mirror_enabled=True,
                objects=build_cube_objects(cube_positions),
            ),
            cube_positions=cube_positions,
        )
        self._apply_run_setup(run_setup)

    def _apply_run_setup(self, run_setup: RunSetup) -> None:
        self.sim_env.reset(run_setup)
        self.agent_color_name = run_setup.agent_color_name
        self.wrong_color = run_setup.wrong_color
        self.current_run_setup = run_setup
        self._reset_state()
        self.current_object_positions_2d = self.sim_env.object_positions_2d

    def _record_step(
        self,
        move: str,
        view_description: str,
        reasoning: str,
        summary: str,
        identification: dict,
        goal_status: Optional[dict] = None,
        task_payload: Optional[dict] = None,
        selected_cube: Optional[str] = None,
        bumped_wall: Optional[bool] = None,
        inner_reasoning: Optional[str] = None,
        view_position: Optional[tuple[float, float, float]] = None,
        view_direction: Optional[tuple[float, float, float]] = None,
        mirror_visible: Optional[bool] = None,
        ego_reflection_visible: Optional[bool] = None,
    ) -> None:
        record = {
            "step": self.iteration,
            "action": move,
            "position": list(self.current_pos),
            "direction": self.current_dir,
            "bumped_wall": self.bumped_wall if bumped_wall is None else bumped_wall,
            "view_description": view_description,
            "reasoning": reasoning,
            "inner_reasoning": inner_reasoning,
            "summary": summary,
            "identification": identification,
            "selected_cube": selected_cube,
        }
        if goal_status is not None:
            record["goal_status"] = goal_status
        if task_payload:
            record["task_payload"] = dict(task_payload)
        if view_position is not None:
            record["view_position"] = [float(value) for value in view_position]
        if view_direction is not None:
            record["view_direction"] = [float(value) for value in view_direction]
        if mirror_visible is not None:
            record["mirror_visible"] = bool(mirror_visible)
        if ego_reflection_visible is not None:
            record["ego_reflection_visible"] = bool(ego_reflection_visible)
        self.step_records.append(record)

    def _reset_run(self, run_id: int, run_setup: RunSetup) -> None:
        if self.experiment_spec is not None:
            expected = scene_geometry_signature(self.scene_template)
            actual = scene_geometry_signature(run_setup.scene_spec)
            if actual != expected:
                if self.task_spec is not None and self.task_spec.task_id == "corridor_task":
                    expected_scene = self.scene_template
                    actual_scene = run_setup.scene_spec
                    compatible_e4_variant = (
                        actual_scene.scene_id == expected_scene.scene_id
                        and actual_scene.layout_kind == expected_scene.layout_kind
                        and float(actual_scene.length) == float(expected_scene.length)
                        and float(actual_scene.width) == float(expected_scene.width)
                        and float(actual_scene.height) == float(expected_scene.height)
                        and float(actual_scene.wall_thickness) == float(expected_scene.wall_thickness)
                        and actual_scene.spawn_pos == expected_scene.spawn_pos
                        and float(actual_scene.spawn_yaw) == float(expected_scene.spawn_yaw)
                        and actual_scene.mirror_plane_x == expected_scene.mirror_plane_x
                        and float(actual_scene.turn_step_deg) == float(expected_scene.turn_step_deg)
                    )
                    if not compatible_e4_variant:
                        raise ValueError(
                            "Experiment plugin returned an incompatible e4 run_setup.scene_spec. "
                            "Only mirror/open corridor variant toggles are allowed per run."
                        )
                else:
                    raise ValueError(
                        "Experiment plugin returned a run_setup.scene_spec with geometry that "
                        "differs from the experiment scene_template. Static geometry per "
                        "experiment is required in v1."
                    )

        # Initialize run-specific output directory
        self.current_run_dir = Path(self.output_dir) / f"run_{run_id:02d}"
        self.current_obs_dir = self.current_run_dir / "obs"
        self.current_run_dir.mkdir(parents=True, exist_ok=True)
        self.current_obs_dir.mkdir(parents=True, exist_ok=True)

        # Re-initialize logger for this run
        self.logger = Logger(output_file=str(self.current_run_dir / self.log_file))
        self.logger.reset()

        self._apply_run_setup(run_setup)
        header = f"\n--- Run {run_id}/{self.num_runs}"
        if _should_display_agent_color(task_spec=self.task_spec, run_setup=run_setup):
            header += f" | agent={run_setup.agent_color_name}"
        if run_setup.wrong_color:
            header += f" | misleading={run_setup.wrong_color}"
        header += " ---"
        print(header)

    def _new_outcome(self, run_id: int, selected_cube: Optional[str], completed: bool) -> RunOutcome:
        """Construct a RunOutcome with all context fields populated."""
        task_result: dict = {}
        if self.task_spec is not None:
            task_result = self.task_spec.capabilities.extract_task_output(
                self.step_records,
                completed,
                self.current_run_setup,
            )
        mirror_exposure_metrics = extract_mirror_exposure_metrics(
            self.step_records,
            completed=completed,
            experiment_id=self.experiment_id,
        )
        return RunOutcome(
            run_id=run_id,
            agent_color=self.agent_color_name,
            wrong_color=self.wrong_color,
            selected_cube=selected_cube,
            completed=completed,
            steps=self.iteration,
            experiment_id=self.experiment_id,
            scene_id=self.current_run_setup.scene_spec.scene_id if self.current_run_setup else None,
            task_id=self.task_spec.task_id if self.task_spec else None,
            task_result=task_result,
            mirror_exposure_metrics=mirror_exposure_metrics,
        )

    # ------------------------------------------------------------------
    # Run dispatch
    # ------------------------------------------------------------------

    def run(self):
        try:
            if self.is_auto_nav:
                self._run_auto_nav()
            elif self.is_human_nav:
                self._run_human_nav()
            elif self.experiment_id:
                self._run_experiment()
            else:
                self._run_llm_nav()
        finally:
            self._close_resources()

    def _run_human_nav(self):
        if not self._human_cube_task_enabled():
            if self.task_spec is not None and self.task_spec.task_id == "corridor_task":
                outcomes: list[RunOutcome] = []
                for run_id in range(1, self.num_runs + 1):
                    run_setup = (
                        self.experiment_spec.initialize_run(
                            run_id,
                            self.experiment_runtime_options,
                        )
                        if self.experiment_spec is not None
                        else RunSetup(
                            agent_color_name="red",
                            scene_spec=SceneSpec(
                                scene_id="narrow_corridor_mirror",
                                mirror_enabled=True,
                                layout_kind="corridor",
                            ),
                        )
                    )
                    self._reset_run(run_id, run_setup)
                    outcome = self._run_single_human_e4_episode(run_id)
                    outcomes.append(outcome)
                if outcomes and self.experiment_spec is not None:
                    print_metrics(outcomes, self.experiment_id)
                    if self.json_export:
                        save_experiment_json(
                            outcomes, self.experiment_id, self.output_dir,
                            run_context=self._build_run_context(),
                        )
                return

            if self.experiment_spec is not None:
                self._apply_run_setup(
                    self.experiment_spec.initialize_run(
                        1,
                        self.experiment_runtime_options,
                    )
                )
            else:
                self._initialize_single_run()
            while True:
                self.sim_env.tick(30)
                if self.sim_env.update_frame() is False:
                    break
            return

        outcomes: list[RunOutcome] = []
        for run_id in range(1, self.num_runs + 1):
            if self.experiment_spec is not None:
                run_setup = self.experiment_spec.initialize_run(
                    run_id,
                    self.experiment_runtime_options,
                )
                self._reset_run(run_id, run_setup)
            else:
                self._initialize_single_run()
                print(f"\n--- Run {run_id}/{self.num_runs} ---")

            outcome = self._run_single_human_cube_episode(run_id)
            outcomes.append(outcome)
            if not outcome.completed:
                break

        if outcomes and self.experiment_spec is not None:
            print_metrics(outcomes, self.experiment_id)
            if self.json_export:
                save_experiment_json(
                    outcomes, self.experiment_id, self.output_dir,
                    run_context=self._build_run_context(),
                )

    def _run_single_human_e4_episode(self, run_id: int) -> RunOutcome:
        self._human_active_run_id = run_id
        self._human_episode_outcome = None

        while True:
            if self._human_episode_outcome is not None:
                outcome = self._human_episode_outcome
                self._human_episode_outcome = None
                self._human_active_run_id = None
                return outcome

            self.sim_env.tick(30)
            if self.sim_env.update_frame() is False:
                self._human_active_run_id = None
                if self._human_episode_outcome is not None:
                    outcome = self._human_episode_outcome
                    self._human_episode_outcome = None
                    return outcome
                outcome = self._finalize_episode(run_id, self._new_outcome(run_id, None, False))
                self._print_e4_run_result(run_id, outcome)
                return outcome

            if self.iteration >= self.max_steps:
                self._human_active_run_id = None
                outcome = self._finalize_episode(run_id, self._new_outcome(run_id, None, False))
                self._print_e4_run_result(run_id, outcome)
                return outcome

    def _run_single_human_cube_episode(self, run_id: int) -> RunOutcome:
        while True:
            self.sim_env.tick(30)
            if self.sim_env.update_frame() is False:
                return self._finalize_episode(run_id, self._new_outcome(run_id, None, False))

            if not self._human_finish_requested:
                continue

            try:
                reply = self._normalize_human_finish_reply(
                    prompt_human_cube_selection(
                        ui_backend=self._human_finish_ui_backend(),
                        valid_cubes=self._human_cube_choices(),
                    )
                )
            except Exception as exc:
                print(f"Human finish dialog failed: {exc}")
                self._human_finish_requested = False
                self._human_finish_observation = None
                continue

            self._human_finish_requested = False
            if reply is None:
                self._human_finish_observation = None
                continue

            observation_metadata = self._human_finish_observation or self.sim_env.get_observation_metadata()
            self._human_finish_observation = None
            selected_cube = reply["selected_cube"]
            self.iteration += 1
            self._record_step(
                move="done",
                view_description="",
                reasoning=reply["reasoning"],
                summary="",
                identification=self._human_identification(selected_cube),
                selected_cube=selected_cube,
                bumped_wall=False,
                view_position=observation_metadata.view_position,
                view_direction=observation_metadata.view_direction,
                mirror_visible=observation_metadata.mirror_visible,
                ego_reflection_visible=observation_metadata.ego_reflection_visible,
            )
            return self._finalize_episode(
                run_id,
                self._new_outcome(run_id, selected_cube, True),
            )

    # ------------------------------------------------------------------
    # Auto navigation
    # ------------------------------------------------------------------

    def _run_auto_nav(self):
        if self.experiment_spec is None:
            raise ValueError("nav=auto requires an experiment preset (e.g. experiment=e6 or experiment=e7).")
        if self.experiment_id not in _AUTO_NAV_EXPERIMENTS:
            raise ValueError("nav=auto is currently supported only for experiment=e6/e7.")
        self._run_auto_experiment()

    def _run_single_auto_episode(self, run_id: int) -> RunOutcome:
        from experiments.specs.e5.experiment import (
            _DISTRACTOR_ACTION_MOVE_STEP,
            _distractor_nav_bounds,
            _sample_distractor_target,
            _select_distractor_action,
        )

        scene_spec = self.current_run_setup.scene_spec
        mirror_plane_x = (
            float(scene_spec.mirror_plane_x)
            if scene_spec.mirror_plane_x is not None
            else -scene_spec.wall_thickness / 2
        )
        x, z = self.current_pos
        bounds = _distractor_nav_bounds(
            scene_spec,
            mirror_plane_x=mirror_plane_x,
            x=float(x),
        )
        target_x, target_z = _sample_distractor_target(
            bounds,
            current=(float(x), float(z)),
        )
        turn_step = math.radians(float(scene_spec.turn_step_deg))
        stuck_steps = 0
        turn_switches = 0
        last_action = "none"
        step_details: list[dict[str, Any]] = []
        initial_frame = self._capture_dataset_frame(run_id, 0)

        while self.iteration < self.max_steps:
            self.sim_env.tick(30)
            if self.sim_env.update_frame() is False:
                outcome = self._finalize_episode(run_id, self._new_outcome(run_id, None, False))
                self._export_dataset_episode(
                    run_id,
                    outcome=outcome,
                    step_details=step_details,
                    initial_frame_relpath=initial_frame,
                )
                return outcome

            observation_metadata = self.sim_env.get_observation_metadata()
            frame_relpath = self._capture_dataset_frame(run_id, self.iteration + 1)
            x, z = self.current_pos
            dist_before = math.hypot(target_x - x, target_z - z)
            if dist_before <= _E6_AUTOPILOT_TARGET_REACHED_EPS:
                target_x, target_z = _sample_distractor_target(
                    bounds,
                    current=(float(x), float(z)),
                )
                dist_before = math.hypot(target_x - x, target_z - z)
            step_target_x, step_target_z = target_x, target_z

            action = _select_distractor_action(
                x=float(x),
                z=float(z),
                yaw=float(getattr(self.sim_env, "_yaw", 0.0)),
                target_x=float(target_x),
                target_z=float(target_z),
                bounds=bounds,
                move_step=float(_DISTRACTOR_ACTION_MOVE_STEP),
                turn_step=float(turn_step),
            )

            previous_pos = self.current_pos
            nx, nz, direction, bumped = self.sim_env.step(action)
            self.bumped_wall = bumped
            self.previous_pos = previous_pos
            self.current_pos = (nx, nz)
            self.current_dir = direction
            self.trajectory.append(self.current_pos)
            self.iteration += 1

            dist_after = math.hypot(target_x - nx, target_z - nz)
            no_progress = dist_after >= (dist_before - 0.05)
            turn_switched = (
                action in {"a", "d"}
                and last_action in {"a", "d"}
                and action != last_action
            )
            if no_progress or bumped:
                stuck_steps += 1
            else:
                stuck_steps = 0
            if turn_switched:
                turn_switches += 1
            else:
                turn_switches = 0
            last_action = action

            if (
                dist_after <= _E6_AUTOPILOT_TARGET_REACHED_EPS
                or bumped
                or stuck_steps >= 5
                or turn_switches >= 4
            ):
                target_x, target_z = _sample_distractor_target(
                    bounds,
                    current=(float(nx), float(nz)),
                )
                stuck_steps = 0
                turn_switches = 0

            step_details.append({
                "step": int(self.iteration),
                "action": action,
                "frame": frame_relpath,
                "position_before": [float(previous_pos[0]), float(previous_pos[1])],
                "position_after": [float(nx), float(nz)],
                "direction_after": direction,
                "target_before_step": [float(step_target_x), float(step_target_z)],
                "distance_before": float(dist_before),
                "distance_after": float(dist_after),
                "bumped_wall": bool(bumped),
                "stuck_steps": int(stuck_steps),
                "turn_switches": int(turn_switches),
                "mirror_visible": bool(observation_metadata.mirror_visible),
                "ego_reflection_visible": (
                    None
                    if observation_metadata.ego_reflection_visible is None
                    else bool(observation_metadata.ego_reflection_visible)
                ),
                "view_position": [float(v) for v in observation_metadata.view_position],
                "view_direction": [float(v) for v in observation_metadata.view_direction],
                "distractors": self._snapshot_distractor_states(),
            })

            view_description = (
                f"Autopilot active in {self.experiment_id}: moving through the room with distractors."
            )
            reasoning = (
                f"Navigating toward waypoint x={target_x:.2f}, z={target_z:.2f}; "
                f"selected action '{action}'."
            )
            summary = (
                f"Step {self.iteration}/{self.max_steps}; "
                f"position=({nx:.2f}, {nz:.2f}), target=({target_x:.2f}, {target_z:.2f})."
            )
            identification = self._human_identification()

            self._record_step(
                move=action,
                view_description=view_description,
                reasoning=reasoning,
                summary=summary,
                identification=identification,
                bumped_wall=bumped,
                view_position=observation_metadata.view_position,
                view_direction=observation_metadata.view_direction,
                mirror_visible=observation_metadata.mirror_visible,
                ego_reflection_visible=observation_metadata.ego_reflection_visible,
            )
            self.sim_env.set_caption(
                "<div style='max-width:800px; word-wrap:break-word;'>"
                f"<b>Run:</b> {run_id}/{self.num_runs}<br>"
                f"<b>Mode:</b> auto ({self.experiment_id})<br>"
                f"<b>Action:</b> {action}<br>"
                f"<b>Target:</b> x={target_x:.2f}, z={target_z:.2f}<br>"
                f"<b>Step:</b> {self.iteration}/{self.max_steps}"
                "</div>"
            )

            if self.sim_env.update_frame() is False:
                outcome = self._finalize_episode(run_id, self._new_outcome(run_id, None, False))
                self._export_dataset_episode(
                    run_id,
                    outcome=outcome,
                    step_details=step_details,
                    initial_frame_relpath=initial_frame,
                )
                return outcome
            if self._autopilot_sleep_sec > 0.0:
                time.sleep(self._autopilot_sleep_sec)

        outcome = self._finalize_episode(run_id, self._new_outcome(run_id, None, True))
        self._export_dataset_episode(
            run_id,
            outcome=outcome,
            step_details=step_details,
            initial_frame_relpath=initial_frame,
        )
        return outcome

    def _run_auto_experiment(self) -> List[RunOutcome]:
        spec = self.experiment_spec
        print(
            f"Starting auto experiment {self.experiment_id.upper()} | mode=auto | runs={self.num_runs} | "
            f"steps_per_run={self.max_steps} | sleep={self._autopilot_sleep_sec:.1f}s"
        )

        outcomes: List[RunOutcome] = []
        for run_id in range(1, self.num_runs + 1):
            run_setup = spec.initialize_run(run_id, self.experiment_runtime_options)
            self._reset_run(run_id, run_setup)
            outcome = self._run_single_auto_episode(run_id)
            outcomes.append(outcome)

        print_metrics(outcomes, self.experiment_id)
        if self.json_export:
            save_experiment_json(
                outcomes, self.experiment_id, self.output_dir,
                run_context=self._build_run_context(),
            )
        return outcomes

    # ------------------------------------------------------------------
    # Standard single-run LLM navigation
    # ------------------------------------------------------------------

    def _run_llm_nav(self):
        print(f"Running LLM navigation with model: {self.llm_model} (client={self.client_type}) | runs={self.num_runs}")

        for run_id in range(1, self.num_runs + 1):
            run_setup = (
                self.experiment_spec.initialize_run(run_id, self.experiment_runtime_options)
                if self.experiment_spec is not None
                else RunSetup(
                    agent_color_name=_DEFAULT_SINGLE_RUN_COLOR,
                    scene_spec=SceneSpec(
                        scene_id="mirror_room",
                        mirror_enabled=True,
                        objects=build_cube_objects(self._default_single_run_cubes()),
                    ),
                    cube_positions=self._default_single_run_cubes(),
                )
            )
            self._reset_run(run_id, run_setup)

            completed = False
            selected_cube = None

            if self.client_type == "ollama" and run_id == 1:
                try:
                    available_models = list_models(self.ollama_host)
                    if self.llm_model not in available_models:
                        preview = ", ".join(available_models[:8]) if available_models else "none"
                        print(
                            "Navigation stopped: model not found in local Ollama.\n"
                            f"Requested: {self.llm_model}\n"
                            f"Available (first 8): {preview}\n"
                            f"Ollama host: {self.ollama_host}"
                        )
                        return
                except Exception as exc:
                    print(f"Warning: could not query Ollama models before start: {exc}")

            while True:
                self.sim_env.tick(30)
                if self.sim_env.update_frame() is False:
                    print("Window closed. Stopping navigation.")
                    return

                step = self.iteration + 1
                observation_metadata = self.sim_env.get_observation_metadata()
                print(f"[Step {step}] Capturing RGB observation...")
                capture_t0 = time.perf_counter()
                img_path = self.current_obs_dir / f"{step:06d}.jpg"
                try:
                    img_path = Path(self.sim_env.capture_image(str(img_path)))
                except Exception as exc:
                    print(f"Navigation stopped: capture failed at step {step}: {exc}")
                    break
                print(f"[Step {step}] Capture in {time.perf_counter() - capture_t0:.2f}s")

                if self.print_reasoning:
                    print(f"[Step {step}] Thinking:")
                print(f"[Step {step}] Querying model...")
                query_t0 = time.perf_counter()
                try:
                    view_description, reasoning, move, summary, selected_cube_reply, thinking, identification, task_payload = get_next_move(
                        self.base_instruction,
                        model=self.llm_model,
                        image_path=str(img_path),
                        history=self.step_history,
                        previous_pos=self.previous_pos,
                        current_pos=self.current_pos,
                        bumped_wall=self.bumped_wall,
                        stream_thinking=self.print_reasoning,
                        wrong_color=self.wrong_color,
                        host=self.ollama_host,
                        third_person=self.third_person,
                        current_dir=self.current_dir,
                        client_type=self.client_type,
                        api_key=self.api_key,
                        base_url=getattr(self, "provider_base_url", None),
                        http_referer=getattr(self, "provider_http_referer", None),
                        app_title=getattr(self, "provider_app_title", None),
                        require_parameters=getattr(self, "provider_require_parameters", None),
                        min_request_interval_sec=getattr(
                            self, "provider_min_request_interval_sec", None
                        ),
                    )
                except Exception as exc:
                    print(f"Navigation stopped: model call failed at step {step}: {exc}")
                    break

                move = move.lower().strip()
                print(f"[Step {step}] action={move} ({time.perf_counter() - query_t0:.2f}s)")

                self.iteration += 1
                self.step_history.append({
                    "action": move,
                    "summary": summary,
                    "reasoning": reasoning,
                    "view_description": view_description,
                    "facing_before": self.current_dir,
                })
                self.step_history = self.step_history[-self.history_steps:]
                self.logger.log(role="Assistant", iteration=self.iteration,
                                reasoning=reasoning, action=move, summary=summary)

                id_str = (
                    f"<br><b>Selected Cube:</b> {selected_cube_reply}"
                    f"<br><b>Identification:</b> color={identification['my_color']} "
                    f"source={identification['source']} confidence={identification['confidence']}"
                    if move == "done" else ""
                )
                self.sim_env.set_caption(
                    "<div style='max-width:800px; word-wrap:break-word;'>"
                    f"<b>Run:</b> {run_id}/{self.num_runs}<br>"
                    f"<b>View Description:</b> {view_description}<br>"
                    f"<b>Reasoning:</b> {reasoning}<br>"
                    f"<b>Summary:</b> {summary}<br>"
                    f"<b>Action:</b> {move}"
                    + id_str
                    + "</div>"
                )

                previous_pos = self.current_pos
                if move in ("w", "s", "a", "d"):
                    x, z, direction, bumped = self.sim_env.step(move)
                    self.bumped_wall = bumped
                    self.previous_pos = previous_pos
                    self.current_pos = (x, z)
                    self.current_dir = direction
                    self.trajectory.append(self.current_pos)
                    self._record_step(
                        move=move,
                        view_description=view_description,
                        reasoning=reasoning,
                        summary=summary,
                        identification=identification,
                        goal_status=task_payload.get("goal_status") if task_payload else None,
                        task_payload=task_payload,
                        inner_reasoning=thinking,
                        view_position=observation_metadata.view_position,
                        view_direction=observation_metadata.view_direction,
                        mirror_visible=observation_metadata.mirror_visible,
                        ego_reflection_visible=observation_metadata.ego_reflection_visible,
                    )
                elif move == "done":
                    print("Model requested stop with action 'done'.")
                    completed = True
                    selected_cube = (
                        selected_cube_reply
                        if selected_cube_reply and selected_cube_reply != "none"
                        else None
                    )
                    self._record_step(
                        move=move,
                        view_description=view_description,
                        reasoning=reasoning,
                        summary=summary,
                        identification=identification,
                        goal_status=task_payload.get("goal_status") if task_payload else None,
                        task_payload=task_payload,
                        selected_cube=selected_cube,
                        bumped_wall=False,
                        inner_reasoning=thinking,
                        view_position=observation_metadata.view_position,
                        view_direction=observation_metadata.view_direction,
                        mirror_visible=observation_metadata.mirror_visible,
                        ego_reflection_visible=observation_metadata.ego_reflection_visible,
                    )
                    break
                else:
                    print(f"Unsupported action '{move}'. Stopping loop.")
                    break

                if self.sim_env.update_frame() is False:
                    print("Window closed. Stopping navigation.")
                    return

                if self.iteration >= self.max_steps:
                    print("Max iterations reached. Stopping loop.")
                    break

            if self.heatmap:
                _save_heatmap_if_supported(
                    run_id=run_id,
                    trajectory=self.trajectory,
                    agent_color=self.agent_color_name or "red",
                    object_positions_2d=self.current_object_positions_2d,
                    output_dir=str(self.current_run_dir),
                    run_setup=self.current_run_setup,
                    task_spec=None,
                )
            if self.json_export:
                _outcome = RunOutcome(
                    run_id=run_id, agent_color=self.agent_color_name or "red",
                    wrong_color=None, selected_cube=selected_cube,
                    completed=completed, steps=self.iteration,
                    trajectory=self.trajectory.copy(),
                    cube_positions_2d=self.current_object_positions_2d.copy(),
                )
                save_run_json(
                    run_id,
                    _outcome,
                    self.step_records,
                    str(self.current_run_dir),
                    task_output=_outcome.task_result,
                    run_setup=self.current_run_setup,
                    run_context=self._build_run_context(),
                )

    # ------------------------------------------------------------------
    # Experiment: single episode
    # ------------------------------------------------------------------

    def _finalize_episode(self, run_id: int, outcome: RunOutcome) -> RunOutcome:
        outcome.trajectory = self.trajectory.copy()
        outcome.cube_positions_2d = self.current_object_positions_2d.copy()
        if self.json_export:
            save_run_json(
                run_id,
                outcome,
                self.step_records,
                str(self.current_run_dir),
                task_output=outcome.task_result,
                run_setup=self.current_run_setup,
                run_context=self._build_run_context(),
            )
        if self.heatmap:
            _save_heatmap_if_supported(
                run_id=run_id,
                trajectory=self.trajectory,
                agent_color=self.agent_color_name,
                object_positions_2d=self.current_object_positions_2d,
                output_dir=str(self.current_run_dir),
                run_setup=self.current_run_setup,
                task_spec=self.task_spec,
            )
        return outcome

    def _run_single_episode(self, run_id: int) -> RunOutcome:
        while True:
            self.sim_env.tick(30)
            if self.sim_env.update_frame() is False:
                return self._finalize_episode(run_id, self._new_outcome(run_id, None, False))

            step = self.iteration + 1
            observation_metadata = self.sim_env.get_observation_metadata()
            img_path = self.current_obs_dir / f"{step:06d}.jpg"
            try:
                img_path = Path(self.sim_env.capture_image(str(img_path)))
            except Exception as exc:
                print(f"  [Step {step}] Capture failed: {exc}")
                return self._finalize_episode(run_id, self._new_outcome(run_id, None, False))

            if self.print_reasoning:
                print(f"  [Step {step}] Thinking:")
            try:
                view_description, reasoning, move, summary, selected_cube, thinking, identification, task_payload = get_next_move(
                    model=self.llm_model,
                    task_spec=self.task_spec,
                    run_setup=self.current_run_setup,
                    image_path=str(img_path),
                    history=self.step_history,
                    previous_pos=self.previous_pos,
                    current_pos=self.current_pos,
                    bumped_wall=self.bumped_wall,
                    stream_thinking=self.print_reasoning,
                    host=self.ollama_host,
                    third_person=self.third_person,
                    current_dir=self.current_dir,
                    client_type=self.client_type,
                    api_key=self.api_key,
                    base_url=getattr(self, "provider_base_url", None),
                    http_referer=getattr(self, "provider_http_referer", None),
                    app_title=getattr(self, "provider_app_title", None),
                    require_parameters=getattr(self, "provider_require_parameters", None),
                    min_request_interval_sec=getattr(
                        self, "provider_min_request_interval_sec", None
                    ),
                )
            except Exception as exc:
                print(f"  [Step {step}] Model call failed: {exc}")
                return self._finalize_episode(run_id, self._new_outcome(run_id, None, False))

            move = move.lower().strip()
            print(f"  [Step {step}] action={move}"
                  + (f"  selected_cube={selected_cube}" if move == "done" else ""))
            is_e4_terminal_move_failure = bool(
                self._is_e4_corridor_run()
                and move in ("w", "a", "d")
                and self._e4_is_terminal_row(self.current_pos)
            )

            self.iteration += 1
            self.step_history.append({
                "action": move,
                "summary": summary,
                "reasoning": reasoning,
                "inner_reasoning": thinking,
                "view_description": view_description,
                "facing_before": self.current_dir,
            })
            self.step_history = self.step_history[-self.history_steps:]
            self.logger.log(role="Assistant", iteration=self.iteration,
                            thinking=thinking, reasoning=reasoning, action=move, summary=summary)

            show_agent_color = _should_display_agent_color(
                task_spec=self.task_spec,
                run_setup=self.current_run_setup,
            )
            self_description = {}
            if task_payload:
                maybe_self_description = task_payload.get("self_description")
                if isinstance(maybe_self_description, dict):
                    self_description = maybe_self_description
            self.sim_env.set_caption(
                "<div style='max-width:800px; word-wrap:break-word;'>"
                f"<b>Run:</b> {run_id}/{self.num_runs}"
                + (
                    f" &nbsp; <b>Agent:</b> {self.agent_color_name}"
                    if show_agent_color and self.agent_color_name
                    else ""
                )
                + "<br>"
                f"<b>View Description:</b> {view_description}<br>"
                f"<b>Reasoning:</b> {reasoning}<br>"
                f"<b>Summary:</b> {summary}<br>"
                f"<b>Action:</b> {move}"
                + (
                    f"<br><b>Selected Cube:</b> {selected_cube}"
                    f"<br><b>Identification:</b> color={identification['my_color']} "
                    f"source={identification['source']} confidence={identification['confidence']}"
                    if move == "done" and self.task_spec and self.task_spec.task_id == "cube_task" else ""
                )
                + (
                    f"<br><b>Self Description:</b> {self_description.get('overall', '')}"
                    if move == "done" and self_description.get("overall")
                    else ""
                )
                + (
                    f"<br><b>Clothing:</b> {self_description.get('clothing', '')}"
                    if move == "done" and self_description.get("clothing")
                    else ""
                )
                + "</div>"
            )

            if move == "done":
                cube = selected_cube if selected_cube and selected_cube != "none" else None
                self._record_step(
                    move=move,
                    view_description=view_description,
                    reasoning=reasoning,
                    summary=summary,
                    identification=identification,
                    goal_status=task_payload.get("goal_status") if task_payload else None,
                    selected_cube=cube,
                    bumped_wall=False,
                    inner_reasoning=thinking,
                    view_position=observation_metadata.view_position,
                    view_direction=observation_metadata.view_direction,
                    mirror_visible=observation_metadata.mirror_visible,
                    ego_reflection_visible=observation_metadata.ego_reflection_visible,
                )
                outcome = self._finalize_episode(run_id, self._new_outcome(run_id, cube, True))
                if self._is_e4_corridor_run():
                    self._print_e4_run_result(run_id, outcome)
                return outcome
            elif is_e4_terminal_move_failure:
                self.bumped_wall = True
                self._record_step(
                    move=move,
                    view_description=view_description,
                    reasoning=reasoning,
                    summary=summary,
                    identification=identification,
                    goal_status=task_payload.get("goal_status") if task_payload else None,
                    bumped_wall=True,
                    inner_reasoning=thinking,
                    view_position=observation_metadata.view_position,
                    view_direction=observation_metadata.view_direction,
                    mirror_visible=observation_metadata.mirror_visible,
                    ego_reflection_visible=observation_metadata.ego_reflection_visible,
                )
                self.step_records[-1]["e4_terminal_violation"] = True
                self.step_records[-1]["e4_terminal_row"] = True
                self.trajectory.append(self.current_pos)
                print("  [E4] Movement at terminal row is invalid; expected 'done'.")
                outcome = self._finalize_episode(run_id, self._new_outcome(run_id, None, False))
                self._print_e4_run_result(run_id, outcome)
                return outcome
            elif move not in ("w", "s", "a", "d"):
                print(f"  [Step {step}] Unsupported action '{move}'. Ending episode.")
                outcome = self._finalize_episode(run_id, self._new_outcome(run_id, None, False))
                if self._is_e4_corridor_run():
                    self._print_e4_run_result(run_id, outcome)
                return outcome

            previous_pos = self.current_pos
            x, z, direction, bumped = self.sim_env.step(move)
            self.bumped_wall = bumped
            self.previous_pos = previous_pos
            self.current_pos = (x, z)
            self.current_dir = direction
            self.trajectory.append(self.current_pos)
            self._record_step(
                move=move,
                view_description=view_description,
                reasoning=reasoning,
                summary=summary,
                identification=identification,
                goal_status=task_payload.get("goal_status") if task_payload else None,
                inner_reasoning=thinking,
                view_position=observation_metadata.view_position,
                view_direction=observation_metadata.view_direction,
                mirror_visible=observation_metadata.mirror_visible,
                ego_reflection_visible=observation_metadata.ego_reflection_visible,
            )

            if self.sim_env.update_frame() is False:
                outcome = self._finalize_episode(run_id, self._new_outcome(run_id, None, False))
                if self._is_e4_corridor_run():
                    self._print_e4_run_result(run_id, outcome)
                return outcome

            if self.iteration >= self.max_steps:
                print(f"  Max steps reached for run {run_id}.")
                outcome = self._finalize_episode(run_id, self._new_outcome(run_id, None, False))
                if self._is_e4_corridor_run():
                    self._print_e4_run_result(run_id, outcome)
                return outcome

    # ------------------------------------------------------------------
    # Experiment: multi-run loop
    # ------------------------------------------------------------------

    def _run_experiment(self) -> List[RunOutcome]:
        spec = self.experiment_spec
        if getattr(self, "paper_experiment_id", None) is not None:
            print(f"Paper {self.paper_experiment_id.upper()} (implementation {self.experiment_id})")
        print(f"Starting experiment {self.experiment_id.upper()} | model={self.llm_model} | client={self.client_type} | runs={self.num_runs}")

        if self.client_type == "ollama":
            try:
                available_models = list_models(self.ollama_host)
                if self.llm_model not in available_models:
                    preview = ", ".join(available_models[:8]) if available_models else "none"
                    print(
                        "Experiment stopped: model not found.\n"
                        f"Requested: {self.llm_model}\n"
                        f"Available (first 8): {preview}"
                    )
                    return []
            except Exception as exc:
                print(f"Warning: could not query Ollama models before start: {exc}")

        outcomes: List[RunOutcome] = []
        for run_id in range(1, self.num_runs + 1):
            run_setup = spec.initialize_run(run_id, self.experiment_runtime_options)
            self._reset_run(run_id, run_setup)
            outcome = self._run_single_episode(run_id)
            outcomes.append(outcome)

        print_metrics(outcomes, self.experiment_id)
        if self.json_export:
            save_experiment_json(
                outcomes, self.experiment_id, self.output_dir,
                run_context=self._build_run_context(),
            )
        return outcomes
