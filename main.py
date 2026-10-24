import os
import random
import secrets

import hydra
from omegaconf import DictConfig

from runtime.controller import Controller

_MAX_RANDOM_SEED = (2 ** 32) - 1


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    resolved_seed = cfg.get("seed")
    seed_origin = "explicit"
    if resolved_seed is None:
        resolved_seed = secrets.randbelow(_MAX_RANDOM_SEED + 1)
        seed_origin = "generated"
    else:
        resolved_seed = int(resolved_seed)

    random.seed(resolved_seed)
    os.environ["EXPERIMENT_SEED"] = str(resolved_seed)
    print(f"Experiment seed: {resolved_seed} ({seed_origin})")

    try:
        import numpy as np
        np.random.seed(resolved_seed)
    except ImportError:
        pass

    if cfg.get("msr_level") is not None:
        os.environ["MSR_LEVEL"] = str(cfg.msr_level)

    if cfg.get("guided_prompt"):
        os.environ["GUIDED_PROMPT"] = "1"
    else:
        os.environ.pop("GUIDED_PROMPT", None)

    controller = Controller(
        nav_mode=cfg.nav,
        client_type=cfg.client,
        print_reasoning=cfg.print_reasoning,
        num_distractors=cfg.distractors,
        max_steps=cfg.max_steps,
        third_person=cfg.third_person,
        camera_height=cfg.camera_height,
        body_style=cfg.get("body_style", "default"),
        self_perception_mode=cfg.get("self_perception_mode", "color"),
        use_mirror=cfg.get("use_mirror", True),
        random_wall_color=cfg.get("random_wall_color", False),
        distractor_color_mode=cfg.get("distractor_color_mode", "allow_duplicates"),
        random_cube_heights=cfg.get("random_cube_heights", False),
        save_dataset=cfg.get("save_dataset", False),
        experiment_id=cfg.experiment_id,
        paper_experiment_id=cfg.get("paper_experiment_id"),
        num_runs=cfg.num_runs,
        history_steps=cfg.history_steps,
        ollama_host=cfg.ollama.host,
        openai_api_key=cfg.openai.api_key,
        openrouter_api_key=cfg.openrouter.api_key,
        openrouter_base_url=cfg.openrouter.base_url,
        openrouter_http_referer=cfg.openrouter.http_referer,
        openrouter_app_title=cfg.openrouter.app_title,
        openrouter_require_parameters=cfg.openrouter.require_parameters,
        openrouter_min_request_interval_sec=cfg.openrouter.min_request_interval_sec,
        openrouter_allow_fallbacks=cfg.openrouter.allow_fallbacks,
        openrouter_provider_order=cfg.openrouter.provider_order,
        openrouter_provider_only=cfg.openrouter.provider_only,
        openrouter_provider_ignore=cfg.openrouter.provider_ignore,
        openrouter_max_output_tokens=cfg.openrouter.max_output_tokens,
        openrouter_structured_string_max_length=cfg.openrouter.structured_string_max_length,
        google_api_key=cfg.google.api_key,
        google_use_vertexai=cfg.google.use_vertexai,
        google_project=cfg.google.project,
        google_location=cfg.google.location,
        google_max_output_tokens=cfg.google.max_output_tokens,
        google_max_retries=cfg.google.max_retries,
        google_initial_retry_delay_sec=cfg.google.initial_retry_delay_sec,
        google_max_retry_delay_sec=cfg.google.max_retry_delay_sec,
        anthropic_api_key=cfg.anthropic.api_key,
        anthropic_use_vertexai=cfg.anthropic.use_vertexai,
        anthropic_project=cfg.anthropic.project,
        anthropic_location=cfg.anthropic.location,
        anthropic_max_output_tokens=cfg.anthropic.max_output_tokens,
        anthropic_max_retries=cfg.anthropic.max_retries,
        hf_device=cfg.hf.device,
        hf_dtype=cfg.hf.dtype,
        hf_max_output_tokens=cfg.get("hf", {}).get("max_output_tokens", 1024),
        output_dir=cfg.outputs.dir,
        log_file=cfg.outputs.log_file,
        json_export=cfg.outputs.json_export,
        heatmap=cfg.outputs.heatmap,
        save_obs=cfg.outputs.save_obs,
        env_type=cfg.env,
        mujoco_headless=cfg.mujoco.headless,
        mujoco_render_width=cfg.mujoco.render_width,
        mujoco_render_height=cfg.mujoco.render_height,
    )
    controller.run()


if __name__ == "__main__":
    main()
