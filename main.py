import hydra
from omegaconf import DictConfig

from runtime.controller import Controller


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.get("seed") is not None:
        import random
        import os
        random.seed(cfg.seed)
        os.environ["EXPERIMENT_SEED"] = str(cfg.seed)
        try:
            import numpy as np
            np.random.seed(cfg.seed)
        except ImportError:
            pass

    if cfg.get("msr_level") is not None:
        import os
        os.environ["MSR_LEVEL"] = str(cfg.msr_level)

    if cfg.get("guided_prompt"):
        import os
        os.environ["GUIDED_PROMPT"] = "1"
    else:
        import os
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
        e7_random_cube_heights=cfg.get("e7_random_cube_heights", False),
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
        google_api_key=cfg.google.api_key,
        anthropic_api_key=cfg.anthropic.api_key,
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
