import hydra
from omegaconf import DictConfig

from runtime.controller import Controller


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    controller = Controller(
        nav_mode=cfg.nav,
        client_type=cfg.client,
        print_reasoning=cfg.print_reasoning,
        num_distractors=cfg.distractors,
        max_steps=cfg.max_steps,
        third_person=cfg.third_person,
        camera_height=cfg.camera_height,
        use_mirror=cfg.get("use_mirror", True),
        random_wall_color=cfg.get("random_wall_color", False),
        distractor_color_mode=cfg.get("distractor_color_mode", "allow_duplicates"),
        e7_random_cube_heights=cfg.get("e7_random_cube_heights", False),
        save_dataset=cfg.get("save_dataset", False),
        experiment_id=cfg.experiment_id,
        num_runs=cfg.num_runs,
        history_steps=cfg.history_steps,
        ollama_host=cfg.ollama.host,
        openai_api_key=cfg.openai.api_key,
        google_api_key=cfg.google.api_key,
        anthropic_api_key=cfg.anthropic.api_key,
        output_dir=cfg.outputs.dir,
        log_file=cfg.outputs.log_file,
        json_export=cfg.outputs.json_export,
        heatmap=cfg.outputs.heatmap,
        save_obs=cfg.outputs.save_obs,
        env_type=cfg.env,
        mujoco_headless=cfg.mujoco.headless,
    )
    controller.run()


if __name__ == "__main__":
    main()
