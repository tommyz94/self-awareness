#!/usr/bin/env python3
"""Check a clean installation without calling models or reading run data."""
from __future__ import annotations

import argparse
import importlib
from importlib.metadata import version
from pathlib import Path
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Render a temporary frame for all five paper scenes.")
    parser.add_argument("--providers", action="store_true", help="Check optional API SDK imports; make no requests.")
    args = parser.parse_args()

    from hydra import compose, initialize_config_dir
    from experiments.naming import PAPER_EXPERIMENTS
    from experiments import ExperimentRuntimeOptions, get_experiment

    for package in ("hydra-core", "numpy", "mujoco", "Pillow", "matplotlib", "opencv-python",
                    "pandas", "seaborn", "jupyterlab"):
        print(f"{package}=={version(package)}")

    if args.providers:
        for module in ("openai", "google.genai", "anthropic"):
            importlib.import_module(module)
            print(f"API SDK import OK: {module}")

    with tempfile.TemporaryDirectory(prefix="self-awareness-install-") as temp_dir:
        for paper_id, implementation_id in PAPER_EXPERIMENTS.items():
            with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
                cfg = compose(config_name="config", overrides=[f"experiment=paper/{paper_id}"])
            if cfg.experiment_id != implementation_id or cfg.paper_experiment_id != paper_id:
                raise RuntimeError(f"Invalid paper preset: {paper_id}")
            options = ExperimentRuntimeOptions(
                distractors=cfg.distractors,
                camera_height=cfg.camera_height,
                random_wall_color=cfg.random_wall_color,
                distractor_color_mode=cfg.distractor_color_mode,
                random_cube_heights=cfg.random_cube_heights,
                body_style=cfg.body_style,
            )
            random.seed(123)
            spec = get_experiment(f"paper/{paper_id}")
            setup = spec.initialize_run(1, options)
            print(f"paper/{paper_id} -> {implementation_id}: preset and scene OK")
            if not args.render:
                continue

            from env.backends.mujoco import MuJoCoEnv
            from PIL import Image

            scene = spec.resolve_scene_template(options)
            env = MuJoCoEnv(
                scene_spec=scene,
                backend_overrides=spec.resolve_backend_overrides(options, scene),
                body_style=cfg.body_style,
                headless=True,
                render_width=cfg.mujoco.render_width,
                render_height=cfg.mujoco.render_height,
            )
            try:
                env.reset(setup)
                path = env.capture_image(str(Path(temp_dir) / f"{paper_id}.jpg"))
                with Image.open(path) as frame:
                    if frame.size != (640, 480):
                        raise RuntimeError(f"Unexpected frame size for {paper_id}: {frame.size}")
                print(f"paper/{paper_id}: native 640x480 rendering OK")
            finally:
                env.close()
    print("Installation check passed. No model calls or evaluation data used.")


if __name__ == "__main__":
    main()
