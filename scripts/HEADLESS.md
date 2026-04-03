# Headless Operation

The simulator supports true headless execution through the MuJoCo backend.

## Fully headless MuJoCo

```sh
pip install mujoco pillow
python main.py env=mujoco nav=gemma3:4b
python main.py env=mujoco +experiment=e1 nav=gemma3:4b
```

With `mujoco.headless=true` (the default), MuJoCo renders offscreen through
`mujoco.Renderer` and writes JPG observations directly. No browser, virtual
display, or GUI window is required.

On Linux, the backend defaults to `MUJOCO_GL=egl` before importing MuJoCo.
If your node does not expose a usable EGL path, override it with:

```sh
export MUJOCO_GL=osmesa
python main.py env=mujoco nav=gemma3:4b
```

Guideline:

- `MUJOCO_GL=egl` for headless GPU nodes
- `MUJOCO_GL=osmesa` for CPU-only nodes

## MuJoCo with a live window

```sh
pip install mujoco pillow opencv-python
python main.py env=mujoco nav=gemma3:4b mujoco.headless=false
python main.py env=mujoco nav=human mujoco.headless=false
```

When `mujoco.headless=false`, MuJoCo opens a live OpenCV RGB window:

- LLM mode: preview of the same camera feed used for captures
- human mode: keyboard-controlled preview window

## Human nav controls in MuJoCo

For `env=mujoco nav=human mujoco.headless=false`:

| Key | Action |
|---|---|
| `w` / `s` | Move forward / backward |
| `a` / `d` | Turn left / right |
| `c` | Toggle first-person / third-person camera |
| `q` / `Esc` / window close | Exit |
