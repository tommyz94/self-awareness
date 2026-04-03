# `assets/`

Static non-Python assets used by the simulator.

The current canonical layout is backend-oriented:

- `assets/mujoco/` contains the MuJoCo XML assets used by the MuJoCo backend

## Contents

### `mujoco/`

MuJoCo asset files loaded by the scene builder and backend:

- `humanoid.xml` is the reference humanoid body used to render the agent and
  mirrored agent
- `cube.xml` is the reusable cube body template used for the colored cubes

## Notes

- `assets/` is a canonical folder, not a wrapper.
- See `assets/mujoco/MUJOCO.md` for the file-level overview of the MuJoCo
  assets.
