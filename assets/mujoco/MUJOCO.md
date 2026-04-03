# `assets/mujoco/`

MuJoCo XML assets used by the MuJoCo backend and scene builder.

## Files

### `humanoid.xml`

Reference humanoid body used to render the agent and mirrored agent.

### `cube.xml`

Reusable cube body template instantiated with different names, positions, and
colors by the MuJoCo environment code.

## Notes

- These files are static assets, not generated output.
- The MuJoCo builder and backend load them from this folder at runtime.
