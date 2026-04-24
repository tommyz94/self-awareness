# `assets/mujoco/objs/`

Optional avatar meshes used by experiment `e8`.

## Usage

- Drop one or more mesh files in this folder: `.obj`, `.glb`, `.gltf`, or `.fbx`.
- `e8` samples avatar meshes from this directory at runtime.
- With a single file, all agents will use the same avatar mesh.

## Notes

- Files are loaded as visual meshes only (no collision).
- Non-OBJ files (`.glb/.gltf/.fbx`) are converted to OBJ through `assimp` and cached under
  `assets/mujoco/objs/.cache_obj/`.
- MuJoCo receives absolute paths for mesh assets at scene-build time.
- If sidecar textures are `.jpg/.jpeg`, they are converted to PNG and cached under
  `assets/mujoco/objs/.cache_png/`.
- If the folder is empty, `e8` falls back to the default articulated humanoid visuals.
