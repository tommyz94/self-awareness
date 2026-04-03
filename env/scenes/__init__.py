__all__ = ["build_humanoid_scene", "build_humanoid_wrapper"]


def __getattr__(name):
    if name == "build_humanoid_scene":
        from env.scenes.mujoco_builder import build_humanoid_scene
        return build_humanoid_scene
    if name == "build_humanoid_wrapper":
        from env.scenes.mujoco_builder import build_humanoid_wrapper
        return build_humanoid_wrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
