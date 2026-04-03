__all__ = [
    "MuJoCoEnv",
]


def __getattr__(name):
    if name == "MuJoCoEnv":
        from env.backends.mujoco import MuJoCoEnv
        return MuJoCoEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
