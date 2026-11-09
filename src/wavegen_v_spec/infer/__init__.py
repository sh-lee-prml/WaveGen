"""Local, explicit-asset WaveGen inference."""

__all__ = ["WaveGen"]


def __getattr__(name):
    if name == "WaveGen":
        from .utils_infer import WaveGen

        return WaveGen
    raise AttributeError(name)
