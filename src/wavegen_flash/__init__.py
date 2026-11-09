"""WaveGen-Flash: four-step waveform inference."""

__all__ = ["WaveGenFlash"]


def __getattr__(name):
    if name == "WaveGenFlash":
        from .infer.utils_infer import WaveGenFlash
        return WaveGenFlash
    raise AttributeError(name)
