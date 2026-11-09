from wavegen.model.backbones.dit import DiT
from wavegen.model.cfm import CFM

__all__ = ["CFM", "DiT", "Trainer"]


def __getattr__(name):
    if name == "Trainer":
        from wavegen.model.trainer import Trainer

        return Trainer
    raise AttributeError(name)
