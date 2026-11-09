"""WaveGen generator with the V-Spec training objective."""
from .cfm import CFM, CFMMSVSpec
from .backbones.dit import DiT

__all__ = ["CFM", "CFMMSVSpec", "DiT"]
