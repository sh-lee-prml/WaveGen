"""WaveGen waveform generation."""

__version__ = "0.1.0"

INFERENCE_METHOD_CHOICES = ("Prompt_xt_Anchor", "baseline", "mismatch")


def normalize_inference_method(value: str) -> str:
    """Resolve the former public name while recording the current method name."""
    if value == "mismatch":
        return "Prompt_xt_Anchor"
    if value in ("Prompt_xt_Anchor", "baseline"):
        return value
    raise ValueError("inference_method must be Prompt_xt_Anchor or baseline")
