"""The demo forwards Flash's initial Mel temperature without changing its schedule."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("gradio")

spec = importlib.util.spec_from_file_location("wavegen_demo", Path(__file__).resolve().parents[1] / "app.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


@pytest.mark.parametrize("temperature", [None, 0.5, 1.0])
def test_flash_demo_forwards_initial_mel_temperature(tmp_path, temperature):
    prompt = tmp_path / "reference.wav"
    sf.write(prompt, np.sin(np.arange(2400) * 0.05).astype(np.float32) * 0.12, 24000)
    calls = []

    class Engine:
        def synthesize(self, **options):
            calls.append(options)
            return 24000, np.zeros(480, dtype=np.float32)

    synth = demo.Synthesizer(demo.build_parser().parse_args([]),
                             engine_factory=lambda **_: Engine(), output_dir=tmp_path / "generated")
    overrides = {} if temperature is None else {"mel_temperature": temperature}
    output, status = synth(str(prompt), "Reference.", "Target.", model_variant="Flash", **overrides)
    assert Path(output).is_file()
    assert calls == [dict(ref_audio=str(prompt), ref_text="Reference.", text="Target.",
                          seed=0, speed=1.0, compute_final_mel=False,
                          mel_temperature=0.0 if temperature is None else temperature)]
    assert "4 steps" in status and "Mel temp" in status


def test_flash_demo_keeps_schedule_fixed_and_temperature_editable():
    steps, schedule, cfg, temperature, method = demo.model_defaults("Flash")
    assert (steps["value"], schedule["value"], cfg["value"]) == (4, "constant", 0.0)
    assert all(not control["interactive"] for control in (steps, schedule, cfg, method))
    assert temperature["value"] == 0.0 and temperature["interactive"] is True
