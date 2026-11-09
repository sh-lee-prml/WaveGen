# Inference

Run from the repository root after `pip install -e .`:

```bash
wavegen_infer-cli \
  --ref_audio reference.wav \
  --ref_text "The transcript of the reference recording." \
  --gen_text "The text to synthesize."
```

The default configuration is `infer/examples/basic.toml`. It loads
`ckpts/WaveGen/model_1300000.pt` and `ckpts/WaveGen/vocab.txt`.
The model preset is `configs/WaveGen_SF_Emilia_24k_50Hz_IBDrop02.yaml`: 24 kHz, 50 Hz waveform frames,
spectral forcing, and bottleneck dropout 0.2.

```bash
wavegen_infer-cli -c src/wavegen/infer/examples/basic.toml
python -m wavegen.infer.infer_cli --help
```

CLI arguments override the TOML configuration. Set `--ckpt_file`, `--vocab_file`
and `--model_cfg` together to use another compatible checkpoint. The reference
transcript is supplied by the user. The CLI does not run ASR.

By default, generation uses EMA weights, 50 Euler steps, CFG 3, seed 0,
waveform temperature 1, mel temperature 0, and power-2 / shift-3 timesteps.
Output is saved to `tests/wavegen.wav`; use `--output_dir` and `--output_file`
to change it, or `--overwrite` to replace an existing output.

The default inference method is Prompt Anchor (`Prompt_xt_Anchor`): the prompt waveform and Mel states
follow the noise-to-reference interpolation at each ODE step, and the joint-null
CFG branch masks those prompt states. Select the original baseline method with:

```bash
wavegen_infer-cli -c src/wavegen/infer/examples/basic.toml --inference_method baseline
```

Set `inference_method = "baseline"` in TOML to save that choice. The Python API
also accepts `model.synthesize(..., inference_method="baseline")`.
This method selection is independent of the CFG schedule.
Use `inference_method = "Prompt_xt_Anchor"` in TOML or
`--inference_method Prompt_xt_Anchor` to select Prompt Anchor explicitly.
Existing configurations using `mismatch` remain accepted as a compatibility alias.

For Scheduled K3, add `--cfg_schedule scheduled_k3`. Unless overridden,
its base CFG is 1.25, with the registered early/late 2.5/1.25 strengths and
three text-only null calls. Constant CFG defaults to 3.

Use `--device cpu --dtype fp32` for CPU execution. CUDA uses BF16 autocast
with FP32 weights. `--nfe_step`, `--cfg_strength`, `--seed`, `--speed`,
`--mel_temperature` and `--inference_dropout` control sampling.

```python
from wavegen.infer.utils_infer import WaveGen
import soundfile as sf

model = WaveGen.from_checkpoint(
    "ckpts/WaveGen/model_1300000.pt", "ckpts/WaveGen/vocab.txt"
)
sample_rate, audio = model.synthesize(
    "reference.wav", "The reference transcript.", "The text to synthesize."
)
sf.write("generated.wav", audio, sample_rate)
```
