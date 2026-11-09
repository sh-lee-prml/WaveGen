# Inference

Run from the WaveGen repository root with its `src` directory on `PYTHONPATH`:

```bash
python -m wavegen_v_spec.infer.infer_cli \
  -c src/wavegen_v_spec/infer/examples/basic.toml
```

The recommended default is the V-Spec 1400k EMA Safetensors checkpoint
(`ckpts/WaveGen_VSpec/model_1400000.safetensors`), Prompt Anchor,
Scheduled K3 (base CFG 1.25), 50 steps,
Mel temperature 0, and seed 0. Inference dropout is read from the model YAML.
Use `--help` for prompt,
output, and sampler options. The default loads local model assets. See the
[package README](../README.md#inference) for the Python API and custom prompts.

## Final-step Mel head

By default, the last Euler step skips the Mel DDT blocks and their output
normalization and projection. The Mel input, information bottleneck, and shared
network still run: the waveform update uses the current Mel state, but no later
step consumes the final Mel prediction. This leaves the waveform unchanged.

To compute the final Mel head as well, pass `--compute-final-mel` to the CLI,
set `compute_final_mel = true` in the inference TOML, or call
`model.synthesize(..., compute_final_mel=True)`. `--no-compute-final-mel`
overrides a TOML value back to the default. The Gradio advanced settings
provide the same option as an unchecked **Compute final-step Mel head** box.
Custom integration grids or cubic interpolation conservatively compute the
Mel head at every call. Training always computes both output heads.

## Hugging Face checkpoints

`WaveGen.from_pretrained` and `--repo-id` opt in to downloading model assets.
The model repository is [LeeSangHoon/WaveGen](https://huggingface.co/LeeSangHoon/WaveGen).
Supply its repository ID and an explicit revision; a full commit SHA is
recommended for reproducibility. The default paths match the
V-Spec folder in the shared WaveGen model repository:

```text
WaveGen_VSpec/
├── model_1400000.safetensors
├── WaveGen_VSpec.yaml
└── vocab.txt
```

The YAML and vocabulary must match the checkpoint. The released `.safetensors`
file contains only FP32 EMA tensors and loads with the default `use_ema=True`.
Online selection fails explicitly for this EMA-only bundle. Full `.pt`
checkpoints from your own training runs remain supported and can select Online weights.

```python
from wavegen_v_spec.infer.utils_infer import WaveGen

model = WaveGen.from_pretrained(
    "LeeSangHoon/WaveGen",
    revision="YOUR_40_CHARACTER_COMMIT_SHA",
    use_ema=True,
)
print(model.hub_revision)
```

```bash
python -m wavegen_v_spec.infer.infer_cli \
  --repo-id LeeSangHoon/WaveGen \
  --revision YOUR_40_CHARACTER_COMMIT_SHA \
  --ref-audio reference.wav \
  --ref-text "The transcript of the reference audio." \
  --gen-text "The text to synthesize."
```

An explicit revision is required. A branch/tag is resolved once to a commit;
the checkpoint, YAML and vocabulary are all downloaded from that same commit.
The CLI prints the resolved commit. `--local-files-only` requires the full
commit SHA and already cached assets, and performs no network requests.
`--cache-dir` selects another Hugging Face cache directory. The Hub library's
standard authentication settings apply to private or gated repositories.

For other repository layouts, use `--checkpoint-filename`, `--config-filename`,
and `--vocab-filename`; Python uses the corresponding underscore names.
For an older repository with files at its root, explicitly supply all three
filenames, including the `.pt` checkpoint filename when applicable.
Local `--ckpt-file`, `--model-cfg`, and `--vocab-file` overrides cannot be mixed
with `--repo-id`. A custom Hub TOML should contain `repo_id`, `revision`, and
prompt/sampler settings without local checkpoint/config/vocabulary settings.
The bundled example's local model defaults are automatically replaced when
`--repo-id` is supplied. Sampling and EMA/Online selection remain unchanged.

For an existing local `.pt` checkpoint, pass `--ckpt_file path/to/model.pt`
with its matching `--model_cfg` and `--vocab_file`. There is no automatic
fallback between `.safetensors` and `.pt` files.
