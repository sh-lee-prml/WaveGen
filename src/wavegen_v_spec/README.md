# WaveGen V-Spec

**Recommended model for inference: 1400k EMA, in Safetensors format.**

This package provides WaveGen training with waveform velocity MSE, spectral
forcing (weight 0.1), and the multi-scale V-Spec objective (weight 0.1), together
with inference and evaluation tools. All generator parameters are trainable.

```text
src/wavegen_v_spec/
├── configs/WaveGen_VSpec.yaml
├── model/                     # generator and spectral objective
├── train/                     # training, checkpoint I/O, and delayed EMA
├── infer/                     # Python API, CLI, and Hugging Face loading
└── eval/                      # batch generation and WER/CER, SIM, UTMOS
```

Use the existing WaveGen environment. No additional training dependency is
needed. From the repository root, set `PYTHONPATH=src`, or use the installed
`wavegen_v_spec_infer-cli` and `wavegen_v_spec_train` entry points.

## Inference

The public inference bundle contains 1400k FP32 EMA weights in Safetensors format, a model YAML, and
the matching vocabulary in `ckpts/WaveGen_VSpec/`. It contains no optimizer or
training state and cannot be used to resume training.

```bash
python -m wavegen_v_spec.infer.infer_cli \
  --ref_audio reference.wav \
  --ref_text "The reference transcript." \
  --gen_text "The text to synthesize."
```

Defaults are EMA 1400k, Prompt Anchor (`Prompt_xt_Anchor`), Scheduled K3 with
base CFG 1.25, 50 steps, Mel temperature 0, and seed 0. Inference dropout is
read from the YAML (0.2). Model weights remain FP32; CUDA inference uses BF16
autocast. CPU inference uses FP32.

```python
from wavegen_v_spec.infer import WaveGen

model = WaveGen.from_checkpoint(
    "ckpts/WaveGen_VSpec/model_1400000.safetensors",
    "ckpts/WaveGen_VSpec/vocab.txt",
    config="ckpts/WaveGen_VSpec/WaveGen_VSpec.yaml",
)
sample_rate, waveform = model.synthesize(
    "reference.wav", "The reference transcript.", "The target text."
)
```

For remote loading, use `WaveGen.from_pretrained(repo_id, revision=...)`.
Its defaults select the Safetensors checkpoint, YAML, and vocabulary from
the shared Hub repository's `WaveGen_VSpec/` subfolder. Existing local `.pt`
checkpoints remain supported through explicit paths.
See the [inference guide](infer/README.md) for local and Hugging Face options.

## Training

The supplied recipe initializes the full generator from the original 1300k
EMA, then trains all parameters with fresh AdamW. It retains both waveform
and Mel/SF branches. The V-Spec term uses the unscaled waveform prediction
error across seven STFT resolutions; it is not an STFT of velocity vectors.

Training uses eight BF16 processes, a 2000-update warmup from 2e-7 to 2e-6,
and a constant learning rate thereafter. EMA starts with an exact Online copy
after update 2000, followed by native EMA averaging with decay 0.9999 and
update interval 10. The budget is three complete prepared-loader epochs.

Place the original full 1300k checkpoint at `ckpts/WaveGen/model_1300000.pt`
and prepared Emilia data in `data/Emilia_ZH_EN_pinyin/`. Then run:

```bash
accelerate launch --num_processes 8 --mixed_precision bf16 \
  -m wavegen_v_spec.train.train
```

Training saves full resumable checkpoints every 10000 updates to
`ckpts/WaveGen_VSpec_Train/`, separate from the inference bundle. Resume this
recipe with `ckpts.resume_ckpt=/path/to/full_training_checkpoint.pt`.
See the [training guide](train/README.md) for details.

## Evaluation

```bash
python -m wavegen_v_spec.eval.eval_infer_batch \
  --eval-config src/wavegen_v_spec/eval/eval_infer_batch.toml --full-eval
```

Set the metadata and local metric assets in the TOML. `--infer-only` runs
generation without scoring. See the [evaluation guide](eval/README.md).
Historical scores from other checkpoints are not results of this release.

See [third-party notices](THIRD_PARTY_NOTICES.md) for source attributions.
