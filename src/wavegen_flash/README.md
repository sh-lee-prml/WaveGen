# WaveGen-Flash

WaveGen-Flash generates 24 kHz speech in **four Euler steps** from text and a
reference recording. The 48k EMA release contains a complete merged FP32
Safetensors model, its configuration, and its vocabulary. Loading requires no
separate base checkpoint.

This package contains inference code only. The Flash training recipe is not
included. It reuses the repository's WaveGen DiT components; install the complete
WaveGen repository to use it.

## Model files

Download the `WaveGen_Flash/` folder from
[LeeSangHoon/WaveGen](https://huggingface.co/LeeSangHoon/WaveGen) and place it under
`ckpts/`:

```text
ckpts/WaveGen_Flash/
├── model.safetensors
├── WaveGen_Flash.yaml
└── vocab.txt
```

The Flash loader accepts `.safetensors` only and loads the complete EMA model.
CUDA BF16 inference uses autocast while keeping model weights in FP32; CPU
inference uses FP32. The checkpoint is licensed under CC BY-NC 4.0. Source-code
licenses and third-party notices remain as documented in the repository root.

## Fixed sampling configuration

`WaveGen_Flash.yaml` records the complete four-step inference configuration:

| Setting | Value |
|---|---|
| Sampling method | Euler |
| Number of steps | 4 |
| Time grid, including final endpoint | `[0, 0.1, 0.25, 0.5, 1.0]` |
| Time shift | 3, already reflected in the grid |
| CFG strength | 0 |
| Inference bottleneck dropout | 0 |
| Waveform / initial Mel temperature | 1 / 0 (default) |
| Final Mel output head | Skipped by default |
| Default seed | 0 |
| Default duration limit | 30 seconds, including the prompt |

The explicit grid is used directly; no additional time shift or power mapping
is applied. The four network calls occur at 0, 0.1, 0.25, and 0.5. The sampler
anchors the prompt states to their interpolation between the initial noise and
the reference at every call, then applies the explicit Euler update.

The released model fixes its step count, time grid, CFG, and dropout. The CLI
does not expose overrides for those settings. Seed, speed, initial Mel
temperature, duration limit, and the optional final Mel computation can be
selected per synthesis request.

### Initial Mel temperature

The supplied YAML sets `mel_temperature: 0.0`. This parameter scales only the
initial Gaussian Mel noise: the initial Mel state is zero at temperature 0,
while waveform noise retains temperature 1. It does not scale the clean
reference Mel or subsequent Mel predictions. Prompt Anchor uses the same
scaled initial noise when interpolating toward the reference at every step.
Both Gaussian draws are still made at temperature 0, preserving random-number
consumption for the same seed.

Use `--mel-temperature 1` with either the single-utterance or batch CLI to
reproduce the previous initialization. The Python equivalent is
`model.synthesize(..., mel_temperature=1.0)`. Omitting the override uses the
YAML value; temperatures must be finite and nonnegative.

In the Gradio demo, selecting **WaveGen-Flash** sets **Mel temperature** to 0.
The slider remains editable from 0 to 1; step count, time grid, CFG, and dropout
remain fixed for Flash.

Skipping the final Mel DDT, output normalization, and projection leaves the
waveform calculation unchanged. The Mel input, information bottleneck, shared
network, and waveform head still run. Pass `--compute-final-mel` or
`compute_final_mel=True` to compute the final Mel head as well.

## Command-line inference

Run from the repository root with `src` on `PYTHONPATH`, using the environment
that already runs WaveGen:

```bash
python -m wavegen_flash.infer.infer_cli \
  --ref-audio reference.wav \
  --ref-text "The exact transcript of the reference audio." \
  --gen-text "The text to synthesize." \
  --output generated_flash.wav
```

The default paths are the three files above. Use `--checkpoint`,
`--model-config`, and `--vocab-file` for another local location. After installing
the updated package, `wavegen_flash_infer-cli` provides the same command.
Automatic transcription is not run; supply the reference transcript.

For Hugging Face loading, select the repository and an explicit revision:

```bash
python -m wavegen_flash.infer.infer_cli \
  --repo-id LeeSangHoon/WaveGen --revision main \
  --ref-audio reference.wav \
  --ref-text "The exact transcript of the reference audio." \
  --gen-text "The text to synthesize." \
  --output generated_flash.wav
```

The loader resolves a branch or tag once and downloads all three assets from
that commit. Use a full commit SHA to pin a release. `--local-files-only`
requires a full SHA and previously cached assets.

## Python API

```python
import soundfile as sf
import torch
from wavegen_flash import WaveGenFlash

torch.set_num_threads(1)
model = WaveGenFlash.from_checkpoint(
    "ckpts/WaveGen_Flash/model.safetensors",
    "ckpts/WaveGen_Flash/vocab.txt",
    config="ckpts/WaveGen_Flash/WaveGen_Flash.yaml",
)
sample_rate, audio = model.synthesize(
    "reference.wav",
    "The exact transcript of the reference audio.",
    "The text to synthesize.",
    seed=0,
)
sf.write("generated_flash.wav", audio, sample_rate, subtype="PCM_16")
```

Replace the loader with
`WaveGenFlash.from_pretrained("LeeSangHoon/WaveGen", revision="main")` to use
Hugging Face. The default Hub filenames are
`WaveGen_Flash/model.safetensors`, `WaveGen_Flash/WaveGen_Flash.yaml`, and
`WaveGen_Flash/vocab.txt`.

The default maximum duration includes the reference and generated suffix.
Longer text should be split into shorter passages. There is no automatic text
splitting; `max_duration_seconds` explicitly changes the duration limit.

## Batch inference and evaluation

Generate every row in a Seed-TTS metadata list:

```bash
python -m wavegen_flash.eval.eval_infer_batch \
  --meta data/seedtts_testset/en/meta.lst \
  --output-dir results/WaveGen_Flash/seedtts_test_en
```

This command performs inference only and writes a generation manifest. It
checks checkpoint, configuration, vocabulary, metadata, source, and sampling
settings before reusing existing outputs. For LibriSpeech-PC, use
`--dataset librispeech` with its metadata and `--librispeech-test-clean-path`.
The manifest records the effective Mel temperature, so results generated with
a different temperature cannot be reused as the current run.

Use the existing WaveGen metric tools to score the generated WAVs. For example,
with a locally available English ASR model:

```bash
python -m wavegen_v_spec.eval.eval_seedtts_testset \
  --meta data/seedtts_testset/en/meta.lst \
  --gen-wav-dir results/WaveGen_Flash/seedtts_test_en \
  --eval-task wer --lang en \
  --asr-model path/to/faster-whisper-large-v3
```

See the [evaluation guide](../wavegen_v_spec/eval/README.md) for the metric-model
requirements. Use `--lang zh` with the matching Chinese metadata and Paraformer
model for CER. No metric assets are bundled with the Flash checkpoint.

### Initial Mel temperature comparison

The following complete Seed-TTS evaluations use the same 48k EMA weights,
four-step schedule, original test prompts, seed 0, FP32 weights, and BF16 CUDA
inference. Only the initial Mel temperature changes. WER/CER and UTMOS are
truncated to two decimal places, and SIM to three decimal places (not rounded).
Error rates are percentages and all metrics are macro-averaged over utterances.

| Test set | Initial Mel temperature | WER / CER (%) ↓ | SIM ↑ | UTMOS ↑ |
|---|---:|---:|---:|---:|
| Seed-en (1,088 utterances) | 1 | 1.62 | 0.724 | 3.47 |
| Seed-en (1,088 utterances) | **0 (default)** | **1.49** | **0.727** | **3.53** |
| Seed-zh (2,020 utterances) | 1 | 1.84 | 0.765 | 2.69 |
| Seed-zh (2,020 utterances) | **0 (default)** | **1.59** | **0.768** | **2.74** |

Initial Mel temperature 0 improves all three metrics over temperature 1 on
both test sets. It is the new inference default; the checkpoint weights and
the final Mel output-head skipping behavior are unchanged.
