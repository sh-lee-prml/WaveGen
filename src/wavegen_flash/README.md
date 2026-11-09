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
| Waveform / Mel temperature | 1 / 1 |
| Final Mel output head | Skipped by default |
| Default seed | 0 |
| Default duration limit | 30 seconds, including the prompt |

The explicit grid is used directly; no additional time shift or power mapping
is applied. The four network calls occur at 0, 0.1, 0.25, and 0.5. The sampler
anchors the prompt states to their interpolation between the initial noise and
the reference at every call, then applies the explicit Euler update.

The released model fixes its step count, time grid, CFG, and dropout. The CLI
does not expose overrides for those settings. Seed, speed, duration limit, and
the optional final Mel computation can be selected per synthesis request.

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
