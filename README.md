<div align="center">
  <h1>WaveGen: Truly End-to-End Waveform Generation via Internal Spectral Trajectory Learning</h1>
  <p><strong>NeurIPS 2026</strong></p>

  <p>
    <a href="#installation"><img src="https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg?logo=python&logoColor=white" alt="Python 3.10+"></a>
    <a href="https://huggingface.co/LeeSangHoon/WaveGen"><img src="https://img.shields.io/badge/Hugging%20Face-Models-yellow?logo=huggingface" alt="Hugging Face Models"></a>
    <a href="https://wavegen26.github.io/demo"><img src="https://img.shields.io/badge/Demo-Page-orange.svg" alt="WaveGen Demo Page"></a>
    <a href="https://wavegen26.github.io/demo/spectral_forcing_visualization/"><img src="https://img.shields.io/badge/Spectral%20Forcing-Demo-blue.svg" alt="Spectral Forcing Demo"></a>
  </p>

  <p><strong>Sang-Hoon Lee<sup>1,*</sup> and Ha-Yeong Choi<sup>2,*</sup></strong></p>
  <p>
    <img src="docs/static/images/logos/ajou-university.png" alt="Ajou University" height="40">
    &nbsp;&nbsp;&nbsp;&nbsp;
    <img src="docs/static/images/logos/kt.png" alt="KT" height="40">
  </p>
  <p>
    <sup>1</sup>Department of Artificial Intelligence, Ajou University, Suwon, Korea<br>
    <sup>2</sup>Frontier AI Lab, KT Corp., Seoul, Korea<br>
  </p>
</div>

<div align="center">
  <a href="docs/static/images/wavegen_fig.pdf">
    <img src="docs/static/images/wavegen_fig.png" alt="Figure 1 from the WaveGen paper: Spectral Forcing and disentangled waveform and spectral diffusion heads." width="100%">
  </a>
</div>

- The base model, training, inference, and evaluation code is in [src/wavegen](src/wavegen).
- **WaveGen V-Spec is the recommended model for inference.** Its 1400k EMA Safetensors bundle is in `ckpts/WaveGen_VSpec/`; training, inference, and evaluation are packaged in [src/wavegen_v_spec](src/wavegen_v_spec/README.md).
- **WaveGen-Flash** generates speech in four steps. Its inference code is in [src/wavegen_flash](src/wavegen_flash/README.md), with a self-contained EMA Safetensors bundle in `ckpts/WaveGen_Flash/`.

## Update

### 26.11.10

- We released the WaveGen source code and checkpoint trained on Emilia (24 kHz audio, 50 Hz waveform patches).
- **WaveGen**, **WaveGen (Fine-tuned with v-spec losses)**, and **WaveGen-Flash (4 Steps)**.
 
### 26.09.24

- WaveGen was accepted to NeurIPS 2026.

### 26.04.25

- WaveGen was submitted to NeurIPS 2026.

## Installation

Use Python 3.10 or newer. Install PyTorch and torchaudio for your system, then install this repository:

```bash
cd WaveGen
pip install -e .
```

FFmpeg is required for audio preprocessing. Evaluation dependencies are optional:

```bash
pip install -e ".[eval]"
```

## Models

| Model | Checkpoint | Use |
|---|---|---|
| WaveGen (Pre-trained) | 1300k EMA | Pre-trained model |
| **WaveGen (Fine-tuned with v-spec losses)** | **1400k EMA** | **Recommended; default for CLI, Gradio, and V-Spec batch evaluation** |
| WaveGen-Flash (4 Steps) | 48k EMA | Fixed four-step inference |

The public checkpoints use Safetensors and contain FP32 EMA weights only.
Download the selected model's checkpoint, YAML, and vocabulary from
[Hugging Face](https://huggingface.co/LeeSangHoon/WaveGen), then place its model folder
under `ckpts/` as shown below.

## Files

```text
WaveGen/
├── ckpts/
│   ├── WaveGen/         # base 1300k checkpoint and vocabulary
│   ├── WaveGen_VSpec/   # 1400k EMA checkpoint, configuration, and vocabulary
│   └── WaveGen_Flash/   # four-step EMA checkpoint, configuration, and vocabulary
├── data/               # dataset metadata and evaluation lists
├── docs/static/images/ # model overview (Figure 1)
└── src/
    ├── wavegen/        # base model
    │   ├── configs/
    │   ├── eval/
    │   ├── infer/
    │   ├── model/
    │   ├── scripts/
    │   └── train/
    ├── wavegen_v_spec/ # V-Spec training, inference, and evaluation
    └── wavegen_flash/  # four-step inference only
```

The recommended default is **WaveGen V-Spec**, using the 1,400,000-update EMA checkpoint:

```text
ckpts/WaveGen_VSpec/model_1400000.safetensors
ckpts/WaveGen_VSpec/vocab.txt
ckpts/WaveGen_VSpec/WaveGen_VSpec.yaml
```

Weights stay in the local `ckpts/` directory and are excluded from Git. The
matching vocabulary and model configuration are included with the source.

## Inference

Run from the repository root:

```bash
wavegen_infer-cli \
  --ckpt_file ckpts/WaveGen_VSpec/model_1400000.safetensors \
  --model_cfg ckpts/WaveGen_VSpec/WaveGen_VSpec.yaml \
  --vocab_file ckpts/WaveGen_VSpec/vocab.txt \
  --ref_audio reference.wav \
  --ref_text "The transcript of the reference audio." \
  --gen_text "The text to synthesize."
```

The example configuration uses the same checkpoint:

```bash
wavegen_infer-cli -c src/wavegen_v_spec/infer/examples/basic.toml
```

The defaults are EMA 1400k, Prompt Anchor (`Prompt_xt_Anchor`), Scheduled K3
(base CFG 1.25), 50 steps, Mel temperature 0, and seed 0. Inference dropout is
read from the model YAML (0.2). Prompt Anchor interpolates the prompt waveform
and Mel states from noise to the reference at each ODE step and masks prompt
states in the joint-null CFG branch. To use baseline prompt conditioning:

```bash
wavegen_infer-cli -c src/wavegen_v_spec/infer/examples/basic.toml --inference_method baseline
```

The inference method and CFG schedule are separate options. Select constant
guidance with `--cfg_schedule constant` (CFG 3 by default).

`wavegen_infer-cli` and `wavegen_v_spec_infer-cli` both use the V-Spec package.
Without reinstalling an existing environment, use
`python -m wavegen_v_spec.infer.infer_cli` with `src` on `PYTHONPATH`.
For the pre-trained 1300k Safetensors release, select its assets explicitly:

```bash
python -m wavegen.infer.infer_cli \
  --ckpt_file ckpts/WaveGen/model_1300000.safetensors \
  --model_cfg ckpts/WaveGen/WaveGen_SF_Emilia_24k_50Hz_IBDrop02.yaml \
  --vocab_file ckpts/WaveGen/vocab.txt \
  --cfg_schedule scheduled_k3 \
  --cfg_strength 1.25 \
  --ref_audio reference.wav \
  --ref_text "The transcript of the reference audio." \
  --gen_text "The text to synthesize."
```

The base module retains its original local defaults; the paths above select
the public Safetensors bundle. Existing `.pt` checkpoints remain supported
through an explicit `--ckpt_file` override with matching model assets.

See [inference](src/wavegen_v_spec/infer/README.md) for the available options,
including Hugging Face loading from the shared repository's `WaveGen_VSpec/` folder.

### WaveGen-Flash (4 Steps)

Place `model.safetensors`, `WaveGen_Flash.yaml`, and `vocab.txt` in
`ckpts/WaveGen_Flash/`, then run:

```bash
python -m wavegen_flash.infer.infer_cli \
  --ref-audio reference.wav \
  --ref-text "The transcript of the reference audio." \
  --gen-text "The text to synthesize." \
  --output generated_flash.wav
```

Flash uses **four Euler steps** with the time grid **`[0, 0.1, 0.25, 0.5, 1.0]`**,
CFG 0, and inference bottleneck dropout 0. The matching YAML stores the complete
sampling settings, including waveform and Mel temperatures of 1. The final Mel
output head is skipped by default. The Safetensors file contains the complete
merged EMA model; no separate base checkpoint is needed. This release includes
inference code and weights; the Flash training recipe is not included.

See the [Flash guide](src/wavegen_flash/README.md) for Python, Hugging Face,
and batch inference examples.

### Gradio demo

Run the local demo with the Python environment that already runs WaveGen:

```bash
python app.py
```

Open `http://127.0.0.1:7860`, upload reference speech, enter its transcript and
the text to generate. Advanced settings expose Prompt Anchor/baseline inference and
constant/Scheduled K3 guidance. The demo defaults to Prompt Anchor with Scheduled K3
(base CFG 1.25), the recommended V-Spec 1400k EMA Safetensors checkpoint, and 50 steps.
Inference dropout uses `model.cfm.inference_bottleneck_dropout` from the loaded
model YAML (0.2 in the supplied config).

Select **WaveGen-Flash** in the model selector for the fixed four-step model.
Its sampling schedule, CFG, and dropout follow the Flash configuration.

The app was tested with the existing Gradio **6.17.3** environment. No package
installation or version change is needed in that environment. It loads the
local checkpoint on the first generation request and does not download models.
Use `python app.py --help` for checkpoint, device, and server options.

## Training

Prepare dataset metadata with the scripts in `src/wavegen/train/datasets`, and set the dataset and output paths in `src/wavegen/configs/WaveGen_SF_Emilia_24k_50Hz_IBDrop02.yaml`.

```bash
accelerate config
accelerate launch src/wavegen/train/train.py --config-name WaveGen_SF_Emilia_24k_50Hz_IBDrop02.yaml
```

See [training](src/wavegen/train/README.md) for data preparation and training options.

## Evaluation

The default V-Spec 1400k Safetensors batch inference and evaluation settings are in
`src/wavegen_v_spec/eval/eval_infer_batch.toml`. See the
[evaluation guide](src/wavegen_v_spec/eval/README.md) for dataset setup and metrics.

```bash
python -m wavegen_v_spec.eval.eval_infer_batch \
  --eval-config src/wavegen_v_spec/eval/eval_infer_batch.toml --full-eval
```

Set the dataset and local metric-model paths in the TOML before running evaluation.

For Flash, generate the complete metadata list with
`python -m wavegen_flash.eval.eval_infer_batch --meta path/to/meta.lst --output-dir results/flash`,
then use the existing metric tools described in the
[Flash evaluation notes](src/wavegen_flash/README.md#batch-inference-and-evaluation).

## Safeguards

WaveGen can reproduce voice characteristics from a reference recording. To support
responsible use of the source code and checkpoints, please follow these guidelines:

- **Consent:** Use reference speech only with the speaker's informed consent and
  permission to use the recording. Obtain permission before sharing generated
  speech that reproduces their voice.
- **Misuse:** Do not use WaveGen for deceptive impersonation, fraud, harassment,
  or fabricated statements presented as authentic speech.
- **Transparency:** Clearly identify shared or published outputs as AI-generated
  audio.
- **Privacy:** Protect reference recordings and personal information; do not
  redistribute them without authorization.

These are usage guidelines, not automatic safety filters. Public-facing services
should add appropriate access controls, abuse reporting, and review of suspected
misuse.

## Reference
- PeriodWave [ICLR 2025]
- FlowTokenizer [ACL Findings 2026] 
- ReGen [ICML 2026]
- F5-TTS [ACL 2025]
- WavTTS 

## License

WaveGen source-code contributions are Copyright (c) 2026 Sang-Hoon Lee and
released under the [MIT License](LICENSE). Reused components retain their
original copyright notices and licenses; see
[third-party notices](docs/THIRD_PARTY_NOTICES.md).

The published model checkpoints are licensed separately under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
