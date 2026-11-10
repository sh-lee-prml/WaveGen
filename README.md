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
    <sup>1</sup>Department of Artificial Intelligence, Ajou University, Suwon, Korea<br>
    <sup>2</sup>Frontier AI Lab, KT Corp., Seoul, Korea<br>
  </p>
</div>

<div align="center">
  <a href="docs/static/images/wavegen_fig.pdf">
    <img src="docs/static/images/wavegen_fig.png" alt="Figure 1 from the WaveGen paper: Spectral Forcing and disentangled waveform and spectral diffusion heads." width="100%">
  </a>
</div>

<p><strong>WaveGen is the first end-to-end waveform text-to-speech model achieving state-of-the-art performance on the Emilia training dataset</strong></p>


- The base model, training, inference, and evaluation code is in [src/wavegen](src/wavegen).
- **WaveGen V-Spec is the recommended model for inference.** Its 1400k EMA Safetensors bundle is in `ckpts/WaveGen_VSpec/`; training, inference, and evaluation are packaged in [src/wavegen_v_spec](src/wavegen_v_spec/README.md).
- **WaveGen-Flash** generates speech in four steps. Its inference code is in [src/wavegen_flash](src/wavegen_flash/README.md), with a self-contained EMA Safetensors bundle in `ckpts/WaveGen_Flash/`.

## GPU Sponsorship

We welcome GPU resources to support the continued development of WaveGen. If you are interested in supporting this work, please contact Sang-Hoon Lee (sanghoonlee@ajou.ac.kr).

## Update

### 26.11.10

- We released the WaveGen source code and checkpoint (**WaveGen**, **WaveGen-VSpec**) trained on Emilia (**24 kHz audio**, **50 Hz waveform patches**)

- **WaveGen-Flash (4 Steps)** is a four-step model distilled from the WaveGen (Vspec). We will share the distillation details soon
   
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
| WaveGen | 1300k EMA | Pre-trained model |
| **WaveGen (Fine-tuned with v-spec losses)** | **1400k EMA** | **Recommended; default for CLI, Gradio, and V-Spec batch evaluation** |
| WaveGen-Flash (4 Steps) | 48k EMA | Fixed four-step inference |

The public checkpoints use Safetensors and contain FP32 EMA weights only.
Download the selected model's checkpoint, YAML, and vocabulary from
[Hugging Face](https://huggingface.co/LeeSangHoon/WaveGen), then place its model folder
under `ckpts/` as shown below.

## Seed-TTS evaluation

Results on the complete Seed-en (1,088 utterances) and Seed-zh (2,020 utterances)
test sets. WaveGen and V-Spec use 50 Euler steps; WaveGen-Flash uses four Euler
steps. All three models use Mel temperature 0.

<table>
  <colgroup span="1"></colgroup>
  <colgroup span="3"></colgroup>
  <colgroup span="3"></colgroup>
  <thead>
    <tr>
      <th rowspan="2" scope="col" align="left">Model</th>
      <th colspan="3" scope="colgroup" align="center">EN</th>
      <th colspan="3" scope="colgroup" align="center">ZH</th>
    </tr>
    <tr>
      <th scope="col" align="center">WER (%) ↓</th>
      <th scope="col" align="center">SIM ↑</th>
      <th scope="col" align="center">UTMOS ↑</th>
      <th scope="col" align="center">CER (%) ↓</th>
      <th scope="col" align="center">SIM ↑</th>
      <th scope="col" align="center">UTMOS ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row" align="left"><strong>WaveGen</strong><br><sub>Pre-trained · 1300k EMA</sub></th>
      <td align="center">1.47</td>
      <td align="center">0.755</td>
      <td align="center">3.68</td>
      <td align="center">1.75</td>
      <td align="center">0.805</td>
      <td align="center">2.87</td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>WaveGen V-Spec</strong><br><sub>1400k EMA · Recommended</sub></th>
      <td align="center">1.39</td>
      <td align="center">0.743</td>
      <td align="center">3.59</td>
      <td align="center">1.65</td>
      <td align="center">0.783</td>
      <td align="center">2.79</td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>WaveGen-Flash</strong><br><sub>48k EMA · 4 steps</sub></th>
      <td align="center">1.49</td>
      <td align="center">0.727</td>
      <td align="center">3.53</td>
      <td align="center">1.59</td>
      <td align="center">0.768</td>
      <td align="center">2.74</td>
    </tr>
  </tbody>
</table>

**WaveGen achieves state-of-the-art performance among TTS models trained on the Emilia dataset.**

WER/CER and UTMOS are displayed to two decimal places, and SIM to three, with
excess digits truncated rather than rounded. See the
[Hugging Face model card](https://huggingface.co/LeeSangHoon/WaveGen#seed-tts-evaluation)
for the evaluation settings.

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
sampling settings, with waveform temperature 1 and **initial Mel temperature 0**.
Mel temperature scales only the initial Mel noise; reference conditioning and
later Mel predictions remain active. Use `--mel-temperature 1` to reproduce the
previous initialization. The final Mel output head is skipped by default.
The Safetensors file contains the complete
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
### Our Previous Work
- PeriodWave [ICLR 2025]: The first CFM-based waveform generation model
- FlowTokenizer [ACL Findings 2026]: The first CFM-based neural audio codec 
- ReGen [ICML 2026]: The first CFM-based audio VAE
### Baseline Models
- F5-TTS [ACL 2025]: A zero-shot TTS model
- WavTTS: Concurrent work from ByteDance. We used the official WavTTS implementation as a baseline during the rebuttal period.

## License

WaveGen source-code contributions are Copyright (c) 2026 Sang-Hoon Lee and
released under the [MIT License](LICENSE). Reused components retain their
original copyright notices and licenses; see
[third-party notices](docs/THIRD_PARTY_NOTICES.md).

The published model checkpoints are licensed separately under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Acknowledgements

This work was supported by Institute of Information \& communications Technology Planning \& Evaluation (IITP) grant funded by the Korea government (MSIT) (RS-2025-02283048, Developing the Next-Generation General AI with Reliability, Ethics, and Adaptability, IITP-2026-RS-2023-00255968, the Artificial Intelligence Convergence Innovation Human Resources Development, RS-2026-25617480,  AI Star Fellowship Support
Program(Gyeongsang National University)), National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2026-25518808), the “Advanced GPU Utilization Support Program” funded by the Government of the Republic of Korea (Ministry of Science and ICT), and Artificial intelligence industrial convergence cluster development project funded by the Ministry of Science and ICT (MSIT, Korea)\&Gwangju Metropolitan City.

- Advanced GPU Utilization Support Program (WaveGen-S, WaveGen-B)
- Artificial intelligence industrial convergence cluster development project (WaveGen, WaveGen (V-Spec), WaveGen-Flash)
