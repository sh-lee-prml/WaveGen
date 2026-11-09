# Checkpoints

**WaveGen V-Spec (Fine-tuned with v-spec losses), 1400k EMA, is the recommended
model for inference.** It is the default for the CLI, Gradio app, and V-Spec
batch evaluation.

Download the checkpoint, YAML, and vocabulary together from
[LeeSangHoon/WaveGen on Hugging Face](https://huggingface.co/LeeSangHoon/WaveGen).
The repository has `WaveGen/`, `WaveGen_VSpec/`, and `WaveGen_Flash/` at its root; place the selected
folder under this local `ckpts/` directory:

```text
ckpts/
├── WaveGen/
│   ├── model_1300000.safetensors
│   ├── WaveGen_SF_Emilia_24k_50Hz_IBDrop02.yaml
│   └── vocab.txt
├── WaveGen_VSpec/
│   ├── model_1400000.safetensors
│   ├── WaveGen_VSpec.yaml
│   └── vocab.txt
└── WaveGen_Flash/
    ├── model.safetensors
    ├── WaveGen_Flash.yaml
    └── vocab.txt
```

All public Safetensors files contain FP32 EMA weights only, with no optimizer
or training state. Use EMA loading; these files cannot resume a training run.

For the pre-trained 1300k model, use `python -m wavegen.infer.infer_cli` with
`--ckpt_file ckpts/WaveGen/model_1300000.safetensors`, its matching `--model_cfg`,
and `--vocab_file`. See the [root inference examples](../README.md#inference).

The base and V-Spec loaders still support existing `.pt` checkpoints when selected explicitly.
The original full 1300k `.pt` file remains the initialization source for the
V-Spec training recipe; training checkpoints and inference releases are separate.

**WaveGen-Flash (48k EMA)** uses a complete merged `model.safetensors` file;
loading it requires only its YAML and vocabulary, with no additional base model.
Use `python -m wavegen_flash.infer.infer_cli` for its fixed four-step sampler.
The Flash loader accepts Safetensors only. Its configuration specifies Euler
steps at `[0, 0.1, 0.25, 0.5]`, ending at 1.0, CFG 0, and IB dropout 0.
See the [Flash guide](../src/wavegen_flash/README.md).

For the default CLI, use `--ckpt_file`, `--vocab_file`, and `--model_cfg` to
select another checkpoint and its matching assets. The Gradio equivalents
are `--checkpoint`, `--vocab-file`, and `--model-config`.
Checkpoints are local files and are excluded from Git and Docker images.

The published model checkpoints are licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), separately
from the source-code licenses.
