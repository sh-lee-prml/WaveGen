"""Local Gradio demo: python app.py (uses the existing Python environment)."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

import gradio as gr
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from wavegen_v_spec import normalize_inference_method


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "ckpts/WaveGen_VSpec/model_1400000.safetensors")
    parser.add_argument("--vocab-file", type=Path, default=ROOT / "ckpts/WaveGen_VSpec/vocab.txt")
    parser.add_argument("--model-config", type=Path,
                        default=ROOT / "ckpts/WaveGen_VSpec/WaveGen_VSpec.yaml")
    parser.add_argument("--flash-checkpoint", type=Path, default=ROOT / "ckpts/WaveGen_Flash/model.safetensors")
    parser.add_argument("--flash-vocab-file", type=Path, default=ROOT / "ckpts/WaveGen_Flash/vocab.txt")
    parser.add_argument("--flash-model-config", type=Path, default=ROOT / "ckpts/WaveGen_Flash/WaveGen_Flash.yaml")
    parser.add_argument("--device", default=None, help="Default: CUDA when available, otherwise CPU.")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--server-name", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--server-port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
    return parser


def _number(value, name, minimum, maximum, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise gr.Error(f"{name} must be a finite number.")
    if not minimum <= value <= maximum or (integer and value != int(value)):
        kind = "an integer" if integer else "a number"
        raise gr.Error(f"{name} must be {kind} between {minimum} and {maximum}.")
    return int(value) if integer else float(value)


class Synthesizer:
    """Keep one lazily loaded model; serialize loading, synthesis and WAV export."""

    def __init__(self, args, *, engine_factory=None, output_dir=None):
        self.args = args
        self.engine_factory = engine_factory
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.engine = None
        self.engine_variant = None
        self.lock = threading.Lock()

    def __call__(self, ref_audio, ref_text, text, inference_method="Prompt_xt_Anchor",
                 cfg_schedule="scheduled_k3", cfg_strength=1.25, steps=50, seed=0,
                 speed=1.0, mel_temperature=0.0, compute_final_mel=False, model_variant="V-Spec"):
        if model_variant not in ("V-Spec", "Flash"):
            raise gr.Error("Select WaveGen V-Spec or WaveGen-Flash.")
        if model_variant == "Flash":
            inference_method, cfg_schedule = "Prompt_xt_Anchor", "constant"
            cfg_strength, steps, mel_temperature = 0.0, 4, 1.0
        if not ref_audio or not Path(ref_audio).is_file():
            raise gr.Error("Upload or record reference speech first.")
        if not isinstance(ref_text, str) or not ref_text.strip():
            raise gr.Error("Enter the exact transcript of the reference speech.")
        if not isinstance(text, str) or not text.strip():
            raise gr.Error("Enter the text to generate.")
        if len(ref_text) > 2000 or len(text) > 2000:
            raise gr.Error("Keep each text field within 2,000 characters.")
        try:
            inference_method = normalize_inference_method(inference_method)
        except ValueError as exc:
            raise gr.Error("Select Prompt Anchor or Baseline inference.") from exc
        if cfg_schedule not in ("constant", "scheduled_k3"):
            raise gr.Error("Select constant or scheduled_k3 guidance.")
        steps = _number(steps, "Steps", 1, 100, integer=True)
        seed = _number(seed, "Seed", 0, 2**32 - 1, integer=True)
        cfg_strength = _number(cfg_strength, "CFG strength", 0, 10)
        speed = _number(speed, "Speed", 0.5, 2)
        mel_temperature = _number(mel_temperature, "Mel temperature", 0, 1)
        if not isinstance(compute_final_mel, bool):
            raise gr.Error("Compute final-step Mel head must be a boolean.")
        if cfg_schedule == "scheduled_k3" and (steps < 3 or cfg_strength < 1e-5):
            raise gr.Error("K3 requires at least 3 steps and positive CFG strength.")
        try:
            info = sf.info(ref_audio)
            if not 0 < info.duration <= 30:
                raise gr.Error("Reference speech must be nonempty and at most 30 seconds long.")
            reference, _ = sf.read(ref_audio, dtype="float32", always_2d=True)
            mono = reference.mean(axis=1)
            if not np.isfinite(mono).all() or np.sqrt(np.mean(mono**2)) <= 1e-8:
                raise gr.Error("Reference audio must contain audible, finite speech.")
        except (RuntimeError, ValueError, OSError) as exc:
            raise gr.Error(f"Could not read the reference audio: {exc}") from exc

        started = time.perf_counter()
        with self.lock:
            try:
                if self.engine is None or self.engine_variant != model_variant:
                    import torch

                    torch.set_num_threads(1)
                    # Keep one model resident when switching between variants.
                    self.engine = None
                    self.engine_variant = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    factory = self.engine_factory
                    if model_variant == "Flash":
                        if factory is None:
                            from wavegen_flash import WaveGenFlash
                            factory = WaveGenFlash.from_checkpoint
                        self.engine = factory(
                            checkpoint=self.args.flash_checkpoint, vocab_file=self.args.flash_vocab_file,
                            config=self.args.flash_model_config, device=self.args.device, dtype=self.args.dtype)
                    else:
                        if factory is None:
                            from wavegen_v_spec.infer.utils_infer import WaveGen
                            factory = WaveGen.from_checkpoint
                        self.engine = factory(
                            checkpoint=self.args.checkpoint, vocab_file=self.args.vocab_file,
                            config=self.args.model_config, device=self.args.device,
                            dtype=self.args.dtype, use_ema=True)
                    self.engine_variant = model_variant
                common = dict(ref_audio=ref_audio, ref_text=ref_text, text=text,
                              seed=seed, speed=speed, compute_final_mel=compute_final_mel)
                if model_variant == "Flash":
                    sample_rate, waveform = self.engine.synthesize(**common)
                else:
                    sample_rate, waveform = self.engine.synthesize(
                        **common, inference_method=inference_method, cfg_schedule=cfg_schedule,
                        cfg_strength=cfg_strength, steps=steps, mel_temperature=mel_temperature,
                        inference_dropout=self.engine.model.inference_bottleneck_dropout)
                if self.output_dir is None:
                    raise RuntimeError("The audio output cache has not been configured.")
                self.output_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(prefix="wavegen-", suffix=".wav",
                                                 dir=self.output_dir, delete=False) as stream:
                    output_path = Path(stream.name)
                try:
                    # Returning a WAV path avoids Gradio's numpy-audio peak normalization.
                    sf.write(output_path, waveform, sample_rate, subtype="PCM_16")
                except Exception:
                    output_path.unlink(missing_ok=True)
                    raise
            except (ValueError, RuntimeError, OSError) as exc:
                raise gr.Error(f"Generation failed: {exc}") from exc
        elapsed = time.perf_counter() - started
        if model_variant == "Flash":
            return str(output_path), f"WaveGen-Flash · 4 steps · seed {seed} · {elapsed:.1f}s"
        method_label = "Prompt Anchor" if inference_method == "Prompt_xt_Anchor" else "Baseline"
        return str(output_path), f"{method_label} · {cfg_schedule} · CFG {cfg_strength:g} · seed {seed} · {elapsed:.1f}s"


def guidance_default(schedule, model_variant="V-Spec"):
    if model_variant == "Flash":
        return 0.0
    return 1.25 if schedule == "scheduled_k3" else 3.0


def model_defaults(model_variant):
    flash = model_variant == "Flash"
    return (
        gr.update(value=4 if flash else 50, interactive=not flash),
        gr.update(value="constant" if flash else "scheduled_k3", interactive=not flash),
        gr.update(value=0.0 if flash else 1.25, interactive=not flash),
        gr.update(value=1.0 if flash else 0.0, interactive=not flash),
        gr.update(value="Prompt_xt_Anchor", interactive=not flash),
    )


def create_app(args=None, synthesizer=None):
    args = args if args is not None else build_parser().parse_args([])
    synthesizer = synthesizer if synthesizer is not None else Synthesizer(args)
    with gr.Blocks(title="WaveGen", analytics_enabled=False, delete_cache=(3600, 86400)) as demo:
        gr.Markdown("# WaveGen\nGenerate speech from a reference voice and your text. "
                    "Enter the reference transcript exactly as spoken.")
        model_variant = gr.Dropdown(
            [("WaveGen V-Spec (Recommended)", "V-Spec"), ("WaveGen-Flash (4 Steps)", "Flash")],
            value="V-Spec", label="Model")
        with gr.Row():
            with gr.Column():
                ref_audio = gr.Audio(label="Reference speech (up to 30 seconds)",
                                     sources=["upload", "microphone"], type="filepath")
                ref_text = gr.Textbox(label="Reference transcript (required)", lines=3)
                text = gr.Textbox(label="Text to generate", lines=4, max_lines=10)
            with gr.Column():
                output = gr.Audio(label="Generated speech", type="filepath", format="wav",
                                  interactive=False, buttons=["download"])
                status = gr.Textbox(label="Generation details", interactive=False)
        with gr.Accordion("Advanced settings", open=False):
            with gr.Row():
                inference_method = gr.Dropdown(
                    [("Prompt Anchor", "Prompt_xt_Anchor"), ("Baseline", "baseline")],
                    value="Prompt_xt_Anchor", label="Inference method")
                cfg_schedule = gr.Dropdown(["constant", "scheduled_k3"], value="scheduled_k3",
                                           label="CFG schedule")
                cfg_strength = gr.Slider(0, 10, value=1.25, step=0.05, label="CFG strength")
            with gr.Row():
                steps = gr.Slider(1, 100, value=50, step=1, label="Steps")
                seed = gr.Number(value=0, precision=0, minimum=0, maximum=2**32 - 1, label="Seed")
                speed = gr.Slider(0.5, 2, value=1.0, step=0.05, label="Speed")
            with gr.Row():
                mel_temperature = gr.Slider(0, 1, value=0, step=0.05, label="Mel temperature")
                compute_final_mel = gr.Checkbox(value=False, label="Compute final-step Mel head")
        generate = gr.Button("Generate speech", variant="primary")
        gr.Examples(
            examples=[[str(ROOT / "src/wavegen_v_spec/infer/examples/basic_ref_en.wav"),
                       "Some call me nature, others call me mother nature.",
                       "Here we generate something just for test."]],
            inputs=[ref_audio, ref_text, text], cache_examples=False,
        )
        cfg_schedule.change(guidance_default, inputs=[cfg_schedule, model_variant], outputs=cfg_strength, queue=False)
        model_variant.change(model_defaults, inputs=model_variant,
                             outputs=[steps, cfg_schedule, cfg_strength, mel_temperature, inference_method], queue=False)
        synthesizer.output_dir = Path(output.GRADIO_CACHE)
        generate.click(
            synthesizer,
            inputs=[ref_audio, ref_text, text, inference_method, cfg_schedule, cfg_strength,
                    steps, seed, speed, mel_temperature, compute_final_mel, model_variant],
            outputs=[output, status], concurrency_limit=1, concurrency_id="wavegen", api_name="generate",
        )
    return demo.queue(default_concurrency_limit=1, max_size=16)


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    create_app(arguments).launch(server_name=arguments.server_name, server_port=arguments.server_port,
                                 allowed_paths=[str(ROOT / "src/wavegen_v_spec/infer/examples/basic_ref_en.wav")],
                                 share=False, show_error=False)
