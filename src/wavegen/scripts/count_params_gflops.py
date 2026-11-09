"""Estimate one WaveGen training forward pass with THOP (optional dependency)."""

import argparse
from importlib.resources import files
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(files("wavegen").joinpath("configs/WaveGen_SF_Emilia_24k_50Hz_IBDrop02.yaml")))
    parser.add_argument("--vocab-file", default="ckpts/WaveGen/vocab.txt")
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument("--text-length", type=int, default=150)
    args = parser.parse_args()
    if args.duration <= 0 or args.text_length < 1:
        parser.error("duration and text-length must be positive")

    import thop
    import torch
    from omegaconf import OmegaConf
    from wavegen.model import CFM, DiT

    config = OmegaConf.to_container(OmegaConf.load(args.config).model, resolve=True)
    vocab = Path(args.vocab_file).read_text(encoding="utf-8").splitlines()
    waveform = dict(config["waveform"])
    arch = dict(config["arch"], text_num_embeds=len(vocab), wav_frame_len=waveform["wav_frame_len"])
    model = CFM(transformer=DiT(**arch), waveform_kwargs=waveform, **config["cfm"])
    samples = int(args.duration * waveform["target_sample_rate"])
    with torch.no_grad():
        operations, _ = thop.profile(
            model, inputs=(torch.randn(1, samples), torch.zeros(1, args.text_length, dtype=torch.long))
        )
    print(f"THOP operations: {operations / 1e9:.3f} G (partial estimate; functional attention is not counted)")
    print(f"Params: {sum(parameter.numel() for parameter in model.parameters()) / 1e6:.3f} M")


if __name__ == "__main__":
    main()
