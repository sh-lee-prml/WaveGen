# Training

Run from the repository root using the existing WaveGen environment.
The complete recipe and data layout are described in the
[package README](../README.md#training).

```bash
python -m wavegen_v_spec.train.train --cfg job
accelerate launch --num_processes 8 --mixed_precision bf16 -m wavegen_v_spec.train.train
```

Training updates the full generator with waveform velocity MSE, SF 0.1, and
V-Spec 0.1. The original 1300k EMA initializes fresh Online weights. The
optimizer, schedule, delayed EMA, rank RNG, and dataset cursor are saved for
resuming this recipe. The released EMA-only 1400k inference file has none of
the optimizer or cursor state required for resuming.

```bash
accelerate launch --num_processes 8 --mixed_precision bf16 \
  -m wavegen_v_spec.train.train \
  ckpts.resume_ckpt=/path/to/full_training_checkpoint.pt
```

The default output is `ckpts/WaveGen_VSpec_Train/`. A custom `ckpts.save_dir`
must end in `WaveGen_VSpec_Train`. Automatic resume uses `model_last.pt`;
numbered checkpoints require the explicit resume option. Source relocation
is accepted only for the bound SHA-256, size, and unchanged training settings.
