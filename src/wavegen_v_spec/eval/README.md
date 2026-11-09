# Evaluation

Configure dataset metadata, output location, and local metric assets in
`eval_infer_batch.toml`, then run from the repository root:

```bash
python -m wavegen_v_spec.eval.eval_infer_batch \
  --eval-config src/wavegen_v_spec/eval/eval_infer_batch.toml --full-eval
```

The recommended default is V-Spec 1400k EMA
(`ckpts/WaveGen_VSpec/model_1400000.safetensors`) and Prompt Anchor +
Scheduled K3 on the full supplied metadata list. `--no-ema` requires a full checkpoint containing Online weights; it is unavailable for the released EMA-only bundle. `--infer-only` skips scoring.
Existing `.pt` checkpoints can still be selected with `--checkpoint` and
matching `--config` and `--vocab-file` paths.
The final-step Mel output head is skipped by default, with the Mel input and
information bottleneck retained. Pass `--compute-final-mel` or set
`compute_final_mel = true` under `[inference]` to compute it as well;
`--no-compute-final-mel` restores the default. The choice is recorded in each
generation manifest. See the [inference notes](../infer/README.md#final-step-mel-head)
for the Euler sampling behavior.

Use `eval_seedtts_testset` or `eval_utmos` to score existing outputs directly.
`--conditions constant` or `both` enables
the other guidance conditions explicitly. Outputs and manifests have a separate
V-Spec namespace.

WER/CER, WavLM/ECAPA SIM, and UTMOS use the same evaluation implementations as
the base WaveGen package. ASR, speaker-similarity, and UTMOS model assets must be
provided locally. Use each module's `--help` for individual scoring commands.

No benchmark scores are bundled as results of this checkpoint.
