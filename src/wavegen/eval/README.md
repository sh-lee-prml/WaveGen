# Evaluation

## 1. Install dependencies

Run the following commands from the WaveGen repository root:

```bash
pip install -e ".[eval]"
```

## 2. Prepare evaluation models

Prepare the local assets for the metrics you want to run. WER uses
Faster-Whisper for English and Paraformer for Chinese (reported as CER).

| Metric | Model | Download |
| --- | --- | --- |
| English WER | Faster-Whisper large-v3, complete model directory | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v3) |
| Chinese CER | Paraformer-zh, complete model directory | [Hugging Face](https://huggingface.co/funasr/paraformer-zh) |
| SIM | WavLM Large feature extractor, `wavlm_large.pt` | [S3PRL checkpoint](https://huggingface.co/s3prl/converted_ckpts/blob/main/wavlm_large.pt) |
| SIM | Fine-tuned speaker model, `wavlm_large_finetune.pth` | [Seed-TTS evaluation checkpoint](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view) |
| UTMOS | `utmos22_strong_step7459_v1.pt` | [SpeechMOS checkpoint](https://github.com/tarepan/SpeechMOS/releases/download/v1.0.0/utmos22_strong_step7459_v1.pt) |

SIM needs both WavLM files; the feature extractor is WavLM **Large**.
Also prepare the S3PRL and SpeechMOS code used by the local loaders:

```bash
git clone https://github.com/s3prl/s3prl.git ckpts/metrics/s3prl
pip install -r ckpts/metrics/s3prl/requirements/install.txt
git clone --branch v1.2.0 --depth 1 https://github.com/tarepan/SpeechMOS.git ckpts/metrics/SpeechMOS
```

Place the downloaded models as follows, or change their paths in the evaluation
config. The ASR folders must include their configuration and tokenizer files.
SpeechMOS v1.2.0 uses the UTMOS weight from its v1.0.0 release.

```text
ckpts/metrics/
├── faster-whisper-large-v3/
├── paraformer-zh/
├── wavlm_large.pt
├── wavlm_large_finetune.pth
├── utmos22_strong_step7459_v1.pt
├── s3prl/
└── SpeechMOS/
```

The evaluation commands load local assets. `--infer-only` does not require metric
models. `--full-eval --eval-task wer`, `sim`, or `utmos` needs only the selected
metric's assets; `--eval-task all` runs all three.

## 3. Prepare test datasets

Download and extract the English and/or Chinese test set following the
[Seed-TTS evaluation instructions](https://github.com/BytedanceSpeech/seed-tts-eval#dataset):

```text
data/seedtts_testset/
├── en/
│   ├── meta.lst
│   ├── prompt-wavs/
│   └── wavs/
└── zh/
    ├── meta.lst
    ├── prompt-wavs/
    └── wavs/
```

Seed-TTS metadata has four or five pipe-separated fields:

```text
id|prompt text|prompt-wavs/example.wav|target text|optional target.wav
```

Audio paths in this list are relative to `meta.lst`.

For LibriSpeech, extract [test-clean](https://www.openslr.org/12/) with its
speaker/chapter structure intact:

```text
data/LibriSpeech/test-clean/<speaker>/<chapter>/<speaker>-<chapter>-<utterance>.flac
```

The cross-sentence list from WavTTS is included at
`data/librispeech_pc_test_clean_cross_sentence.lst`. It has six tab-separated
fields: reference ID, reference duration, reference text, target ID, target
duration, and target text. Select `--dataset librispeech` and pass the test-clean
directory with `--librispeech-test-clean-path`.

Generation uses reference audio, reference text, and target text. Output duration
is estimated by WaveGen; the target recording and metadata durations are not
used for generation.

## 4. Batch inference and evaluation

Edit [eval_infer_batch.toml](eval_infer_batch.toml) to set the checkpoint,
dataset, output directory, inference options, and metric model paths. Paths in
the config are relative to the working directory. Command-line options override
the config; `--config` selects the model YAML, while `--eval-config` selects this
batch configuration.

Generate the full test list:

```bash
python -m wavegen.eval.eval_infer_batch --eval-config src/wavegen/eval/eval_infer_batch.toml --infer-only
```

Generate and then evaluate each selected CFG condition:

```bash
python -m wavegen.eval.eval_infer_batch --eval-config src/wavegen/eval/eval_infer_batch.toml --full-eval
```

On Linux or in Bash, the wrapper uses the same config by default:

```bash
bash src/wavegen/eval/eval_infer_batch.sh --infer-only
bash src/wavegen/eval/eval_infer_batch.sh --full-eval
```

The Python commands also work in PowerShell. Metric paths are checked before
generation in `--full-eval` mode. The generation model is released before scoring.

| Setting | Default | Options |
| --- | --- | --- |
| Checkpoint | `ckpts/WaveGen/model_1300000.pt`, EMA | `--checkpoint`, `--no-ema` / `--ema` |
| Inference method | Prompt Anchor (`Prompt_xt_Anchor`) | `--inference-method Prompt_xt_Anchor` / `baseline` |
| CFG conditions | Both schedules | `--conditions both` / `constant` / `scheduled_k3` |
| ODE steps | 50 | `--steps` |
| Constant / K3 base excess CFG | 3 / 1.25 | `--cfg-strength`, `--scheduled-cfg-strength` |
| Seed / speed | 0 / 1 | `--seed`, `--speed` |
| Mel temperature / dropout | 0 / 0.2 | `--mel-temperature`, `--inference-dropout` |
| Generation device / dtype | CUDA if available, otherwise CPU / BF16 | `--device`, `--dtype` |
| Evaluation device | CPU | `--eval-device cuda:0` for GPU scoring |
| Metrics | All | `--eval-task all` / `wer` / `sim` / `utmos` |

Parameters remain FP32; CUDA inference uses BF16 autocast and CPU uses FP32.
Waveform temperature is 1, with Euler power-2 time mapping and shift 3.
K3 uses text-only null for the first three calls, then joint-null; its base
excess CFG is doubled for mapped time below 0.5. Prompt Anchor corrects prompt states
at each ODE step and masks them in joint-null CFG. Method and CFG schedule can
be selected independently.

For Chinese, override the metadata, language, ASR model, and output directory:

```bash
python -m wavegen.eval.eval_infer_batch --eval-config src/wavegen/eval/eval_infer_batch.toml --full-eval --lang zh --meta data/seedtts_testset/zh/meta.lst --asr-model ckpts/metrics/paraformer-zh --output-dir results/WaveGen/1300000/seedtts_test_zh
```

For LibriSpeech test-clean:

```bash
python -m wavegen.eval.eval_infer_batch --eval-config src/wavegen/eval/eval_infer_batch.toml --full-eval --dataset librispeech --librispeech-test-clean-path data/LibriSpeech/test-clean --meta data/librispeech_pc_test_clean_cross_sentence.lst --lang en --output-dir results/WaveGen/1300000/librispeech_test_clean
```

To compare baseline inference, select `--inference-method baseline` and a separate
`--output-dir`. Each run processes the complete metadata list, one utterance at
a time on one device. This entry point does not implement minibatch or
multi-GPU generation.

### Results and separate scoring

Each selected condition writes its WAV files, `inference_manifest.json`, and
metric result/summary files under `constant/` or `scheduled_k3/` in the output
directory. Complete stages with matching settings and assets are reused. Use a
new directory when changing settings; incomplete or mismatched results are
preserved for inspection.

To score an existing completed condition separately:

```bash
python -m wavegen.eval.eval_seedtts_testset --meta data/seedtts_testset/en/meta.lst --gen-wav-dir results/WaveGen/1300000/seedtts_test_en/constant --eval-task wer --lang en --device cpu --asr-model ckpts/metrics/faster-whisper-large-v3
```

Use `--eval-task sim` with `--s3prl-repo`, `--wavlm-checkpoint`, and
`--sim-checkpoint`, or `--eval-task utmos` with `--utmos-repo` and
`--utmos-checkpoint`. The same scorer accepts `--dataset librispeech` and
`--librispeech-test-clean-path`. WER/CER is the uncapped mean across utterances;
SIM compares generated speech with the prompt. The retained
`eval_librispeech_test_clean` module also supports the original WavTTS
WER/SIM scoring interface for existing LibriSpeech audio.

See the [WavTTS evaluation guide](https://github.com/cwx-worst-one/WavTTS/tree/main/src/wavtts/eval)
for the upstream workflow and [Seed-TTS evaluation](https://github.com/BytedanceSpeech/seed-tts-eval)
for the datasets and metric references.

## 5. Benchmark results

The following results were measured on October 3, 2026 with the 1.3M-step EMA
checkpoint and the inference settings in Section 4, selecting
`Prompt_xt_Anchor` and `scheduled_k3`. All requested utterances were scored;
the small volume pilot is identified separately below.

Two experimental preprocessing conditions are used in the comparisons:

- **Trimmed Prompts:** reference audio processed using silence preprocessing from the
  [F5-TTS CLI/Gradio pipeline](https://github.com/SWivid/F5-TTS/blob/283252563dbf91be625e0c27926acfaac449186c/src/f5_tts/infer/utils_infer.py),
  including leading/trailing trimming and 50 ms of appended silence. Supplied
  reference transcripts are retained. When trimming produces empty audio, the
  original prompt is retained; dataset-specific exceptions are listed below.
- **chunk30:** split long target text so that **the reference prompt plus each
  generated segment is at most 30 seconds**. Each segment uses the same prompt,
  with seed 0 reset for each call. All target text is preserved, and PCM segments
  are concatenated in order without added gaps or crossfades before scoring.
  The final concatenated recording can therefore exceed 30 seconds.

These conditions were run with separate experimental scripts; the batch
commands above do not automatically enable trimming or text splitting.
The native conditional RMS normalization is unchanged, and no additional gain
is applied except in the explicitly labeled volume pilot. Speaker similarity
always uses the original, unprocessed reference audio.

### LibriSpeech-PC

All **1,127 cross-sentence pairs** were evaluated using original prompts, with
no input/output silence trimming and no text splitting.

| Condition | WER (%) ↓ | WavLM SIM ↑ | UTMOS ↑ |
| --- | ---: | ---: | ---: |
| Original prompts; no splitting | 2.32 | 0.7470 | 4.0127 |

An audit of the actual WAV files found **no samples above 30 seconds**, including
the prompt. Thus, applying the same 30-second splitting rule would not split
any utterance in this run.

| Duration | Mean (s) | Maximum (s) |
| --- | ---: | ---: |
| Reference prompt | 3.53 | 4.50 |
| Generated speech | 6.35 | 18.27 |
| Prompt + generated speech | 9.87 | 22.13 |

Across the 1,127 pairs (448 unique prompts), leading and trailing prompt silence
totaled **0.61 s on average**, with a maximum of **1.27 s**. One pair had an
internal prompt pause of at least 1 s. Generated audio had internal pauses of
at least 1 s in 48 samples, with a maximum of 2.54 s. These are diagnostic
measurements using RMS below -42 dBFS in 10 ms windows, not neural VAD labels;
quiet speech and natural pauses can also fall below the threshold.

Applying trimming in memory shortened prompts by an average of **0.56 s**.
No 12-second hard crop or prompt shorter than 1 s occurred. This was an
audio-only audit, **not a new inference/evaluation run**. Its effect on WER,
SIM, and UTMOS has not been measured. Since prompt duration enters the native
length estimate, trimming can change generation timing even without splitting.

### Seed Hard

This is the **Chinese `zh/hardcase.lst` subset, 400 utterances**. CER uses
Paraformer; speaker similarity and quality use the WavLM SIM and UTMOS metrics
described above.

| Condition | CER (%) ↓ | WavLM SIM ↑ | UTMOS ↑ |
| --- | ---: | ---: | ---: |
| Original prompts; no splitting | 13.46 | 0.7653 | 2.5920 |
| Trimmed Prompts; no splitting | 11.76 | 0.7682 | 2.6081 |
| Trimmed Prompts + chunk30 | 8.64 | 0.7807 | 2.6647 |

After trimming, **34 long utterances were split into 74 generated segments**.
Their CER decreased from **45.37% to 8.65%**: 30 improved and 4 worsened. The
longest model call, including its prompt, was **29.96 s**. The other 366 WAVs
were reused after a fresh short-utterance equivalence check; all 400 were
rescored, and the reused WAVs had identical scores across all three metrics.
No trimmed prompt became empty or shorter than 1 s. No additional volume correction
was used.

### CV3-Eval

The [CV3-Eval benchmark](https://github.com/QwenAudio/CV3-Eval) was evaluated
separately on its regular EN/ZH sets and its Hard sets. These runs use Whisper
large-v3 for English WER, SeACo-Paraformer for Chinese CER, **ERes2Net** speaker
similarity, and **DNSMOS P.808** quality scores. Error rates are uncapped means
over utterances. ERes2Net SIM and DNSMOS are different metrics from the WavLM
SIM and UTMOS reported for LibriSpeech-PC and Seed Hard.

#### Regular sets

Each language contains **500 utterances**. These runs did not split target text.

| Language | Prompt processing | WER / CER (%) ↓ | ERes2Net SIM ↑ | DNSMOS P.808 ↑ |
| --- | --- | ---: | ---: | ---: |
| EN | Original prompts | 12.89 | 0.7209 | 3.5655 |
| EN | Trimmed Prompts | 6.72 | 0.7225 | 3.6105 |
| ZH | Original prompts | 11.17 | 0.7655 | 3.6575 |
| ZH | Trimmed Prompts | 6.57 | 0.7679 | 3.6864 |

For EN with trimming, SIM averages **499 finite scores**: `uttid_46` has an undefined
speaker score after an extremely short generation. It is retained as undefined,
not assigned zero; WER and DNSMOS cover all 500 samples. The original-prompt SIM
on those same 499 IDs is 0.7211. Thirteen EN prompts that became empty after
trimming retained their original audio; no ZH prompt required this fallback.

#### Hard sets

The Hard sets contain **64 EN** and **60 ZH** utterances. Every condition was
evaluated on the complete corresponding set.

| Language | Condition | WER / CER (%) ↓ | ERes2Net SIM ↑ | DNSMOS P.808 ↑ |
| --- | --- | ---: | ---: | ---: |
| EN | Original prompts; no splitting | 47.95 | 0.5554 | 3.1520 |
| EN | Trimmed Prompts; no splitting | 26.72 | 0.6600 | 3.4761 |
| EN | Original prompts + chunk30 | 25.73 | 0.7131 | 3.4320 |
| EN | Trimmed Prompts + chunk30 | 14.67 | 0.7141 | 3.6190 |
| ZH | Original prompts; no splitting | 34.68 | 0.6836 | 3.3292 |
| ZH | Trimmed Prompts; no splitting | 18.72 | 0.7336 | 3.5362 |
| ZH | Original prompts + chunk30 | 19.95 | 0.7326 | 3.5253 |
| ZH | Trimmed Prompts + chunk30 | 13.33 | 0.7478 | 3.5971 |

With original prompts, 31 EN and 17 ZH utterances required splitting; after
trimming, 15 EN and 9 ZH required it. Across the four chunk30 conditions, these
72 utterance/condition pairs produced 193 new segments, with a maximum
prompt-plus-segment duration of **29.975 s**. The other 176 WAVs were reused
after fresh equivalence controls, and all 248 final recordings were rescored.
One reused EN WAV in the Trimmed Prompts condition (`uttid_9`) changed WER on
rescoring due to Whisper decoding variation, contributing +0.0558 percentage
points to that condition's mean; the other 175 reused WAVs had unchanged scores.

EN `uttid_43` retained its original prompt because trimming returned empty
audio. Very quiet prompts can also be over-trimmed without becoming empty:
EN `uttid_45` was reduced to about 0.30 s and ZH `uttid_60` to about 0.97 s.
Text splitting alone does not repair these shortened prompts.

#### Selected low-volume pilot

This is a **selected subset, not a full-set benchmark**: the six quietest
original prompts with RMS below 0.01 per language, plus one normal-volume
control per language. Selection used input audio statistics only. The table
below includes only the six quiet samples in each language.

| Condition | EN WER (%), n=6 ↓ | ZH CER (%), n=6 ↓ |
| --- | ---: | ---: |
| Trimmed Prompts + chunk30 (baseline) | 41.87 | 5.86 |
| Output gain only | 39.61 | 5.86 |
| Prompt gain before trimming + output peak protection | 13.67 | 18.71 |

Gain targeted RMS 0.1 with a peak limit of 0.95. In the prompt-gain condition,
gain was applied before trimming and inference; a single output gain across
all segments prevented PCM clipping before concatenation. The normal-volume
controls retained identical PCM and scores.

Prompt gain improved the EN subset but worsened the ZH subset. It also changes
which speech survives trimming, the duration estimate, and sometimes the number
of generated segments, so this does not isolate an amplitude-only effect.
These results do **not** support enabling extra gain for every quiet prompt.
