"""Seed-TTS metric definitions with lazy, explicitly selected model loaders."""

from pathlib import Path
import string
import warnings


def normalize_text(text, lang):
    from zhon.hanzi import punctuation

    for character in punctuation + string.punctuation:
        text = text.replace(character, "")
    text = text.replace("  ", " ")
    return " ".join(text) if lang == "zh" else text.lower()


def word_error_rate(truth, hypothesis, lang="en"):
    import jiwer

    truth, hypothesis = normalize_text(truth, lang), normalize_text(hypothesis, lang)
    if not truth.strip():
        raise ValueError("Reference text is empty after ASR normalization")
    return float(jiwer.wer(truth, hypothesis))


def run_asr_wer(items, directory, *, lang, asr_model, device="cpu", compute_type=None):
    directory = Path(directory)
    if not Path(asr_model).is_dir():
        raise ValueError("--asr-model must be an explicit local model directory")
    if lang == "en":
        from faster_whisper import WhisperModel

        cuda = str(device).startswith("cuda")
        index = int(str(device).split(":")[1]) if ":" in str(device) else 0
        model = WhisperModel(str(asr_model), device="cuda" if cuda else "cpu", device_index=index,
                             compute_type=compute_type or ("float16" if cuda else "int8"),
                             local_files_only=True)
        def transcribe(path):
            segments, _ = model.transcribe(str(path), beam_size=5, language="en")
            return "".join(" " + segment.text for segment in segments)
    elif lang == "zh":
        from funasr import AutoModel
        import torch
        import zhconv

        model = AutoModel(model=str(asr_model), device=str(device), disable_update=True,
                          ncpu=torch.get_num_threads())
        def transcribe(path):
            result = model.generate(input=str(path), batch_size_s=300, disable_pbar=True)
            return zhconv.convert(result[0]["text"], "zh-cn")
    else:
        raise ValueError("Only en and zh are supported")
    records = []
    for index, item in enumerate(items, 1):
        hypothesis = transcribe(directory / f"{item.utt}.wav")
        records.append({"wav": item.utt, "truth": item.text, "hypo": hypothesis,
                        "wer": word_error_rate(item.text, hypothesis, lang)})
        print(f"{'CER' if lang == 'zh' else 'WER'}: {index}/{len(items)}", flush=True)
    return records


def load_local_wavlm(s3prl_repo, wavlm_checkpoint):
    import importlib
    import sys

    repo = Path(s3prl_repo).resolve(strict=True)
    existing = sys.modules.get("s3prl")
    if existing is not None and Path(existing.__file__).resolve().parent != repo / "s3prl":
        raise RuntimeError("A different S3PRL checkout is already imported; use a fresh process")
    sys.path.insert(0, str(repo))
    try:
        # The all-model hub imports unrelated legacy audio backends. Load only WavLM.
        module = importlib.import_module("s3prl.upstream.wavlm.hubconf")
        expected = repo / "s3prl/upstream/wavlm/hubconf.py"
        if Path(module.__file__).resolve() != expected.resolve():
            raise RuntimeError("WavLM resolved outside the explicitly selected S3PRL checkout")
        return module.wavlm_local(ckpt=str(wavlm_checkpoint))
    finally:
        sys.path.remove(str(repo))


def load_sim_model(*, s3prl_repo, wavlm_checkpoint, sim_checkpoint, device):
    import torch
    from .ecapa_tdnn import ECAPA_TDNN_SMALL

    extractor = load_local_wavlm(s3prl_repo, wavlm_checkpoint)
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", feature_extract=extractor)
    state = torch.load(sim_checkpoint, map_location="cpu", weights_only=True)
    if "model" not in state:
        raise ValueError("SIM checkpoint must contain the ECAPA 'model' state dict")
    incompatible = model.load_state_dict(state["model"], strict=False)
    missing = [key for key in incompatible.missing_keys
               if not key.startswith("feature_extract.") and not key.endswith("num_batches_tracked")]
    if missing:
        raise ValueError(f"SIM checkpoint is missing scoring weights: {missing}")
    if incompatible.unexpected_keys:
        warnings.warn(f"Unused SIM checkpoint keys: {incompatible.unexpected_keys}", stacklevel=2)
    return model.to(device).eval()


def load_similarity_audio(path, device="cpu"):
    import soundfile as sf
    import torch
    import torchaudio

    wave, rate = sf.read(str(path), dtype="float32", always_2d=True)
    wave = torch.from_numpy(wave.T.copy()).mean(dim=0, keepdim=True).to(device)
    if rate != 16000:
        wave = torchaudio.transforms.Resample(rate, 16000).to(device)(wave)
    return wave


def run_sim(items, directory, *, s3prl_repo, wavlm_checkpoint, sim_checkpoint, device="cpu"):
    import torch
    import torch.nn.functional as F

    model = load_sim_model(s3prl_repo=s3prl_repo, wavlm_checkpoint=wavlm_checkpoint,
                           sim_checkpoint=sim_checkpoint, device=device)
    records = []
    with torch.inference_mode():
        for index, item in enumerate(items, 1):
            generated = model(load_similarity_audio(Path(directory) / f"{item.utt}.wav", device))
            reference = model(load_similarity_audio(item.ref_audio, device))
            score = F.cosine_similarity(generated, reference)[0].item()
            records.append({"wav": item.utt, "sim": score})
            print(f"SIM: {index}/{len(items)}", flush=True)
    return records


def run_utmos(items, directory, *, utmos_repo, utmos_checkpoint, device="cpu"):
    import librosa
    import torch

    model = torch.hub.load(str(utmos_repo), "utmos22_strong", source="local", pretrained=False)
    model.load_state_dict(torch.load(utmos_checkpoint, map_location="cpu", weights_only=True), strict=True)
    model = model.to(device).eval()
    records = []
    with torch.inference_mode():
        for index, item in enumerate(items, 1):
            wave, rate = librosa.load(Path(directory) / f"{item.utt}.wav", sr=None, mono=True)
            score = model(torch.from_numpy(wave).to(device).unsqueeze(0), rate).item()
            records.append({"wav": item.utt, "utmos": score})
            print(f"UTMOS: {index}/{len(items)}", flush=True)
    return records


# LibriSpeech test-clean partitioning follows the upstream WavTTS evaluator.
def get_librispeech_test(metalst, gen_wav_dir, gpus, librispeech_test_clean_path, eval_ground_truth=False):
    import os
    from tqdm import tqdm

    with open(metalst, encoding="utf-8") as handle:
        lines = handle.readlines()
    test_set_ = []
    seen = set()
    for line in tqdm(lines):
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line.strip().split("\t")
        if gen_utt in seen:
            raise ValueError(f"Duplicate generated utterance: {gen_utt}")
        seen.add(gen_utt)
        if eval_ground_truth:
            gen_spk_id, gen_chaptr_id, _ = gen_utt.split("-")
            gen_wav = os.path.join(librispeech_test_clean_path, gen_spk_id, gen_chaptr_id, gen_utt + ".flac")
        else:
            gen_wav = os.path.join(gen_wav_dir, gen_utt + ".wav")
        if not os.path.isfile(gen_wav):
            raise FileNotFoundError(f"Generated wav not found: {gen_utt}")
        ref_spk_id, ref_chaptr_id, _ = ref_utt.split("-")
        ref_wav = os.path.join(librispeech_test_clean_path, ref_spk_id, ref_chaptr_id, ref_utt + ".flac")
        if not os.path.isfile(ref_wav):
            raise FileNotFoundError(f"Reference wav not found: {ref_wav}")
        test_set_.append((gen_wav, ref_wav, gen_txt))
    if not test_set_:
        raise ValueError("LibriSpeech metadata contains no utterances")
    if len(gpus) == 1:
        return [(gpus[0], test_set_)]
    wav_per_job = len(test_set_) // len(gpus) + 1
    return [(rank, test_set_[index * wav_per_job:(index + 1) * wav_per_job])
            for index, rank in enumerate(gpus)
            if test_set_[index * wav_per_job:(index + 1) * wav_per_job]]


def _librispeech_items(test_set):
    from types import SimpleNamespace

    items = [SimpleNamespace(utt=Path(generated).stem, ref_audio=Path(reference), text=text)
             for generated, reference, text in test_set]
    return items, Path(test_set[0][0]).parent


def run_librispeech_asr_wer(args):
    rank, lang, test_set, asr_model = args
    items, directory = _librispeech_items(test_set)
    return run_asr_wer(items, directory, lang=lang, asr_model=asr_model, device=f"cuda:{rank}")


def run_librispeech_sim(args):
    rank, test_set, sim_checkpoint, s3prl_repo, wavlm_checkpoint = args
    items, directory = _librispeech_items(test_set)
    return run_sim(items, directory, s3prl_repo=s3prl_repo, wavlm_checkpoint=wavlm_checkpoint,
                   sim_checkpoint=sim_checkpoint, device=f"cuda:{rank}")
