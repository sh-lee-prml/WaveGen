# Evaluate with Librispeech test-clean, ~3s prompt to generate 4-10s audio (the way of valle/voicebox evaluation)

import argparse
import ast
import json
import os
import sys


sys.path.append(os.getcwd())

import multiprocessing as mp

import numpy as np

from wavegen_v_spec.eval.utils_eval import (
    get_librispeech_test,
    run_librispeech_asr_wer as run_asr_wer,
    run_librispeech_sim as run_sim,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eval_task", type=str, default="wer", choices=["sim", "wer"])
    parser.add_argument("-l", "--lang", choices=["en"], default="en")
    parser.add_argument("-g", "--gen_wav_dir", type=str, required=True)
    parser.add_argument("-p", "--librispeech_test_clean_path", type=str, required=True)
    parser.add_argument(
        "-n", "--gpu_nums", type=str, default="1", help="Number of GPUs to use (e.g., 8) or GPU list (e.g., [0,1,2,3])"
    )
    parser.add_argument("-t", "--task", type=str, default="ls_pc_test_clean", choices=["ls_pc_test_clean"])
    parser.add_argument(
        "--wavlm_ckpt_dir",
        type=str,
        default="",
        help="Path to wavlm_large_finetune.pth for SIM evaluation.",
    )
    parser.add_argument("--meta", default="data/librispeech_pc_test_clean_cross_sentence.lst")
    parser.add_argument("--asr-model", help="Local Faster-Whisper large-v3 directory")
    parser.add_argument("--s3prl-repo", help="Local S3PRL checkout for SIM")
    parser.add_argument("--wavlm-checkpoint", help="Local upstream WavLM Large checkpoint for SIM")
    return parser.parse_args()


def parse_gpu_nums(gpu_nums_str):
    try:
        if gpu_nums_str.startswith("[") and gpu_nums_str.endswith("]"):
            gpu_list = ast.literal_eval(gpu_nums_str)
            if isinstance(gpu_list, list) and gpu_list and all(type(rank) is int and rank >= 0 for rank in gpu_list):
                return gpu_list
            raise ValueError("Expected a nonempty list of GPU indices")
        count = int(gpu_nums_str)
        if count < 1:
            raise ValueError("Expected a positive GPU count")
        return list(range(count))
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(
            f"Invalid GPU specification: {gpu_nums_str}. Use a number (e.g., 8) or a list (e.g., [0,1,2,3])"
        )


def main():
    args = get_args()
    eval_task = args.eval_task
    lang = args.lang
    librispeech_test_clean_path = args.librispeech_test_clean_path  # test-clean path
    gen_wav_dir = args.gen_wav_dir

    metalst = args.meta

    gpus = parse_gpu_nums(args.gpu_nums)
    test_set = get_librispeech_test(metalst, gen_wav_dir, gpus, librispeech_test_clean_path)

    ## In LibriSpeech, some speakers utilized varying voice characteristics for different characters in the book,
    ## leading to a low similarity for the ground truth in some cases.
    # test_set = get_librispeech_test(metalst, gen_wav_dir, gpus, librispeech_test_clean_path, eval_ground_truth = True)  # eval ground truth

    wavlm_ckpt_dir = args.wavlm_ckpt_dir
    if eval_task == "sim" and not all((wavlm_ckpt_dir, args.s3prl_repo, args.wavlm_checkpoint)):
        raise ValueError(
            "SIM requires --wavlm_ckpt_dir, --s3prl-repo and --wavlm-checkpoint."
        )

    asr_ckpt_dir = args.asr_model
    if eval_task == "wer" and not asr_ckpt_dir:
        raise ValueError("WER requires --asr-model with a local Faster-Whisper large-v3 directory")

    # --------------------------------------------------------------------------

    full_results = []
    metrics = []

    if eval_task == "wer":
        with mp.Pool(processes=len(gpus)) as pool:
            args = [(rank, lang, sub_test_set, asr_ckpt_dir) for (rank, sub_test_set) in test_set]
            results = pool.map(run_asr_wer, args)
            for r in results:
                full_results.extend(r)
    elif eval_task == "sim":
        with mp.Pool(processes=len(gpus)) as pool:
            jobs = [(rank, sub_test_set, wavlm_ckpt_dir, args.s3prl_repo, args.wavlm_checkpoint)
                    for (rank, sub_test_set) in test_set]
            results = pool.map(run_sim, jobs)
            for r in results:
                full_results.extend(r)
    else:
        raise ValueError(f"Unknown metric type: {eval_task}")

    result_path = f"{gen_wav_dir}/_{eval_task}_results.jsonl"
    with open(result_path, "x", encoding="utf-8") as f:
        for line in full_results:
            metrics.append(line[eval_task])
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        metric = round(np.mean(metrics), 5)

    print(f"\nTotal {len(metrics)} samples")
    print(f"{eval_task.upper()}: {metric}")
    print(f"{eval_task.upper()} results saved to {result_path}")


if __name__ == "__main__":
    main()
