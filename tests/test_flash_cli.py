"""Flash entry points keep the published schedule fixed and record every option."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wavegen_flash.infer import infer_cli
from wavegen_flash.infer.infer_cli import build_parser, load_engine, validate_options
from wavegen_flash.eval.eval_infer_batch import build_parser as batch_parser, run


PROMPT = ['-r', 'reference.wav', '-s', 'Reference.', '-t', 'Target.']


@pytest.mark.parametrize('flag,value', [('--steps', '4'), ('--nfe-step', '4'),
                                       ('--cfg-strength', '0'), ('--inference-dropout', '0'),
                                       ('--time-shift', '3')])
def test_flash_cli_cannot_override_published_sampler(flag, value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(PROMPT + [flag, value])


def test_flash_local_and_hub_loading_options():
    calls = []

    class Engine:
        @classmethod
        def from_checkpoint(cls, *args, **kwargs):
            calls.append(('local', args, kwargs))

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(('hub', args, kwargs))

    parser = build_parser()
    load_engine(parser.parse_args(PROMPT), engine_class=Engine)
    assert calls[-1] == ('local', ('ckpts/WaveGen_Flash/model.safetensors', 'ckpts/WaveGen_Flash/vocab.txt'),
                         dict(config='ckpts/WaveGen_Flash/WaveGen_Flash.yaml',
                              device=None, dtype='bf16', compute_final_mel=None))
    args = parser.parse_args(PROMPT + ['--repo-id', 'owner/model', '--revision', 'a' * 40,
                                      '--local-files-only', '--compute-final-mel'])
    load_engine(args, engine_class=Engine)
    assert calls[-1][0] == 'hub'
    assert calls[-1][2]['checkpoint_filename'] == 'WaveGen_Flash/model.safetensors'
    assert calls[-1][2]['local_files_only'] is True
    assert calls[-1][2]['compute_final_mel'] is True
    assert parser.parse_args(PROMPT + ['--no-compute-final-mel']).compute_final_mel is False
    for extra in (['--repo-id', 'owner/model'], ['--revision', 'main'],
                  ['--repo-id', 'owner/model', '--revision', 'main', '--checkpoint', 'local.safetensors']):
        with pytest.raises(ValueError):
            validate_options(parser.parse_args(PROMPT + extra))


@pytest.mark.parametrize('value', ['-0.1', 'nan', 'inf'])
def test_flash_cli_rejects_invalid_mel_temperature(value):
    with pytest.raises(ValueError, match='mel-temperature'):
        validate_options(build_parser().parse_args(PROMPT + ['--mel-temperature', value]))


@pytest.mark.parametrize('temperature', [None, 0.0, 0.5, 1.0])
def test_flash_cli_forwards_initial_mel_temperature(monkeypatch, tmp_path, temperature):
    calls = []
    def synthesize(*args, **kwargs):
        calls.append(kwargs)
        return 24000, np.zeros(480, dtype=np.float32)
    monkeypatch.setattr(infer_cli, 'load_engine', lambda _args: SimpleNamespace(synthesize=synthesize))
    argv = PROMPT + ['--output', str(tmp_path / 'output.wav')]
    if temperature is not None:
        argv += ['--mel-temperature', str(temperature)]
    assert infer_cli.main(argv) == 0
    assert calls[0]['mel_temperature'] == temperature


def test_flash_batch_records_fixed_sampler_and_prevents_incompatible_reuse(tmp_path):
    ref = tmp_path / 'reference.wav'
    ref.write_bytes(b'fixture input used by the stub only')
    meta = tmp_path / 'meta.lst'
    meta.write_text('sample|Reference.|reference.wav|Target.\n', encoding='utf-8')
    checkpoint = tmp_path / 'model.safetensors'
    checkpoint.write_bytes(b'fixture checkpoint used by the stub only')
    config = tmp_path / 'WaveGen_Flash.yaml'
    config.write_text('inference:\n  num_steps: 4\n', encoding='utf-8')
    vocab = tmp_path / 'vocab.txt'
    vocab.write_text(' \na\n', encoding='utf-8')
    calls = []
    settings = dict(num_steps=4, time_grid=[0, 0.1, 0.25, 0.5, 1], time_shift=3,
                    cfg_strength=0, inference_bottleneck_dropout=0, mel_temperature=0,
                    compute_final_mel=False, max_duration_seconds=30)

    class Engine:
        @classmethod
        def from_checkpoint(cls, *args, **kwargs):
            def synthesize(*audio_args, **options):
                calls.append(options)
                return 24000, np.zeros(480, dtype=np.float32)
            return SimpleNamespace(checkpoint=checkpoint, config=config, vocab_file=vocab,
                                   settings=settings, device='cpu', dtype='fp32', synthesize=synthesize)

    parser = batch_parser()
    output = tmp_path / 'output'
    argv = ['--meta', str(meta), '--output-dir', str(output)]
    result = run(parser.parse_args(argv), engine_class=Engine)
    assert result['count'] == 1 and result['reused'] is False
    assert calls == [dict(seed=0, speed=1.0, max_duration_seconds=30, compute_final_mel=False,
                         mel_temperature=0)]
    manifest = json.loads((output / 'inference_manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['contract']['sampler'] == settings
    assert set(manifest['contract']['source']) == {'wavegen_flash', 'wavegen_v_spec'}
    assert run(parser.parse_args(argv), engine_class=Engine)['reused'] is True
    with pytest.raises(ValueError, match='different manifest'):
        run(parser.parse_args(argv + ['--compute-final-mel']), engine_class=Engine)
    with pytest.raises(ValueError, match='different manifest'):
        run(parser.parse_args(argv + ['--mel-temperature', '1']), engine_class=Engine)
    second = tmp_path / 'with-final-mel'
    run(parser.parse_args(['--meta', str(meta), '--output-dir', str(second), '--compute-final-mel']),
        engine_class=Engine)
    assert calls[-1]['compute_final_mel'] is True
    other = json.loads((second / 'inference_manifest.json').read_text())
    assert other['contract']['sampler']['compute_final_mel'] is True
    legacy = tmp_path / 'legacy-temperature'
    legacy_argv = ['--meta', str(meta), '--output-dir', str(legacy), '--mel-temperature', '1']
    run(parser.parse_args(legacy_argv), engine_class=Engine)
    assert calls[-1]['mel_temperature'] == 1
    legacy_manifest = json.loads((legacy / 'inference_manifest.json').read_text())
    assert legacy_manifest['contract']['sampler']['mel_temperature'] == 1
    assert run(parser.parse_args(legacy_argv), engine_class=Engine)['reused'] is True
