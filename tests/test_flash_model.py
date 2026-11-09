"""Fixed Flash inference contract, masks, RNG, and complete EMA loading."""
from importlib.resources import files

import numpy as np
from omegaconf import OmegaConf
import pytest
from safetensors.torch import save_file
import soundfile as sf
import torch
from torch.nn import functional as F

from wavegen_flash.infer.utils_infer import WaveGenFlash, read_config
from wavegen_flash.model import FlashModel, FlashSampler
from wavegen_v_spec.model.backbones.dit import DiT


GRID = [0.0, 0.1, 0.25, 0.5, 1.0]


@pytest.fixture(autouse=True, scope='module')
def cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def tiny_model():
    torch.manual_seed(215)
    model = FlashModel(DiT(
        dim=32, depth=1, heads=2, dim_head=16, dropout=0.0, ff_mult=2,
        wav_frame_len=480, mel_channels=8, text_num_embeds=5, text_dim=16,
        conv_layers=0, use_mel_information_bottleneck=True, mel_bottleneck_dim=4,
        mel_bottleneck_dropout=0.2, use_audio_proj=True,
        audio_proj_dim=32, audio_proj_hidden=64),
        {' ': 0, 'a': 1, 'b': 2, 'c': 3, '.': 4})
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.02)
    return model


@pytest.fixture
def tiny_bundle(tmp_path):
    model = tiny_model()
    document = OmegaConf.load(str(files('wavegen_flash').joinpath('configs/WaveGen_Flash.yaml')))
    document.model.arch = dict(dim=32, depth=1, heads=2, dim_head=16, dropout=0.0, ff_mult=2,
        mel_channels=8, text_dim=16, conv_layers=0, use_mel_information_bottleneck=True,
        mel_bottleneck_dim=4, mel_bottleneck_dropout=0.2, use_audio_proj=True,
        audio_proj_dim=32, audio_proj_hidden=64)
    config = tmp_path / 'WaveGen_Flash.yaml'
    OmegaConf.save(document, config)
    vocab = tmp_path / 'vocab.txt'
    vocab.write_text(' \na\nb\nc\n.\n', encoding='utf-8')
    checkpoint = tmp_path / 'model.safetensors'
    state = {'ema_model.' + key: value.detach().clone() for key, value in model.state_dict().items()}
    metadata = {'format': 'pt', 'model': 'WaveGen_Flash', 'weights': 'ema', 'license': 'cc-by-nc-4.0'}
    save_file(state, str(checkpoint), metadata=metadata)
    return model, document, config, vocab, checkpoint, state, metadata


@pytest.mark.parametrize('field,value', [
    ('num_steps', 2), ('num_steps', 4.0), ('num_steps', True),
    ('sampling_method', 'midpoint'), ('time_shift', 1.0),
    ('time_grid', [0.0, 0.25, 0.5, 0.75, 1.0]),
    ('time_grid', [False, 0.1, 0.25, 0.5, True]),
    ('cfg_strength', 1.25), ('inference_bottleneck_dropout', 0.2),
    ('waveform_temperature', 0.0), ('mel_temperature', 0.0),
    ('compute_final_mel', 'false'), ('max_duration_seconds', float('nan')),
])
def test_flash_config_rejects_untrained_sampling_settings(tmp_path, field, value):
    document = OmegaConf.load(str(files('wavegen_flash').joinpath('configs/WaveGen_Flash.yaml')))
    document.inference[field] = value
    config = tmp_path / 'invalid.yaml'
    OmegaConf.save(document, config)
    with pytest.raises(ValueError):
        read_config(config)


def test_flash_sampler_anchors_and_noise_consumption():
    model = tiny_model()
    sampler = FlashSampler(model, GRID)
    wav = torch.randn(2, 3847) * 0.1
    lengths = torch.tensor([3847, 3491])
    starts = torch.tensor([1440, 960])
    mask = torch.arange(wav.shape[-1])[None, :] >= starts[:, None]
    padded = F.pad(wav, (0, 4320 - wav.shape[-1]))
    valid = torch.arange(4320)[None, :] < lengths[:, None]
    target = F.pad(mask, (0, 4320 - wav.shape[-1]), value=False) & valid
    known = valid & ~target
    raw = padded.masked_fill(~valid, 0)
    clean_wav = raw * model.latents_scale
    mel_valid = model._wav_mask_to_mel_mask(valid, 9)
    mel_known = mel_valid & ~model._wav_mask_to_mel_mask(target, 9)
    clean_mel = model._mel_from_wav(raw, 9).masked_fill(~mel_valid[..., None], 0)
    torch.manual_seed(17)
    noise_wav, noise_mel = torch.randn_like(clean_wav), torch.randn_like(clean_mel)
    expected_rng = torch.get_rng_state().clone()
    calls = []
    def capture(_module, _args, kwargs):
        calls.append({key: value.detach().clone() if torch.is_tensor(value) else value
                      for key, value in kwargs.items()})
    handle = model.transformer.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        torch.manual_seed(17)
        out, mel = sampler.generate(wav, torch.tensor([[1, 2], [1, 3]]), lengths, mask)
    finally:
        handle.remove()
    assert len(calls) == 4 and mel is None
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for index, call in enumerate(calls):
        t = GRID[index]
        assert torch.equal(call['time'], torch.full((2,), t, dtype=torch.float32))
        assert call['predict_mel'] is (index != 3)
        assert call['cache'] is False and call['inference_bottleneck_dropout'] == 0.0
        assert torch.equal(call['mask'], valid) and torch.equal(call['lens'], lengths)
        assert torch.equal(call['x'][known], ((1 - t) * noise_wav + t * clean_wav)[known])
        known_mel = mel_known[..., None].expand_as(clean_mel)
        assert torch.equal(call['mel'][known_mel], ((1 - t) * noise_mel + t * clean_mel)[known_mel])
        assert torch.count_nonzero(call['x'][~valid]) == 0
        assert torch.count_nonzero(call['mel'][~mel_valid]) == 0
    assert torch.equal(out[known], (clean_wav / model.latents_scale)[known])
    assert torch.count_nonzero(out[~valid]) == 0


def test_flash_complete_ema_load_and_final_head_invariance(tiny_bundle, tmp_path):
    model, _, config, vocab, checkpoint, _, _ = tiny_bundle
    torch.manual_seed(845)
    initial_rng = torch.get_rng_state().clone()
    engine = WaveGenFlash.from_checkpoint(checkpoint, vocab, config=config, device='cpu', dtype='fp32')
    assert torch.equal(torch.get_rng_state(), initial_rng)
    for key, expected in model.state_dict().items():
        assert torch.equal(engine.model.state_dict()[key].view(torch.uint8), expected.view(torch.uint8))
    assert engine.num_steps == 4 and engine.sampler.time_grid == tuple(GRID)
    assert all(not parameter.requires_grad for parameter in engine.model.parameters())
    prompt = tmp_path / 'reference.wav'
    sf.write(prompt, np.sin(np.arange(3840) * 2 * np.pi * 220 / 24000).astype(np.float32) * 0.12,
             24000, subtype='FLOAT')
    counts = {'wav': 0, 'mel': 0}
    def wav_hook(*_args):
        counts['wav'] += 1
    def mel_hook(*_args):
        counts['mel'] += 1
    handles = [engine.model.transformer.wav_ddt[0].register_forward_hook(wav_hook),
               engine.model.transformer.mel_ddt[0].register_forward_hook(mel_hook)]
    try:
        rate, full = engine.synthesize(prompt, 'a b.', 'a c.', seed=19, compute_final_mel=True)
        assert counts == {'wav': 4, 'mel': 4}
        counts.update(wav=0, mel=0)
        skipped_rate, skipped = engine.synthesize(prompt, 'a b.', 'a c.', seed=19)
        assert counts == {'wav': 4, 'mel': 3}
    finally:
        for handle in handles:
            handle.remove()
    assert rate == skipped_rate == 24000 and np.array_equal(full, skipped)
    assert torch.equal(torch.get_rng_state(), initial_rng)
    with pytest.raises(TypeError):
        engine.synthesize(prompt, 'a b.', 'a c.', steps=8)


@pytest.mark.parametrize('damage', ['metadata', 'missing', 'extra', 'dtype', 'prefix', 'shape', 'pt'])
def test_flash_rejects_incompatible_checkpoints(tiny_bundle, damage, tmp_path):
    _, _, config, vocab, _, state, metadata = tiny_bundle
    checkpoint = tmp_path / ('wrong.pt' if damage == 'pt' else 'wrong.safetensors')
    if damage == 'metadata':
        metadata['model'] = 'WaveGen_VSpec'
    elif damage == 'missing':
        state.pop('ema_model.transformer.norm_out.linear.weight')
    elif damage == 'extra':
        state['ema_model.unexpected.weight'] = torch.ones(1)
    elif damage == 'dtype':
        key = next(iter(state))
        state[key] = state[key].half()
    elif damage == 'prefix':
        key = next(iter(state))
        state[key.removeprefix('ema_model.')] = state.pop(key)
    elif damage == 'shape':
        state['ema_model.transformer.norm_out.linear.bias'] = torch.ones(1)
    save_file(state, str(checkpoint), metadata=metadata)
    with pytest.raises((ValueError, RuntimeError)):
        WaveGenFlash.from_checkpoint(checkpoint, vocab, config=config, device='cpu', dtype='fp32')
