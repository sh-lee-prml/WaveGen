"""CPU regression checks for full-generator V-Spec training and EMA-only inference."""
import copy
import importlib.util
from importlib.resources import files
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from wavegen_v_spec.infer.infer_cli import build_parser, resolve_options
from wavegen_v_spec.infer.utils_infer import _select_state
from wavegen_v_spec.model import CFM, DiT
from wavegen_v_spec.train.checkpoint import (
    relocated_stage_source, SOURCE_CHECKPOINT_SHA256, SOURCE_CHECKPOINT_SIZE,
    SPECTRAL_CONTRACT, STAGE_FORMAT, LEGACY_STAGE_FORMAT,
)
from wavegen_v_spec.train.delayed_ema import DelayedGeneratorEMA
from wavegen_v_spec.train.train import (
    _validate_config, _validate_model_contract, learning_rate_at_update,
    source_ema_initial_state, STAGE_CHECKPOINT_KEYS, STAGE_KEY, SOURCE_UPDATE,
    VSpecTrainer,
)


@pytest.fixture(autouse=True, scope='session')
def cpu_threads():
    torch.set_num_threads(1)


def tiny_generator():
    torch.manual_seed(281)
    model = CFM(transformer=DiT(
        dim=32, depth=1, heads=2, dim_head=16, dropout=0.0, ff_mult=2,
        wav_frame_len=480, mel_channels=8, text_num_embeds=16, text_dim=16,
        conv_layers=0, use_mel_information_bottleneck=True, mel_bottleneck_dim=4,
        mel_bottleneck_dropout=0.2, use_audio_proj=True,
        audio_proj_dim=32, audio_proj_hidden=64),
        sample_rate=24000, waveform_kwargs={'wav_frame_len': 480},
        prediction='x_pred', loss_space='v', t_eps=0.02, latents_scale=9.0,
        frac_lengths_mask=(0.8, 0.8), ms_v_spec_loss_weight=0.1,
        sf_loss_weight=0.1, inference_bottleneck_dropout=0.2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.02)
    return model


def test_full_training_keeps_sf_and_waveform_gradients():
    model = tiny_generator().train()
    _validate_model_contract(model)
    random.seed(42)
    torch.manual_seed(42)
    loss, _, _, metrics = model(torch.randn(2, 3840) * 0.12,
                                torch.tensor([[1, 2], [3, 4]]))
    expected = (metrics['flow_loss'] + 0.1 * metrics['sf_loss']) + metrics['ms_v_spec_loss_contribution']
    assert torch.equal(loss, expected)
    assert torch.equal(metrics['ms_v_spec_loss_contribution'], 0.1 * metrics['ms_v_spec_loss'])
    loss.backward()
    for name in ('transformer.proj_out.weight', 'transformer.mel_proj_out.weight'):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert all(p.requires_grad for p in model.parameters())


def test_model_contract_rejects_frozen_parameters():
    model = tiny_generator()
    next(model.parameters()).requires_grad_(False)
    with pytest.raises(ValueError, match='All generator parameters'):
        _validate_model_contract(model)


def test_bootstrap_uses_ema_and_removes_only_legacy_analysis_buffers():
    target = {'weight': torch.zeros(2, 3)}
    ema = {'step': torch.tensor(1300000), 'initted': torch.tensor(True),
           'ema_model.weight': torch.full((2, 3), 7.0)}
    for i, (window, bands) in enumerate(zip(SPECTRAL_CONTRACT['window_lengths'], SPECTRAL_CONTRACT['band_counts'])):
        root = f'ema_model.aux_mel_loss.mel_transforms.{i}'
        ema[root + '.spectrogram.window'] = torch.ones(window)
        ema[root + '.mel_scale.fb'] = torch.ones(window // 2 + 1, bands)
    state = source_ema_initial_state(ema, target)
    assert set(state) == {'weight'}
    assert torch.equal(state['weight'], torch.full((2, 3), 7.0))
    assert len(ema) == 17  # Source state is never modified.


def test_ema_only_checkpoint_never_falls_back_to_online():
    checkpoint = {'ema_model_state_dict': {'initted': torch.tensor(True),
                  'step': torch.tensor(98001), 'ema_model.weight': torch.ones(2)}}
    assert torch.equal(_select_state(checkpoint, True)['weight'], torch.ones(2))
    with pytest.raises(ValueError, match='model_state_dict'):
        _select_state(checkpoint, False)


def test_defaults_and_training_output_are_separate_from_inference_assets():
    parser = build_parser()
    options = resolve_options(parser.parse_args([]), parser)
    assert options['ckpt_file'] == 'ckpts/WaveGen_VSpec/model_1400000.safetensors'
    assert options['checkpoint_state'] == 'ema'
    assert options['cfg_schedule'] == 'scheduled_k3'
    assert options['inference_method'] == 'Prompt_xt_Anchor'
    cfg = OmegaConf.load(str(files('wavegen_v_spec').joinpath('configs/WaveGen_VSpec.yaml')))
    assert _validate_config(cfg) == 512
    assert cfg.ckpts.save_dir == 'ckpts/WaveGen_VSpec_Train'
    cfg.model.cfm.sf_loss_weight = 0.0
    with pytest.raises(ValueError, match='model settings'):
        _validate_config(cfg)


def test_learning_rate_warmup_and_checkpoint_payload():
    assert learning_rate_at_update(0) == pytest.approx(2e-7)
    assert learning_rate_at_update(1000) == pytest.approx(1.1e-6)
    assert learning_rate_at_update(2000) == pytest.approx(2e-6)
    assert learning_rate_at_update(100000) == pytest.approx(2e-6)
    with pytest.raises(ValueError):
        learning_rate_at_update(-1)
    assert not any('discriminator' in name for name in STAGE_CHECKPOINT_KEYS)
    assert importlib.util.find_spec('wavegen_v_spec.model.discriminators') is None


def test_delayed_ema_exact_start_resume_and_rank_ownership():
    model = torch.nn.Linear(2, 1)
    delayed = DelayedGeneratorEMA(model, enabled=True)
    for update in range(1, 2000):
        delayed.after_step(update)
    assert delayed.shadow is None and not delayed.initialized
    assert delayed.state_dict()['step'].item() == 0
    with torch.no_grad():
        model.weight.fill_(3.25)
        model.bias.fill_(-1.5)
    delayed.after_step(2000)
    assert delayed.initialized and delayed.native_step == 1 and delayed.age == 0
    for key, value in model.state_dict().items():
        assert torch.equal(delayed.shadow[key], value)
    resumed = DelayedGeneratorEMA(copy.deepcopy(model), enabled=True)
    resumed.load_state_dict(copy.deepcopy(delayed.state_dict()), stage_update=2000,
                            tracking=delayed.tracking_state())
    resumed.after_step(2001)
    assert resumed.native_step == 2 and resumed.age == 1
    with pytest.raises(RuntimeError, match='exactly once'):
        resumed.after_step(2001)
    follower = DelayedGeneratorEMA(model, enabled=False)
    for update in range(1, 2002):
        follower.after_step(update)
    assert follower.initialized and follower.shadow is None
    with pytest.raises(RuntimeError, match='main rank'):
        follower.state_dict()


def test_relocation_preserves_source_identity_and_training_recipe():
    old = {'source_checkpoint': '/old/model_1300000.pt',
           'source_sha256': SOURCE_CHECKPOINT_SHA256,
           'recipe': {'ms_v_spec_loss_weight': 0.1}, 'update': 1300000}
    new = copy.deepcopy(old)
    new['source_checkpoint'] = '/portable/ckpts/WaveGen/model_1300000.pt'
    assert relocated_stage_source(old, new, SOURCE_CHECKPOINT_SIZE) == old['source_checkpoint']
    for key, value in [('source_sha256', '0' * 64),
                       ('recipe', {'ms_v_spec_loss_weight': 0.05})]:
        bad = copy.deepcopy(new)
        bad[key] = value
        with pytest.raises(RuntimeError):
            relocated_stage_source(old, bad, SOURCE_CHECKPOINT_SIZE)


@pytest.mark.parametrize('saved_format', [STAGE_FORMAT, LEGACY_STAGE_FORMAT])
def test_stage_format_loads_and_resaves_with_public_identifier(tmp_path, saved_format):
    # Exercise the same stage validation and serialization used by full resume,
    # without allocating the 680M-parameter model or initializing distributed workers.
    trainer = object.__new__(VSpecTrainer)
    trainer.source_ckpt = Path('/portable/ckpts/WaveGen/model_1300000.pt')
    trainer._saved_source_path = None
    trainer._source_checkpoint_size = SOURCE_CHECKPOINT_SIZE
    trainer.total_updates = 30
    trainer._prepared_batches_per_epoch = 10
    trainer.stage_epochs = 3
    trainer.accelerator = SimpleNamespace(num_processes=8, mixed_precision='bf16')
    trainer._resumable_seed = 666
    trainer.training_contract = {'model': {'ms_v_spec_loss_weight': 0.1}}
    trainer._stage_epoch, trainer._next_batch_in_epoch, trainer._stage_update = 0, 3, 3
    trainer.ema = SimpleNamespace(tracking_state=lambda: {'initialized': False})
    stage = trainer._stage_metadata()
    stage['contract']['format'] = saved_format
    payload = {STAGE_KEY: stage, 'update': SOURCE_UPDATE + 3,
               'scheduler_state_dict': trainer._lr_schedule_state()}
    path = tmp_path / 'resume.pt'
    torch.save(payload, path)
    restored = torch.load(path, weights_only=True, map_location='cpu')
    before = copy.deepcopy(restored)
    trainer._stage_epoch, trainer._next_batch_in_epoch, trainer._stage_update = 0, 0, 0
    trainer._validate_stage_cursor(restored, restored[STAGE_KEY])
    assert (trainer._stage_epoch, trainer._next_batch_in_epoch, trainer._stage_update) == (0, 3, 3)
    assert restored == before
    output = tmp_path / 'resaved.pt'
    torch.save(trainer._stage_metadata(), output)
    assert torch.load(output, weights_only=True)['contract']['format'] == 'wavegen_v_spec'


@pytest.mark.parametrize('changed', [
    {'format': 'unknown_training_recipe'},
    {'version': 2},
    {'recipe': {'ms_v_spec_loss_weight': 0.05}},
    {'source_sha256': '0' * 64},
])
def test_legacy_format_does_not_relax_other_contract_checks(changed):
    expected = {'format': STAGE_FORMAT, 'version': 1,
                'source_checkpoint': '/portable/ckpts/WaveGen/model_1300000.pt',
                'source_sha256': SOURCE_CHECKPOINT_SHA256,
                'recipe': {'ms_v_spec_loss_weight': 0.1}}
    saved = copy.deepcopy(expected)
    saved['format'] = LEGACY_STAGE_FORMAT
    saved.update(changed)
    with pytest.raises(RuntimeError):
        relocated_stage_source(saved, expected, SOURCE_CHECKPOINT_SIZE)
