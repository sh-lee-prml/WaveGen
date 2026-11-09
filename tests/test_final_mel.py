"""Final Mel-head omission must preserve every waveform Euler state."""
import pytest
import torch

from test_wavegen_v_spec import tiny_generator
from wavegen_v_spec.infer.infer_cli import build_parser, resolve_options
from wavegen_v_spec.eval.eval_infer_batch import build_parser as batch_parser


def compare_heads(model, sample_options, *, expected_calls, skipped=True):
    counts = {}
    handles = []
    modules = dict(trunk=model.transformer.transformer_blocks[0],
                   wav=model.transformer.wav_ddt[0],
                   mel=model.transformer.mel_ddt[0],
                   mel_norm=model.transformer.mel_norm_out,
                   mel_projection=model.transformer.mel_proj_out)
    for name, module in modules.items():
        def hook(_module, _args, _output, name=name):
            counts[name] = counts.get(name, 0) + 1
        handles.append(module.register_forward_hook(hook))
    try:
        full, full_trajectory = model.sample(**sample_options, compute_final_mel=True)
        full_counts = counts.copy()
        counts.clear()
        actual, trajectory = model.sample(**sample_options)
        assert torch.equal(full, actual)
        assert torch.equal(full_trajectory, trajectory)
        assert full_counts['wav'] == expected_calls
        for name in ('trunk', 'wav'):
            assert counts[name] == full_counts[name]
        for name in ('mel', 'mel_norm', 'mel_projection'):
            assert counts.get(name, 0) == full_counts[name] - int(skipped)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize('prediction', ['x_pred', 'flow'])
@pytest.mark.parametrize('schedule,strength', [('constant', 0.0), ('constant', 3.0), ('scheduled_k3', 1.25)])
@pytest.mark.parametrize('method', ['baseline', 'Prompt_xt_Anchor'])
def test_final_mel_preserves_waveform_trajectory(prediction, schedule, strength, method):
    model = tiny_generator()
    model.prediction = prediction
    compare_heads(model, dict(cond=torch.randn(2, 3840) * 0.1,
        text=torch.tensor([[1, 2], [3, 4]]), duration=torch.tensor([7680, 6720]),
        lens=torch.tensor([3840, 2880]), steps=5, seed=42, cfg_schedule=schedule,
        cfg_strength=strength, inference_method=method, mel_temperature=0.5,
        timestep_mapping='power', timestep_power=2.0, shift=3.0), expected_calls=5)


@pytest.mark.parametrize('variant', ['single_step', 'epss', 'duplicate', 'perturb',
                                     'step_size', 'grid_constructor', 'cubic'])
def test_final_mel_solver_variants(variant):
    model = tiny_generator()
    options = dict(cond=torch.randn(1, 3840) * 0.1, text=torch.tensor([[1, 2]]),
                   duration=8640, steps=5, seed=0, cfg_strength=0,
                   inference_method='baseline', mel_temperature=0.0)
    calls, skipped = 5, True
    if variant == 'single_step':
        options['steps'], calls = 1, 1
    elif variant == 'duplicate':
        options.update(duplicate_test=True, t_inter=0.2)
        calls = 4
    elif variant == 'perturb':
        model.odeint_kwargs = {'options': {'perturb': True}}
    elif variant in ('step_size', 'grid_constructor', 'cubic'):
        skipped = False
        if variant == 'step_size':
            model.odeint_kwargs = {'options': {'step_size': 0.125}}
            calls = 8
        elif variant == 'grid_constructor':
            model.odeint_kwargs = {'options': {'grid_constructor': lambda _f, _y, t: t}}
        else:
            model.odeint_kwargs = {'options': {'interp': 'cubic'}}
            calls = 10
    compare_heads(model, options, expected_calls=calls, skipped=skipped)


def test_final_mel_option_requires_boolean():
    with pytest.raises(ValueError, match='compute_final_mel must be a boolean'):
        tiny_generator().sample(cond=None, text=None, duration=1, compute_final_mel='false')


def test_final_mel_cli_and_toml_precedence(tmp_path):
    parser = build_parser()
    assert resolve_options(parser.parse_args([]), parser)['compute_final_mel'] is False
    assert resolve_options(parser.parse_args(['--compute-final-mel']), parser)['compute_final_mel'] is True
    config = tmp_path / 'infer.toml'
    config.write_text('compute_final_mel = true\nref_audio = "ref.wav"\nref_text = "Reference."\ngen_text = "Target."\n')
    assert resolve_options(parser.parse_args(['-c', str(config)]), parser)['compute_final_mel'] is True
    assert resolve_options(parser.parse_args(['-c', str(config), '--no-compute-final-mel']), parser)['compute_final_mel'] is False
    batch = batch_parser()
    args = ['--meta', 'meta.lst', '--output-dir', 'generated']
    assert batch.parse_args(args).compute_final_mel is False
    assert batch.parse_args(args + ['--compute-final-mel']).compute_final_mel is True
    config.write_text('[inference]\nmeta = "meta.lst"\noutput_dir = "generated"\ncompute_final_mel = true\n')
    assert batch.parse_args(['--eval-config', str(config)]).compute_final_mel is True
    assert batch.parse_args(['--eval-config', str(config), '--no-compute-final-mel']).compute_final_mel is False
    config.write_text('[inference]\ncompute_final_mel = "false"\n')
    with pytest.raises(SystemExit):
        batch.parse_args(args + ['--eval-config', str(config)])
