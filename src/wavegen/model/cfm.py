"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

from random import random
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from wavegen import normalize_inference_method
from wavegen.model.mel_spectrogram import TorchMel
from wavegen.model.modules import MelSpectrogramLoss
from wavegen.model.utils import (
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        joint_cond_drop_prob=0.0,
        waveform_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
        prediction: str = "flow",       # "flow" | "x_pred"
        loss_space: str = "flow",       # "flow" | "v" | "x"
        t_sampling: str = "uniform",    # "uniform" | "logistic_normal"
        P_mean: float = 0.0,
        P_std: float = 1.0,
        time_shift: float = 1.0,
        t_eps: float = 1e-4,
        use_aux_mel_loss: bool = False,
        aux_mel_loss_weight: float = 0.0,
        aux_mel_loss_masked: bool = True,
        sample_rate: int = 16000,
        latents_scale: float = 1.0,
        sf_loss_weight: float = 0.1,
        mel_xt_drop_prob: float = 0.0,
        inference_bottleneck_dropout: float = 0.0,
        inference_bottleneck_dropout_seed: int = 0,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask
        
        # waveform geometry
        waveform_kwargs = dict(waveform_kwargs)
        self.wav_frame_len = int(waveform_kwargs.pop("wav_frame_len", 480))
        self.num_channels = self.wav_frame_len

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.joint_cond_drop_prob = joint_cond_drop_prob

        # transformer
        self.transformer = transformer
        self.dim = transformer.dim

        if hasattr(self.transformer, "set_wav_frame_len"):
            self.transformer.set_wav_frame_len(self.wav_frame_len)

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

        # enhanced flow model settings
        self.prediction = prediction
        self.loss_space = loss_space
        self.t_sampling = t_sampling
        self.P_mean = P_mean
        self.P_std = P_std
        self.time_shift = time_shift
        self.t_eps = t_eps
        self.latents_scale = latents_scale
        self.target_sample_rate = sample_rate
        self.sf_loss_weight = float(sf_loss_weight)
        self.mel_xt_drop_prob = float(mel_xt_drop_prob)
        if not 0.0 <= self.mel_xt_drop_prob <= 1.0:
            raise ValueError(f"mel_xt_drop_prob must be in [0, 1], got {self.mel_xt_drop_prob}")
        self.inference_bottleneck_dropout = float(inference_bottleneck_dropout)
        if (
            not math.isfinite(self.inference_bottleneck_dropout)
            or not 0.0 <= self.inference_bottleneck_dropout < 1.0
        ):
            raise ValueError(
                "inference_bottleneck_dropout must be finite and in [0, 1), "
                f"got {inference_bottleneck_dropout}"
            )
        self.inference_bottleneck_dropout_seed = int(
            inference_bottleneck_dropout_seed
        )
        if self.inference_bottleneck_dropout > 0.0 and not bool(
            getattr(transformer, "use_mel_information_bottleneck", False)
        ):
            raise ValueError(
                "inference_bottleneck_dropout > 0 requires "
                "use_mel_information_bottleneck=True"
            )
        self.mel_channels = int(
            getattr(transformer, "mel_channels", getattr(transformer, "mel_proj_out_dim", 64))
        )
        if self.mel_channels <= 0:
            raise ValueError(f"mel_channels must be positive, got {self.mel_channels}")
        self.mel_dim = self.mel_channels  # backward-compatible internal alias
        # aux mel loss
        self.use_aux_mel_loss = use_aux_mel_loss
        self.aux_mel_loss_masked = aux_mel_loss_masked
        if self.use_aux_mel_loss:
            self.aux_mel_loss = MelSpectrogramLoss(
                sample_rate=sample_rate,
                n_mels=[5, 10, 20, 40, 80, 160, 320],
                window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
                mel_fmin=[0, 0, 0, 0, 0, 0, 0],
                mel_fmax=[None] * 7,
                pow=1.0,
                clamp_eps=1e-5,
                weight=aux_mel_loss_weight,
            )
        else:
            self.aux_mel_loss = None

        self.mask_align_to = 1
        alignments = []
        if self.aux_mel_loss is not None and hasattr(self.aux_mel_loss, "mel_transforms"):
            hop_lengths = [int(m.hop_length) for m in self.aux_mel_loss.mel_transforms]
            if len(hop_lengths) > 0:
                lcm_hop = hop_lengths[0]
                for h in hop_lengths[1:]:
                    lcm_hop = math.lcm(lcm_hop, h)
                alignments.append(lcm_hop)
        if len(alignments) > 0:
            lcm_align = alignments[0]
            for a in alignments[1:]:
                lcm_align = math.lcm(lcm_align, a)
            self.mask_align_to = max(1, lcm_align)

        self.mel_f = TorchMel(
            n_fft=self.wav_frame_len * 4,
            n_mel_channels=self.mel_channels,
            target_sample_rate=sample_rate,
            hop_length=self.wav_frame_len,
            win_length=self.wav_frame_len * 4,
        )

    @property
    def device(self):
        return next(self.parameters()).device
    
    def _sample_time(self, batch: int, dtype, device):
        if self.t_sampling == "uniform":
            t = torch.rand((batch,), dtype=dtype, device=device)
        elif self.t_sampling == "logistic_normal":
            # JiT: t = sigmoid(N(P_mean, P_std))
            z = torch.randn((batch,), device=device, dtype=dtype) * self.P_std + self.P_mean
            t = torch.sigmoid(z)
        else:
            raise ValueError(f"Unknown t_sampling: {self.t_sampling}")

        if self.time_shift != 1.0:
            t = t / (t + self.time_shift * (1 - t))

        return t

    def _x_to_v(self, x_pred, z, t):
        # v_pred = (x_pred - z) / (1 - t)
        denom = (1.0 - t).clamp_min(self.t_eps)
        while denom.ndim < z.ndim:
            denom = denom.unsqueeze(-1)
        return (x_pred - z) / denom

    def _ceil_to_frame_multiple(self, lengths: torch.Tensor) -> torch.Tensor:
        lengths = lengths.to(dtype=torch.long)
        return ((lengths + self.wav_frame_len - 1) // self.wav_frame_len) * self.wav_frame_len

    def _pad_wav_to_frame_multiple(self, wav: torch.Tensor) -> torch.Tensor:
        pad_len = (self.wav_frame_len - (wav.shape[-1] % self.wav_frame_len)) % self.wav_frame_len
        if pad_len == 0:
            return wav
        return F.pad(wav, (0, pad_len), value=0.0)

    def _wav_mask_to_mel_mask(self, wav_mask: torch.Tensor, mel_len: int | None = None) -> torch.Tensor:
        pad_len = (self.wav_frame_len - (wav_mask.shape[-1] % self.wav_frame_len)) % self.wav_frame_len
        if pad_len > 0:
            wav_mask = F.pad(wav_mask, (0, pad_len), value=False)

        mel_mask = wav_mask.view(wav_mask.shape[0], -1, self.wav_frame_len).any(dim=-1)
        if mel_len is None:
            return mel_mask
        if mel_mask.shape[1] < mel_len:
            mel_mask = F.pad(mel_mask, (0, mel_len - mel_mask.shape[1]), value=False)
        elif mel_mask.shape[1] > mel_len:
            mel_mask = mel_mask[:, :mel_len]
        return mel_mask

    def _mel_from_wav(self, wav: torch.Tensor, target_frames: int | None = None) -> torch.Tensor:
        mel = self.mel_f(wav.to(dtype=torch.float32)).to(dtype=wav.dtype)
        if target_frames is None:
            return mel
        if mel.shape[1] < target_frames:
            mel = F.pad(mel, (0, 0, 0, target_frames - mel.shape[1]), value=0.0)
        elif mel.shape[1] > target_frames:
            mel = mel[:, :target_frames]
        return mel

    def _unpack_transformer_pred(
        self,
        raw_pred: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(raw_pred, (tuple, list)):
            if len(raw_pred) != 2:
                raise ValueError(f"Expected transformer to return (wav_pred, mel_pred), got {len(raw_pred)} outputs.")
            return raw_pred[0], raw_pred[1]
        raise ValueError("CFM expects transformer to return (wav_pred, mel_pred).")

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str],
        duration: int | int["b"],
        *,
        lens: int["b"] | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        timestep_mapping="sway_sampling",
        timestep_power=None,
        shift=1.0,
        seed: int | None = None,
        mel_temperature: float = 1.0,
        cfg_schedule: str = "constant",
        inference_method: str = "Prompt_xt_Anchor",
        inference_bottleneck_dropout: float | None = None,
        inference_bottleneck_dropout_seed: int | None = None,
        max_duration=4096,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()
        inference_method = normalize_inference_method(inference_method)
        if inference_method == "Prompt_xt_Anchor" and (no_ref_audio or duplicate_test):
            raise ValueError("Prompt_xt_Anchor inference requires reference audio and does not support duplicate_test")
        if not math.isfinite(mel_temperature) or mel_temperature < 0:
            raise ValueError("mel_temperature must be finite and nonnegative")
        if cfg_schedule not in {"constant", "scheduled_k3"}:
            raise ValueError("cfg_schedule must be constant or scheduled_k3")
        if cfg_schedule == "scheduled_k3" and (steps < 3 or cfg_strength < 1e-5):
            raise ValueError("scheduled_k3 requires at least three steps and positive CFG")
        bottleneck_dropout = float(
            self.inference_bottleneck_dropout
            if inference_bottleneck_dropout is None
            else inference_bottleneck_dropout
        )
        if (
            not math.isfinite(bottleneck_dropout)
            or not 0.0 <= bottleneck_dropout < 1.0
        ):
            raise ValueError(
                "inference_bottleneck_dropout must be finite and in [0, 1), "
                f"got {bottleneck_dropout}"
            )
        bottleneck_dropout_seed = int(
            self.inference_bottleneck_dropout_seed
            if inference_bottleneck_dropout_seed is None
            else inference_bottleneck_dropout_seed
        )
        if bottleneck_dropout > 0.0 and not bool(
            getattr(self.transformer, "use_mel_information_bottleneck", False)
        ):
            raise ValueError(
                "inference_bottleneck_dropout > 0 requires "
                "use_mel_information_bottleneck=True"
            )
        active_bottleneck_dropout = (
            bottleneck_dropout if bottleneck_dropout > 0.0 else None
        )

        # raw wave
        if cond.ndim != 2:
            raise ValueError(f"WaveGen expects raw waveform conditioning [B, N], got {tuple(cond.shape)}")

        cond = cond.to(next(self.parameters()).dtype)
        cond_unscaled = cond
        cond = cond_unscaled * self.latents_scale

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        bottleneck_generator = None
        if active_bottleneck_dropout is not None:
            bottleneck_generator = torch.Generator(device=device)
            bottleneck_generator.manual_seed(bottleneck_dropout_seed)
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)
        else:
            lens = lens.to(device=device, dtype=torch.long)
        if inference_method == "Prompt_xt_Anchor":
            if lens.shape != (batch,) or not bool(((lens > 0) & (lens <= cond_seq_len)).all()):
                raise ValueError("Prompt_xt_Anchor inference requires a nonempty valid reference length for each sample")

        # text
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch
        elif torch.is_tensor(text):
            text = text.to(device)

        # duration
        cond_mask = lens_to_mask(lens, length=cond_seq_len)
        if edit_mask is not None:
            if inference_method == "Prompt_xt_Anchor" and (
                not torch.is_tensor(edit_mask)
                or edit_mask.dtype != torch.bool or edit_mask.shape != cond_mask.shape
                or edit_mask.device != device
            ):
                raise ValueError("Prompt_xt_Anchor edit_mask must be a boolean reference mask on the conditioning device")
            cond_mask = cond_mask & edit_mask
        if inference_method == "Prompt_xt_Anchor" and not bool(cond_mask.any(dim=-1).all()):
            raise ValueError("Prompt_xt_Anchor inference requires at least one reference sample per item")

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)
        else:
            duration = duration.to(device=device, dtype=torch.long)

        # keep legacy max_duration semantics for wav path: 4096 means 4096 frame-tokens.
        max_duration_limit = max_duration * self.wav_frame_len
        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = torch.maximum(duration, torch.full_like(duration, cond_seq_len))
        duration = self._ceil_to_frame_multiple(duration)
        duration = duration.clamp(max=max_duration_limit)
        if inference_method == "Prompt_xt_Anchor" and (
            cond_seq_len > int(duration.amax().item()) or bool((duration <= lens).any())
        ):
            raise ValueError("Prompt_xt_Anchor duration must leave room after the reference within max_duration")
        max_duration = int(duration.amax().item())
        mel_duration = duration // self.wav_frame_len
        max_mel_duration = max_duration // self.wav_frame_len

        # duplicate test corner for inner time step oberservation
        test_mel_cond = None
        if duplicate_test:
            test_cond = F.pad(cond, (cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)
            test_mel_cond = self._mel_from_wav(
                test_cond / self.latents_scale,
                target_frames=max_mel_duration,
            ).to(dtype=cond.dtype)

        cond = F.pad(cond, (0, max_duration - cond_seq_len), value=0.0)
        cond_unscaled = F.pad(cond_unscaled, (0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)
            cond_unscaled = torch.zeros_like(cond_unscaled)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
        mel_cond_mask = self._wav_mask_to_mel_mask(cond_mask, mel_len=max_mel_duration)
        if no_ref_audio:
            mel_cond_mask = torch.zeros_like(mel_cond_mask)
        step_mel_cond = self._mel_from_wav(cond_unscaled, target_frames=max_mel_duration)
        step_mel_cond = step_mel_cond.to(dtype=step_cond.dtype)
        step_mel_cond = torch.where(
            mel_cond_mask.unsqueeze(-1),
            step_mel_cond,
            torch.zeros_like(step_mel_cond),
        )

        if batch > 1:
            mask = lens_to_mask(duration, length=max_duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode

        cfg_call_index = 0
        # This is the original length/edit mask, including silent reference samples.
        prompt_kwargs = {"prompt_mask": cond_mask} if inference_method == "Prompt_xt_Anchor" else {}

        def fn(t, state):
            nonlocal cfg_call_index
            x, mel = state
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
            def to_v(raw_pred):
                pred_x_or_v, pred_mel_x_or_v = self._unpack_transformer_pred(raw_pred)
                if self.prediction == "flow":
                    return pred_x_or_v, pred_mel_x_or_v
                else:  # x_pred
                    return self._x_to_v(pred_x_or_v, x, t), self._x_to_v(pred_mel_x_or_v, mel, t)

            # predict flow (cond)
            if cfg_strength < 1e-5:
                raw_pred, *_ = self.transformer(
                    x=x, cond=step_cond, mel=mel, mel_cond=step_mel_cond, text=text, time=t, mask=mask,
                    drop_audio_cond=False, drop_text=False, cache=True,
                    inference_bottleneck_dropout=active_bottleneck_dropout,
                    inference_bottleneck_generator=bottleneck_generator,
                    **prompt_kwargs,
                )
                return to_v(raw_pred)

            # predict flow (cond and uncond), for classifier-free guidance
            scheduled_kwargs = {}
            if cfg_schedule == "scheduled_k3":
                scheduled_kwargs["cfg_null_audio"] = cfg_call_index >= 3
            raw_pred_cfg, *_ = self.transformer(
                x=x, cond=step_cond, mel=mel, mel_cond=step_mel_cond, text=text, time=t, mask=mask,
                cfg_infer=True, cache=True,
                inference_bottleneck_dropout=active_bottleneck_dropout,
                inference_bottleneck_generator=bottleneck_generator,
                **scheduled_kwargs,
                **prompt_kwargs,
            )
            cfg_call_index += 1
            pred_cfg, mel_pred_cfg = self._unpack_transformer_pred(raw_pred_cfg)
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            mel_pred, mel_null_pred = torch.chunk(mel_pred_cfg, 2, dim=0)
            if cfg_schedule == "scheduled_k3" and float(t.item()) < 0.5:
                # Preserve the registered FP32 null-proxy arithmetic followed
                # by the head's original dtype rounding (BF16 on CUDA). Direct
                # doubling of cfg_strength is NOT numerically equivalent.
                null_pred = (pred.float() - 2.0 * (pred.float() - null_pred.float())).to(pred.dtype)
                mel_null_pred = (mel_pred.float() - 2.0 * (mel_pred.float() - mel_null_pred.float())).to(mel_pred.dtype)
            v_cond, mel_v_cond = to_v((pred, mel_pred))
            v_uncond, mel_v_uncond = to_v((null_pred, mel_null_pred))

            return (
                v_cond + (v_cond - v_uncond) * cfg_strength,
                mel_v_cond + (mel_v_cond - mel_v_uncond) * cfg_strength,
            )

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        y0_mel = []
        for dur, mel_dur in zip(duration, mel_duration):
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(int(dur.item()), device=self.device, dtype=step_cond.dtype))
            mel_noise = torch.randn(
                    int(mel_dur.item()),
                    self.mel_dim,
                    device=self.device,
                    dtype=step_mel_cond.dtype,
                )
            # Always consume the same Gaussian draw, including at T=0.
            y0_mel.append(mel_noise if mel_temperature == 1.0 else mel_noise.mul(mel_temperature))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)
        y0_mel = pad_sequence(y0_mel, padding_value=0, batch_first=True)

        # Keep the spectral trajectory prior Gaussian over the full sequence,
        # matching training where phi_mel is noised at every frame.  The clean
        # prompt Mel is already supplied separately through step_mel_cond.

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            y0_mel = (1 - t_start) * y0_mel + t_start * test_mel_cond
            steps = int(steps * (1 - t_start))

        use_epss = use_epss and timestep_mapping == "sway_sampling"

        if t_start == 0 and use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=self.device, dtype=torch.float32)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=torch.float32)

        if timestep_mapping == "uniform":
            pass
        elif timestep_mapping == "sway_sampling":
            if sway_sampling_coef is not None:
                t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        elif timestep_mapping == "power":
            if timestep_power is None:
                raise ValueError("timestep_power must be provided when timestep_mapping='power'")
            t = t.pow(timestep_power)
        else:
            raise ValueError(f"Unknown timestep_mapping: {timestep_mapping}")

        effective_shift = shift
        if effective_shift != 1.0:
            t = t / (t + effective_shift * (1 - t))

        # WaveGen inference uses Euler ODE sampling.
        odeint_kwargs = dict(self.odeint_kwargs)
        odeint_kwargs["method"] = "euler"
        rhs = fn
        if inference_method == "Prompt_xt_Anchor":
            initial = (y0.clone(), y0_mel.clone())
            clean = (step_cond, step_mel_cond)
            prompt = (cond_mask, mel_cond_mask.unsqueeze(-1).expand_as(y0_mel))

            def rhs(t, state):
                # Keep the known prompt on its training interpolation at every
                # model call; use the original post-temperature Gaussian draws.
                corrected = tuple(
                    torch.where(prompt[h], (1.0 - t) * initial[h] + t * clean[h], state[h])
                    for h in range(2)
                )
                velocity = fn(t, corrected)
                return tuple(
                    torch.where(prompt[h], clean[h] - initial[h], velocity[h])
                    for h in range(2)
                )

        trajectory = odeint(rhs, (y0, y0_mel), t, **odeint_kwargs)
        self.transformer.clear_cache()

        wav_trajectory, _ = trajectory
        sampled = wav_trajectory[-1]
        out = sampled

        out = out / self.latents_scale
        cond_unscaled = cond / self.latents_scale
        out = torch.where(cond_mask, cond_unscaled, out)

        # ---- wav-only: trim to generated waveform duration ----
        wav_list = []
        for b in range(batch):
            wav_list.append(out[b, : duration[b].item()])
        out = pad_sequence(wav_list, batch_first=True, padding_value=0.0)  # [B, N]

        return out, wav_trajectory

    def forward(
        self,
        inp: float["b nw"],  # raw waveform
        text: int["b nt"] | list[str],
        *,
        lens: int["b"] | None = None,
        noise_scheduler: str | None = None,
    ):
        # handle raw waveform
        if inp.ndim != 2:
            raise ValueError(f"WaveGen expects raw waveform input [B, N], got {tuple(inp.shape)}")

        inp = self._pad_wav_to_frame_multiple(inp)
        batch, seq_len, dtype, device, _sigma1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch
        elif torch.is_tensor(text):
            text = text.to(device)

        # lens and mask
        if not exists(lens):  # if lens not acquired by trainer from collate_fn
            lens = torch.full((batch,), seq_len, device=device)
        else:
            lens = lens.to(device=device, dtype=torch.long)
        mask = lens_to_mask(lens, length=seq_len)

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        if self.mask_align_to > 1:
            rand_span_mask = MelSpectrogramLoss._aligned_random_span_mask(
                lengths=lens,
                frac_lengths=frac_lengths,
                align_to=self.mask_align_to,
                max_length=seq_len,
            )
        else:
            rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        # x1 is raw waveform
        x1 = inp
        mel_frames = seq_len // self.wav_frame_len
        x1_mel = self._mel_from_wav(x1, target_frames=mel_frames)

        x1 = x1 * self.latents_scale

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)
        x0_mel = torch.randn_like(x1_mel)

        # time step
        time = self._sample_time(batch, dtype=dtype, device=self.device)

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        φ_mel = (1 - t.unsqueeze(-1)) * x0_mel + t.unsqueeze(-1) * x1_mel

        flow = x1 - x0
        mel_flow = x1_mel - x0_mel

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask, torch.zeros_like(x1), x1)

        mel_rand_span_mask = self._wav_mask_to_mel_mask(rand_span_mask, mel_len=mel_frames)
        mel_cond = torch.where(mel_rand_span_mask.unsqueeze(-1), torch.zeros_like(x1_mel), x1_mel)

        # transformer and cfg training with a drop rate
        if self.joint_cond_drop_prob > 0.0:
            joint_drop = random() < self.joint_cond_drop_prob
            drop_audio_cond = joint_drop
            drop_text = joint_drop
        else:
            drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
            if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
                drop_audio_cond = True
                drop_text = True
            else:
                drop_text = False

        # Modality dropout prevents the shared trunk from always taking a shortcut
        # through the noised mel trajectory. Keep the original φ_mel for prediction
        # interpretation and targets; only its transformer input copy is dropped.
        mel_xt_drop_mask = torch.zeros(batch, dtype=torch.bool, device=device)
        transformer_mel = φ_mel
        if self.mel_xt_drop_prob > 0.0:
            mel_xt_drop_mask = torch.rand(batch, device=device) < self.mel_xt_drop_prob
            transformer_mel = φ_mel.masked_fill(mel_xt_drop_mask[:, None, None], 0.0)

        # apply mask will use more memory; might adjust batchsize or batchsampler long sequence threshold
        raw_pred, *_ = self.transformer(
            x=φ, cond=cond, mel=transformer_mel, mel_cond=mel_cond, text=text, time=time,
            drop_audio_cond=drop_audio_cond, drop_text=drop_text, mask=mask,
            lens=lens,
        )

        pred_wav_raw, pred_mel_raw = self._unpack_transformer_pred(raw_pred)

        # interpret prediction
        if self.prediction == "flow":
            v_pred, mel_v_pred = pred_wav_raw, pred_mel_raw
            x_pred = φ + (1.0 - t) * v_pred
            mel_x_pred = φ_mel + (1.0 - t.unsqueeze(-1)) * mel_v_pred

        elif self.prediction == "x_pred":
            x_pred, mel_x_pred = pred_wav_raw, pred_mel_raw
            v_pred = self._x_to_v(x_pred, φ, time)
            mel_v_pred = self._x_to_v(mel_x_pred, φ_mel, time.unsqueeze(-1))

        else:
            raise ValueError(f"Unknown prediction: {self.prediction}")

        # loss space
        if self.loss_space == "flow":
            loss = F.mse_loss(v_pred, flow, reduction="none")
            sf_loss = F.mse_loss(mel_v_pred, mel_flow, reduction="none")
        elif self.loss_space == "v":
            # v-loss (same target flow, but v_pred computed from x_pred) & use clamp_min
            denom = (1.0 - time).clamp_min(self.t_eps)
            while denom.ndim < φ.ndim:
                denom = denom.unsqueeze(-1)
            target = (x1 - φ) / denom
            mel_target = (x1_mel - φ_mel) / denom.unsqueeze(-1)
            loss = F.mse_loss(v_pred, target, reduction="none")
            sf_loss = F.mse_loss(mel_v_pred, mel_target, reduction="none")
        elif self.loss_space == "x":
            loss = F.mse_loss(x_pred, x1, reduction="none")
            sf_loss = F.mse_loss(mel_x_pred, x1_mel, reduction="none")
        else:
            raise ValueError(f"Unknown loss_space: {self.loss_space}")

        loss = loss[rand_span_mask]
        flow_loss = loss.mean()
        total_loss = flow_loss

        # A dropped sample trains the waveform path without spectral forcing.
        # Excluding it here also keeps flow-prediction modes from being asked to
        # recover the independently sampled x0_mel without observing φ_mel.
        sf_loss_mask = mel_rand_span_mask & ~mel_xt_drop_mask[:, None]
        sf_loss = sf_loss[sf_loss_mask]
        sf_loss = sf_loss.mean() if sf_loss.numel() > 0 else flow_loss.new_zeros(())
        total_loss = total_loss + sf_loss * self.sf_loss_weight

        aux_mel_loss = torch.tensor(0.0, device=device)
        if self.use_aux_mel_loss and self.aux_mel_loss is not None:
            x1_flat_unscaled = x1 / self.latents_scale
            x1_pred_flat_unscaled = x_pred / self.latents_scale

            aux_mel_kwargs = {}
            if self.aux_mel_loss_masked:
                aux_mel_kwargs.update(
                    frame_mask=rand_span_mask,
                    frame_lengths=lens,
                )

            aux_mel_loss = self.aux_mel_loss(
                x1_pred_flat_unscaled,
                x1_flat_unscaled,
                **aux_mel_kwargs,
            )
            total_loss = total_loss + aux_mel_loss
            
        loss_dict = {
            "total_loss": total_loss,
            "flow_loss": flow_loss,
            "aux_mel_loss": aux_mel_loss,
            "sf_loss": sf_loss,
            "mel_xt_drop_fraction": mel_xt_drop_mask.float().mean(),
        }

        return total_loss, cond, v_pred, loss_dict
