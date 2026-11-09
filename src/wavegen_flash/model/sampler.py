"""Inference-only generator and fixed four-step Euler sampler."""

import torch
from torch import nn
from torch.nn import functional as F

from wavegen_v_spec.model.mel_spectrogram import TorchMel
from wavegen_v_spec.model.utils import list_str_to_idx


class FlashModel(nn.Module):
    """WaveGen generator without training objectives or optimizer state."""

    def __init__(self, transformer, vocab_char_map, *, wav_frame_len=480,
                 sample_rate=24000, latents_scale=9.0, t_eps=0.02):
        super().__init__()
        self.transformer = transformer
        self.wav_frame_len = wav_frame_len
        self.target_sample_rate = sample_rate
        self.latents_scale = latents_scale
        self.t_eps = t_eps
        self.vocab_char_map = vocab_char_map
        self.mel_channels = transformer.mel_channels
        self.mel_f = TorchMel(n_fft=wav_frame_len * 4, n_mel_channels=self.mel_channels,
                              target_sample_rate=sample_rate, hop_length=wav_frame_len,
                              win_length=wav_frame_len * 4)

    def _x_to_v(self, x_pred, z, t):
        denom = (1.0 - t).clamp_min(self.t_eps)
        while denom.ndim < z.ndim:
            denom = denom.unsqueeze(-1)
        return (x_pred - z) / denom

    def _pad_wav_to_frame_multiple(self, wav):
        pad_len = (self.wav_frame_len - (wav.shape[-1] % self.wav_frame_len)) % self.wav_frame_len
        return F.pad(wav, (0, pad_len), value=0.0) if pad_len else wav

    def _wav_mask_to_mel_mask(self, wav_mask, mel_len):
        pad_len = (self.wav_frame_len - (wav_mask.shape[-1] % self.wav_frame_len)) % self.wav_frame_len
        if pad_len:
            wav_mask = F.pad(wav_mask, (0, pad_len), value=False)
        mel_mask = wav_mask.view(wav_mask.shape[0], -1, self.wav_frame_len).any(dim=-1)
        if mel_mask.shape[1] < mel_len:
            mel_mask = F.pad(mel_mask, (0, mel_len - mel_mask.shape[1]), value=False)
        return mel_mask[:, :mel_len]

    def _mel_from_wav(self, wav, target_frames):
        mel = self.mel_f(wav.to(dtype=torch.float32)).to(dtype=wav.dtype)
        if mel.shape[1] < target_frames:
            mel = F.pad(mel, (0, 0, 0, target_frames - mel.shape[1]), value=0.0)
        return mel[:, :target_frames]


class FlashSampler:
    """Keep the released time grid and FP32 Euler update order explicit."""

    def __init__(self, model, time_grid):
        if any(isinstance(t, bool) for t in time_grid) or tuple(time_grid) != (0.0, 0.1, 0.25, 0.5, 1.0):
            raise ValueError("WaveGen-Flash requires time_grid=[0, 0.1, 0.25, 0.5, 1]")
        self.model = model
        self.time_grid = tuple(time_grid)
        self.num_steps = 4

    def _euler_step(self, state, endpoint, time, next_time):
        dt = next_time - time
        return tuple(
            value + dt.view(-1, *([1] * (value.ndim - 1)))
            * self.model._x_to_v(clean, value, time)
            for value, clean in zip(state, endpoint)
        )

    @torch.no_grad()
    def generate(self, wav, text, lengths, mask, *, compute_final_mel=False):
        if not isinstance(compute_final_mel, bool):
            raise ValueError("compute_final_mel must be a boolean")
        model = self.model.eval()
        if wav.ndim != 2 or not bool(torch.isfinite(wav).all()):
            raise ValueError("wav must be a finite [batch, samples] tensor")
        wav = model._pad_wav_to_frame_multiple(wav.float())
        batch, size = wav.shape
        lengths = lengths.to(device=wav.device, dtype=torch.long)
        if lengths.shape != (batch,) or bool(((lengths < 2 * model.wav_frame_len) | (lengths > size)).any()):
            raise ValueError("Each length must fit the input and contain at least two waveform frames")
        valid = torch.arange(size, device=wav.device)[None, :] < lengths[:, None]
        if mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[0] != batch or mask.shape[1] > size:
            raise ValueError("mask must be a boolean [batch, samples] generation mask")
        mask = F.pad(mask.to(wav.device), (0, size - mask.shape[1]), value=False)
        target = mask & valid
        known = valid & ~target
        if not bool(target.any(-1).all()) or not bool(known.any(-1).all()):
            raise ValueError("Generation requires nonempty target and reference regions")
        frames = size // model.wav_frame_len
        mel_valid = model._wav_mask_to_mel_mask(valid, frames)
        mel_target = model._wav_mask_to_mel_mask(target, frames)
        mel_known = mel_valid & ~mel_target
        raw = wav.masked_fill(~valid, 0)
        clean_mel = model._mel_from_wav(raw, frames).float().masked_fill(~mel_valid[..., None], 0)
        clean_wav = raw * model.latents_scale
        if not torch.is_tensor(text):
            text = list_str_to_idx(text, model.vocab_char_map).to(wav.device)

        # Both initial states use unit Gaussian noise. The clean reference is
        # interpolated from these same draws at every model evaluation.
        initial = (torch.randn_like(clean_wav), torch.randn_like(clean_mel))
        state = initial
        for step in range(self.num_steps):
            t = self.time_grid[step]
            time = torch.full((batch,), t, device=wav.device, dtype=torch.float32)
            state = (
                torch.where(known, (1 - t) * initial[0] + t * clean_wav, state[0]).masked_fill(~valid, 0),
                torch.where(mel_known[..., None], (1 - t) * initial[1] + t * clean_mel, state[1]).masked_fill(~mel_valid[..., None], 0),
            )
            last = step == self.num_steps - 1
            (pred_wav, pred_mel), *_ = model.transformer(
                x=state[0], mel=state[1], time=time, text=text,
                cond=clean_wav.masked_fill(~known, 0),
                mel_cond=clean_mel.masked_fill(~mel_known[..., None], 0),
                mask=valid, lens=lengths, cache=False, prompt_mask=known,
                drop_audio_cond=False, drop_text=False,
                inference_bottleneck_dropout=0.0,
                predict_mel=compute_final_mel or not last,
            )
            endpoint_wav = torch.where(target, pred_wav.float(), clean_wav).masked_fill(~valid, 0)
            if last and not compute_final_mel:
                # Preserve the explicit Euler operation, even on the last
                # interval: replacing it with the clean prediction can round differently.
                endpoint_wav = self._euler_step((state[0],), (endpoint_wav,), time, 1.0)[0]
                endpoint_wav = torch.where(target, endpoint_wav.float(), clean_wav).masked_fill(~valid, 0)
                return endpoint_wav / model.latents_scale, None
            endpoint_mel = torch.where(mel_target[..., None], pred_mel.float(), clean_mel).masked_fill(~mel_valid[..., None], 0)
            endpoint = (endpoint_wav, endpoint_mel)
            next_time = 1.0 if last else torch.full_like(time, self.time_grid[step + 1])
            state = self._euler_step(state, endpoint, time, next_time)
            if last:
                out_wav = torch.where(target, state[0].float(), clean_wav).masked_fill(~valid, 0)
                out_mel = torch.where(mel_target[..., None], state[1].float(), clean_mel).masked_fill(~mel_valid[..., None], 0)
                return out_wav / model.latents_scale, out_mel
