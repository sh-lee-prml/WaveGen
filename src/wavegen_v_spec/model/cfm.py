"""SF with WaveGen's exact MS-V-Spec loss and no auxiliary Mel loss.

The historical name MS-V-Spec denotes a variance-normalized *waveform-error*
spectrogram loss. Its inputs are unscaled clean x predictions/targets, not
velocities. The original waveform v-MSE and Mel spectral-forcing objective,
including Mel-trajectory dropout, remain unchanged. This branch has no
discriminator, adversarial objective, feature matching, or crop payload.

The loss has only deterministic nonpersistent analysis buffers. Generator
parameter keys and non-auxiliary buffers therefore match the original SF CFM;
legacy checkpoints containing aux_mel_loss.* buffers require the explicit
training-entrypoint migration before strict loading.
"""

from __future__ import annotations

import math
from random import random

import torch
import torch.nn.functional as F

from wavegen_v_spec.model.spectral_loss import (
    MULTISCALE_SPECTROGRAM_BANDS,
    MULTISCALE_SPECTROGRAM_HOPS,
    MULTISCALE_SPECTROGRAM_WINDOWS,
    MultiScaleLinearSpectrogramLoss,
    build_multiscale_spectrogram_masks,
)
from wavegen_v_spec.model.base_cfm import BaseCFM
from wavegen_v_spec.model.modules import MelSpectrogramLoss
from wavegen_v_spec.model.utils import (
    exists,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
)


class CFMMSVSpec(BaseCFM):
    """Isolated MS-V-Spec SF branch; aux-Mel cannot be re-enabled."""

    def __init__(
        self,
        *args,
        use_aux_mel_loss: bool = False,
        aux_mel_loss_weight: float = 0.0,
        use_ms_v_spec_loss: bool = True,
        ms_v_spec_loss_weight: float = 0.05,
        **kwargs,
    ):
        if use_aux_mel_loss is not False or float(aux_mel_loss_weight) != 0.0:
            raise ValueError("This branch requires use_aux_mel_loss=False and aux_mel_loss_weight=0")
        weight = float(ms_v_spec_loss_weight)
        if use_ms_v_spec_loss is not True or not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("This branch requires use_ms_v_spec_loss=True and a finite positive weight")
        super().__init__(
            *args,
            use_aux_mel_loss=False,
            aux_mel_loss_weight=0.0,
            **kwargs,
        )
        self.use_ms_v_spec_loss = True
        self.ms_v_spec_loss_weight = weight
        self.spectrogram_loss_hop_lengths = MULTISCALE_SPECTROGRAM_HOPS
        self.mask_align_to = math.lcm(*self.spectrogram_loss_hop_lengths)
        self.ms_v_spec_loss = MultiScaleLinearSpectrogramLoss(
            sample_rate=self.target_sample_rate,
            n_filter=MULTISCALE_SPECTROGRAM_BANDS,
            n_fft=MULTISCALE_SPECTROGRAM_WINDOWS,
            win_length=MULTISCALE_SPECTROGRAM_WINDOWS,
            hop_length=MULTISCALE_SPECTROGRAM_HOPS,
        )

    def forward(
        self,
        inp: torch.Tensor,
        text: torch.Tensor | list[str],
        *,
        lens: torch.Tensor | None = None,
        noise_scheduler: str | None = None,
    ):
        """Preserve the SF forward, substituting only the auxiliary objective."""
        if inp.ndim != 2:
            raise ValueError(f"WaveGen expects raw waveform input [B, N], got {tuple(inp.shape)}")

        inp = self._pad_wav_to_frame_multiple(inp)
        batch, seq_len, dtype, device, _sigma1 = (
            *inp.shape[:2], inp.dtype, self.device, self.sigma,
        )
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch
        elif torch.is_tensor(text):
            text = text.to(device)

        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)
        else:
            lens = lens.to(device=device, dtype=torch.long)
        mask = lens_to_mask(lens, length=seq_len)
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = MelSpectrogramLoss._aligned_random_span_mask(
            lengths=lens,
            frac_lengths=frac_lengths,
            align_to=self.mask_align_to,
            max_length=seq_len,
        )
        rand_span_mask &= mask
        # One canonical generated-span mask, projected once onto all seven
        # centered-STFT grids exactly as in wavegen_v_spec.model.cfm.CFM.forward.
        spectrogram_scale_masks = build_multiscale_spectrogram_masks(
            rand_span_mask, lens, self.spectrogram_loss_hop_lengths,
        )

        x1 = inp
        mel_frames = seq_len // self.wav_frame_len
        x1_mel = self._mel_from_wav(x1, target_frames=mel_frames)
        x1 = x1 * self.latents_scale
        x0 = torch.randn_like(x1)
        x0_mel = torch.randn_like(x1_mel)
        time = self._sample_time(batch, dtype=dtype, device=self.device)
        t = time.unsqueeze(-1)
        phi = (1 - t) * x0 + t * x1
        phi_mel = (1 - t.unsqueeze(-1)) * x0_mel + t.unsqueeze(-1) * x1_mel
        flow = x1 - x0
        mel_flow = x1_mel - x0_mel
        cond = torch.where(rand_span_mask, torch.zeros_like(x1), x1)
        mel_rand_span_mask = self._wav_mask_to_mel_mask(rand_span_mask, mel_len=mel_frames)
        mel_cond = torch.where(mel_rand_span_mask.unsqueeze(-1), torch.zeros_like(x1_mel), x1_mel)

        if self.joint_cond_drop_prob > 0.0:
            joint_drop = random() < self.joint_cond_drop_prob
            drop_audio_cond = joint_drop
            drop_text = joint_drop
        else:
            drop_audio_cond = random() < self.audio_drop_prob
            if random() < self.cond_drop_prob:
                drop_audio_cond = True
                drop_text = True
            else:
                drop_text = False

        mel_xt_drop_mask = torch.zeros(batch, dtype=torch.bool, device=device)
        transformer_mel = phi_mel
        if self.mel_xt_drop_prob > 0.0:
            mel_xt_drop_mask = torch.rand(batch, device=device) < self.mel_xt_drop_prob
            transformer_mel = phi_mel.masked_fill(mel_xt_drop_mask[:, None, None], 0.0)
        raw_pred, *_ = self.transformer(
            x=phi,
            cond=cond,
            mel=transformer_mel,
            mel_cond=mel_cond,
            text=text,
            time=time,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
            mask=mask,
            lens=lens,
        )
        pred_wav_raw, pred_mel_raw = self._unpack_transformer_pred(raw_pred)
        if self.prediction == "flow":
            v_pred, mel_v_pred = pred_wav_raw, pred_mel_raw
            x_pred = phi + (1.0 - t) * v_pred
            mel_x_pred = phi_mel + (1.0 - t.unsqueeze(-1)) * mel_v_pred
        elif self.prediction == "x_pred":
            x_pred, mel_x_pred = pred_wav_raw, pred_mel_raw
            v_pred = self._x_to_v(x_pred, phi, time)
            mel_v_pred = self._x_to_v(mel_x_pred, phi_mel, time.unsqueeze(-1))
        else:
            raise ValueError(f"Unknown prediction: {self.prediction}")

        if self.loss_space == "flow":
            loss = F.mse_loss(v_pred, flow, reduction="none")
            sf_loss = F.mse_loss(mel_v_pred, mel_flow, reduction="none")
        elif self.loss_space == "v":
            denom = (1.0 - time).clamp_min(self.t_eps)
            while denom.ndim < phi.ndim:
                denom = denom.unsqueeze(-1)
            target = (x1 - phi) / denom
            mel_target = (x1_mel - phi_mel) / denom.unsqueeze(-1)
            loss = F.mse_loss(v_pred, target, reduction="none")
            sf_loss = F.mse_loss(mel_v_pred, mel_target, reduction="none")
        elif self.loss_space == "x":
            loss = F.mse_loss(x_pred, x1, reduction="none")
            sf_loss = F.mse_loss(mel_x_pred, x1_mel, reduction="none")
        else:
            raise ValueError(f"Unknown loss_space: {self.loss_space}")

        flow_loss = loss[rand_span_mask].mean()
        sf_loss_mask = mel_rand_span_mask & ~mel_xt_drop_mask[:, None]
        sf_loss = sf_loss[sf_loss_mask]
        sf_loss = sf_loss.mean() if sf_loss.numel() > 0 else flow_loss.new_zeros(())
        total_loss = flow_loss + sf_loss * self.sf_loss_weight

        # Exact WaveGen convention: STFT(x_pred - x1), target-normalized,
        # with no division by (1-t), additional time weight, or log-Mel term.
        ms_v_spec_loss = self.ms_v_spec_loss(
            x_pred / self.latents_scale,
            x1 / self.latents_scale,
            scale_masks=spectrogram_scale_masks,
        )
        ms_v_spec_loss_contribution = ms_v_spec_loss * self.ms_v_spec_loss_weight
        total_loss = total_loss + ms_v_spec_loss_contribution
        loss_dict = {
            "total_loss": total_loss,
            "flow_loss": flow_loss,
            "aux_mel_loss": flow_loss.new_zeros(()),
            "sf_loss": sf_loss,
            "mel_xt_drop_fraction": mel_xt_drop_mask.float().mean(),
            "ms_v_spec_loss": ms_v_spec_loss,
            "ms_v_spec_loss_contribution": ms_v_spec_loss_contribution,
        }
        return total_loss, cond, v_pred, loss_dict


__all__ = ["CFMMSVSpec"]

CFM = CFMMSVSpec
