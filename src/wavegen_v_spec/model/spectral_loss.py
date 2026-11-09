"""WaveGen's variance-normalized waveform-error spectrogram objective."""
from __future__ import annotations

from typing import Optional, Sequence
import torch
import torchaudio
from torch import nn
from wavegen_v_spec.model.modules import MelSpectrogramLoss

MULTISCALE_SPECTROGRAM_BANDS = (5, 10, 20, 40, 80, 160, 320)
MULTISCALE_SPECTROGRAM_WINDOWS = (32, 64, 128, 256, 512, 1024, 2048)
MULTISCALE_SPECTROGRAM_HOPS = tuple(window // 4 for window in MULTISCALE_SPECTROGRAM_WINDOWS)

def build_multiscale_spectrogram_masks(
    frame_mask: torch.Tensor,
    frame_lengths: torch.Tensor | None = None,
    hop_lengths: Sequence[int] = MULTISCALE_SPECTROGRAM_HOPS,
) -> tuple[torch.Tensor, ...]:
    """Project one waveform mask onto all centered-STFT time grids once.

    ``frame_mask`` remains the single canonical random mask.  Both the Mel and
    linear-filter losses consume the returned tensors, so no loss-specific
    random mask or duplicate span-to-grid conversion is needed.
    """
    if frame_mask.ndim != 2:
        raise ValueError(f"frame_mask must have shape [B, T], got {tuple(frame_mask.shape)}")

    frame_mask = frame_mask.bool()
    span_starts, span_ends = MelSpectrogramLoss._get_span_bounds_from_mask(frame_mask, frame_lengths)
    num_samples = int(frame_mask.shape[-1])
    scale_masks = []

    for hop_length in hop_lengths:
        hop_length = int(hop_length)
        if hop_length <= 0:
            raise ValueError(f"hop lengths must be positive, got {hop_length}")

        # Both transforms use center=True and pad=0, hence floor(T / hop) + 1.
        num_frames = num_samples // hop_length + 1
        scale_starts = torch.div(span_starts, hop_length, rounding_mode="floor")
        scale_ends = torch.div(span_ends + hop_length - 1, hop_length, rounding_mode="floor")
        positions = torch.arange(num_frames, device=frame_mask.device).unsqueeze(0)
        scale_masks.append((positions >= scale_starts.unsqueeze(1)) & (positions < scale_ends.unsqueeze(1)))

    return tuple(scale_masks)

class LinearFilterSpectrogram(nn.Module):
    """Power spectrogram projected onto uniformly spaced frequency filters."""

    def __init__(
        self,
        sample_rate: int,
        n_filter: int,
        n_fft: int,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        power: float = 2.0,
        normalized: bool = False,
        center: bool = True,
        pad_mode: str = "reflect",
    ):
        super().__init__()

        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length if win_length is not None else n_fft)
        self.hop_length = int(hop_length if hop_length is not None else self.win_length // 2)
        self.pad = int(pad)
        self.power = float(power)
        self.normalized = normalized
        self.n_filter = int(n_filter)
        self.f_min = float(f_min)
        self.f_max = float(f_max if f_max is not None else sample_rate // 2)

        if self.f_min > self.f_max:
            raise ValueError(f"Require f_min {self.f_min} <= f_max {self.f_max}")

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            pad=self.pad,
            power=self.power,
            normalized=self.normalized,
            center=center,
            pad_mode=pad_mode,
            onesided=True,
        )
        # These analysis kernels are deterministic from config and should not
        # make checkpoint loading depend on whether the optional loss is on.
        self.spectrogram.register_buffer(
            "window",
            self.spectrogram.window,
            persistent=False,
        )
        fb = torchaudio.functional.linear_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=self.f_min,
            f_max=self.f_max,
            n_filter=self.n_filter,
            sample_rate=self.sample_rate,
        )
        self.register_buffer("fb", fb, persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform = waveform.squeeze(1)
        specgram = self.spectrogram(waveform)
        # [B, F, T] -> [B, T, K].  Do not drop the final centered-STFT frame:
        # its time grid intentionally matches MelSpectrogramLoss exactly.
        return torch.matmul(specgram.transpose(-1, -2), self.fb)

class MultiScaleLinearSpectrogramLoss(nn.Module):
    """Multi-scale variance-normalized spectrogram loss on waveform error."""

    def __init__(
        self,
        sample_rate: int = 24000,
        n_filter: Sequence[int] = MULTISCALE_SPECTROGRAM_BANDS,
        n_fft: Sequence[int] = MULTISCALE_SPECTROGRAM_WINDOWS,
        win_length: Sequence[int] = MULTISCALE_SPECTROGRAM_WINDOWS,
        hop_length: Sequence[int] = MULTISCALE_SPECTROGRAM_HOPS,
        clamp_min: float = 1e-2,
        clamp_max: float = 1e2,
    ):
        super().__init__()
        if not (len(n_filter) == len(n_fft) == len(win_length) == len(hop_length)):
            raise ValueError("n_filter, n_fft, win_length, and hop_length must have the same length")

        self.sample_rate = int(sample_rate)
        self.n_filters = tuple(int(value) for value in n_filter)
        self.n_ffts = tuple(int(value) for value in n_fft)
        self.window_lengths = tuple(int(value) for value in win_length)
        self.hop_lengths = tuple(int(value) for value in hop_length)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.loss_spec = nn.ModuleList(
            [
                LinearFilterSpectrogram(
                    sample_rate=self.sample_rate,
                    n_filter=nf,
                    n_fft=nf_fft,
                    win_length=wl,
                    hop_length=hl,
                )
                for nf, nf_fft, wl, hl in zip(
                    self.n_filters,
                    self.n_ffts,
                    self.window_lengths,
                    self.hop_lengths,
                )
            ]
        )

    def forward(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        frame_lengths: torch.Tensor | None = None,
        scale_masks: Sequence[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if x_pred.ndim == 3 and x_pred.shape[1] == 1:
            x_pred = x_pred.squeeze(1)
        if x_true.ndim == 3 and x_true.shape[1] == 1:
            x_true = x_true.squeeze(1)
        if x_pred.shape != x_true.shape:
            raise ValueError(f"x_pred and x_true must match, got {x_pred.shape} and {x_true.shape}")
        if scale_masks is not None and frame_mask is not None:
            raise ValueError("Pass either scale_masks or frame_mask, not both")
        if scale_masks is None and frame_mask is not None:
            scale_masks = build_multiscale_spectrogram_masks(
                frame_mask,
                frame_lengths,
                self.hop_lengths,
            )
        if scale_masks is not None and len(scale_masks) != len(self.loss_spec):
            raise ValueError(f"Expected {len(self.loss_spec)} scale masks, got {len(scale_masks)}")

        total_loss = x_pred.new_zeros(())
        error_waveform = x_pred - x_true
        active_scales = 0

        for scale_index, spec_fn in enumerate(self.loss_spec):
            # This is STFT(x_true) and STFT(x_pred - x_true), deliberately not
            # a difference between independently transformed spectrograms.
            gt_spec = spec_fn(x_true)
            error_spec = spec_fn(error_waveform)
            spec_scale = ((gt_spec + 1e-7) ** -0.5).clamp(
                min=self.clamp_min,
                max=self.clamp_max,
            )
            loss = error_spec * spec_scale

            if scale_masks is not None:
                spec_mask = scale_masks[scale_index].to(device=loss.device, dtype=torch.bool)
                expected_shape = (loss.shape[0], loss.shape[1])
                if tuple(spec_mask.shape) != expected_shape:
                    raise ValueError(
                        f"Scale {scale_index} mask shape {tuple(spec_mask.shape)} "
                        f"does not match linear time grid {expected_shape}"
                    )
                if not spec_mask.any():
                    continue
                loss = loss[spec_mask]

            # Match MelSpectrogramLoss accumulation: preserve the waveform
            # dtype for the returned scalar under mixed-precision autocast.
            total_loss += loss.mean()
            active_scales += 1

        if active_scales == 0:
            return total_loss
        return total_loss
