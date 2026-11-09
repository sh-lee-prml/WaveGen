from __future__ import annotations

import torch
import torchaudio

class TorchMel(torch.nn.Module):
    def __init__(self, 
                n_fft=640,
                n_mel_channels=64,
                target_sample_rate=16000,
                hop_length=160,
                win_length=640):
        super().__init__()

        self.mel_stft = torchaudio.transforms.MelSpectrogram(
                sample_rate=target_sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mel_channels,
                power=1,
                center=True,
                normalized=False,
                norm=None,
            )
 
    def forward(self, waveform):
        if len(waveform.shape) == 3:
            waveform = waveform.squeeze(1)  # 'b 1 nw -> b nw'
        assert len(waveform.shape) == 2
        mel = self.mel_stft(waveform)
        mel = mel.clamp(min=1e-5).log()

        return mel[:,:,:-1].transpose(1,2)
