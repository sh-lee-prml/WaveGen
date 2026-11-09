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

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from x_transformers.x_transformers import RotaryEmbedding

from wavegen_v_spec.model.modules import (
    AdaLayerNorm_Final,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    TimestepEmbedding,
    precompute_freqs_cis,
)


# Text embedding


class TextEmbedding(nn.Module):
    def __init__(
        self, text_num_embeds, text_dim, mask_padding=True, average_upsampling=False, conv_layers=0, conv_mult=2
    ):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not
        self.average_upsampling = average_upsampling  # zipvoice-style text late average upsampling (after text encoder)
        if average_upsampling:
            assert mask_padding, "text_embedding_average_upsampling requires text_mask_padding to be True"

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 8192  # 8192 waveform tokens is ~81.92s at 16k / wav_frame_len=160
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def average_upsample_text_by_mask(self, text, text_mask):
        batch, seq_len, text_dim = text.shape

        text_lens = text_mask.sum(dim=1)  # [batch]

        upsampled_text = torch.zeros_like(text)

        for i in range(batch):
            valid_text_len = text_lens[i].item()

            if valid_text_len == 0:
                continue

            valid_ind = torch.where(text_mask[i])[0]
            valid_data = text[i, valid_ind, :]  # [valid_text_len, text_dim]

            base_repeat = seq_len // valid_text_len
            remainder = seq_len % valid_text_len

            indices = []
            for j in range(valid_text_len):
                repeat_count = base_repeat + (1 if j >= valid_text_len - remainder else 0)
                indices.extend([j] * repeat_count)

            indices = torch.tensor(indices[:seq_len], device=text.device, dtype=torch.long)
            upsampled = valid_data[indices]  # [seq_len, text_dim]

            upsampled_text[i, :seq_len, :] = upsampled

        return upsampled_text

    def forward(self, text: int["b nt"], seq_len, drop_text=False):
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than waveform tokens
        text = F.pad(text, (0, seq_len - text.shape[1]), value=0)  # (opt.) if not self.average_upsampling:
        if self.mask_padding:
            text_mask = text == 0

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            text = text + self.freqs_cis[:seq_len, :]

            # convnextv2 blocks
            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text)

        if self.average_upsampling:
            text = self.average_upsample_text_by_mask(text, ~text_mask)

        return text


# noised waveform and context mixing embedding


class MelInformationBottleneck(nn.Module):
    """Dropout-linear-dropout suffix for the noised mel trajectory."""

    def __init__(self, bottleneck_dim: int, dropout: float = 0.5):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"mel bottleneck dropout must be in [0, 1), got {dropout}")

        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(bottleneck_dim, bottleneck_dim, bias=True)
        self.dropout2 = nn.Dropout(dropout)

    @staticmethod
    def _inverted_dropout(
        value: torch.Tensor,
        probability: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if probability == 0.0:
            return value
        if generator is None:
            raise RuntimeError(
                "Inference Mel bottleneck dropout requires a dedicated generator"
            )
        keep = torch.rand(
            value.shape,
            device=value.device,
            dtype=torch.float32,
            generator=generator,
        ).ge(probability)
        return value * keep.to(dtype=value.dtype) / (1.0 - probability)

    def forward(
        self,
        mel: torch.Tensor,
        *,
        inference_dropout_probability: float | None = None,
        inference_dropout_generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if inference_dropout_probability is None:
            mel = self.linear2(self.dropout1(mel))
            return self.dropout2(mel)

        probability = float(inference_dropout_probability)
        if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
            raise ValueError(
                "inference Mel bottleneck dropout must be finite and in [0, 1), "
                f"got {inference_dropout_probability}"
            )
        if self.training:
            raise RuntimeError(
                "Explicit inference Mel bottleneck dropout requires eval mode"
            )
        mel = self._inverted_dropout(
            mel, probability, inference_dropout_generator
        )
        mel = self.linear2(mel)
        return self._inverted_dropout(
            mel, probability, inference_dropout_generator
        )


class InputEmbedding(nn.Module):
    def __init__(
        self,
        wav_frame_len,
        text_dim,
        mel_dim,
        out_dim,
        use_mel_information_bottleneck: bool = False,
        mel_bottleneck_dim: int | None = None,
        mel_bottleneck_dropout: float = 0.5,
        mel_bottleneck_zero_init: bool = False,
        use_audio_proj: bool = False,
        audio_proj_dim: int | None = None,
        audio_proj_hidden: int | None = None,
    ):
        super().__init__()

        self.use_audio_proj = use_audio_proj
        self.use_mel_information_bottleneck = bool(use_mel_information_bottleneck)
        self.mel_bottleneck_dim = int(
            mel_dim if mel_bottleneck_dim is None else mel_bottleneck_dim
        )
        if self.mel_bottleneck_dim <= 0:
            raise ValueError(
                f"mel_bottleneck_dim must be positive, got {self.mel_bottleneck_dim}"
            )
        if self.use_mel_information_bottleneck and self.mel_bottleneck_dim > mel_dim:
            raise ValueError(
                "mel_bottleneck_dim must not exceed mel_channels when the "
                "bottleneck is enabled: "
                f"{self.mel_bottleneck_dim} > {mel_dim}"
            )

        if self.use_mel_information_bottleneck:
            self.mel_proj = nn.Linear(mel_dim, self.mel_bottleneck_dim, bias=False)
            self.mel_bottleneck = MelInformationBottleneck(
                self.mel_bottleneck_dim, dropout=mel_bottleneck_dropout
            )
            mel_xt_dim = self.mel_bottleneck_dim
        else:
            # Identity keeps the disabled path and old state dict unchanged.
            self.mel_proj = nn.Identity()
            self.mel_bottleneck = nn.Identity()
            mel_xt_dim = mel_dim

        self.mel_bottleneck_zero_init = bool(mel_bottleneck_zero_init)
        if self.mel_bottleneck_zero_init:
            if not self.use_mel_information_bottleneck:
                raise ValueError(
                    "mel_bottleneck_zero_init requires "
                    "use_mel_information_bottleneck=True"
                )
            nn.init.zeros_(self.mel_proj.weight)
            nn.init.zeros_(self.mel_bottleneck.linear2.bias)

        if not use_audio_proj:
            self.proj = nn.Linear(
                wav_frame_len * 2 + mel_xt_dim + mel_dim + text_dim,
                out_dim,
            )
        else:
            audio_proj_dim = out_dim if audio_proj_dim is None else audio_proj_dim
            audio_proj_hidden = audio_proj_dim if audio_proj_hidden is None else audio_proj_hidden
            self.x_proj = nn.Sequential(
                nn.Linear(wav_frame_len, audio_proj_hidden, bias=False),
                nn.Linear(audio_proj_hidden, audio_proj_dim),
            )
            self.cond_proj = nn.Sequential(
                nn.Linear(wav_frame_len, audio_proj_hidden, bias=False),
                nn.Linear(audio_proj_hidden, audio_proj_dim),
            )

            self.fuse = nn.Linear(
                audio_proj_dim * 2 + mel_xt_dim + mel_dim + text_dim,
                out_dim,
            )
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(
        self,
        x: float["b n d"],
        cond: float["b n d"],
        mel: float["b n d"],
        mel_cond: float["b n d"],
        text_embed: float["b n d"],
        drop_audio_cond=False,
        audio_mask: bool["b n"] | None = None,
        inference_bottleneck_dropout: float | None = None,
        inference_bottleneck_generator: torch.Generator | None = None,
    ):
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
            mel_cond = torch.zeros_like(mel_cond)

        mel_h = self.mel_proj(mel)
        if self.use_mel_information_bottleneck:
            mel_h = self.mel_bottleneck(
                mel_h,
                inference_dropout_probability=inference_bottleneck_dropout,
                inference_dropout_generator=inference_bottleneck_generator,
            )

        if not self.use_audio_proj:
            h = self.proj(torch.cat((x, cond, mel_h, mel_cond, text_embed), dim=-1))
        else:
            x_h = self.x_proj(x)
            c_h = self.cond_proj(cond)
            h = self.fuse(torch.cat((x_h, c_h, mel_h, mel_cond, text_embed), dim=-1))

        h = self.conv_pos_embed(h, mask=audio_mask) + h
        return h

class DiTBackbone(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        wav_frame_len=160,
        mel_dim=64,
        mel_channels: int | None = None,
        use_mel_information_bottleneck: bool = False,
        mel_bottleneck_dim: int | None = None,
        mel_bottleneck_dropout: float = 0.5,
        mel_bottleneck_zero_init: bool = False,
        text_num_embeds=256,
        text_dim=None,
        text_mask_padding=True,
        text_embedding_average_upsampling=False,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" | "flash_attn"
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
        use_audio_proj: bool = False,
        audio_proj_dim: int | None = None,
        audio_proj_hidden: int | None = None,
    ):
        super().__init__()

        # `mel_dim` remains as a backward-compatible API alias. Configs use
        # `mel_channels` as the single source for forcing input/output width.
        mel_dim = int(mel_dim if mel_channels is None else mel_channels)
        if mel_dim <= 0:
            raise ValueError(f"mel_channels must be positive, got {mel_dim}")

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = wav_frame_len
        self.text_embed = TextEmbedding(
            text_num_embeds,
            text_dim,
            mask_padding=text_mask_padding,
            average_upsampling=text_embedding_average_upsampling,
            conv_layers=conv_layers,
        )
        self.text_cond, self.text_uncond = None, None  # text cache
        self.input_embed = InputEmbedding(
            wav_frame_len, text_dim, mel_dim, dim,
            use_mel_information_bottleneck=use_mel_information_bottleneck,
            mel_bottleneck_dim=mel_bottleneck_dim,
            mel_bottleneck_dropout=mel_bottleneck_dropout,
            mel_bottleneck_zero_init=mel_bottleneck_zero_init,
            use_audio_proj=use_audio_proj,
            audio_proj_dim=audio_proj_dim,
            audio_proj_hidden=audio_proj_hidden,
        )

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth
        self.mel_dim = mel_dim
        self.mel_channels = mel_dim
        self.use_mel_information_bottleneck = bool(use_mel_information_bottleneck)
        self.mel_bottleneck_dim = self.input_embed.mel_bottleneck_dim
        self.mel_bottleneck_dropout = float(mel_bottleneck_dropout)
        self.mel_bottleneck_zero_init = bool(mel_bottleneck_zero_init)

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                )
                for _ in range(depth)
            ]
        )


        self.wav_ddt = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                )
                for _ in range(2)
            ]
        )
        self.mel_ddt = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                )
                for _ in range(2)
            ]
        )

        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
        self.proj_out_dim = wav_frame_len
        self.proj_out = nn.Linear(dim, wav_frame_len)
        self.proj_out_output_layer = self.proj_out

        self.mel_norm_out = AdaLayerNorm_Final(dim)  # final modulation
        self.mel_proj_out_dim = mel_dim
        self.mel_proj_out = nn.Linear(dim, mel_dim)
        self.mel_proj_out_output_layer = self.mel_proj_out

        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

        # raw waveform tokenization.
        self.wav_frame_len = wav_frame_len

    def initialize_weights(self):
        # def _basic_init(module):
        #     if isinstance(module, nn.Linear):
        #         torch.nn.init.xavier_uniform_(module.weight)
        #         if module.bias is not None:
        #             nn.init.constant_(module.bias, 0)

        # self.apply(_basic_init)

        # # Initialize timestep embedding MLP:
        # nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)
        for block in self.wav_ddt:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)
        for block in self.mel_ddt:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out_output_layer.weight, 0)
        if self.proj_out_output_layer.bias is not None:
            nn.init.constant_(self.proj_out_output_layer.bias, 0)

        nn.init.constant_(self.mel_norm_out.linear.weight, 0)
        nn.init.constant_(self.mel_norm_out.linear.bias, 0)
        nn.init.constant_(self.mel_proj_out_output_layer.weight, 0)
        if self.mel_proj_out_output_layer.bias is not None:
            nn.init.constant_(self.mel_proj_out_output_layer.bias, 0)

    def set_wav_frame_len(self, wav_frame_len: int):
        self.wav_frame_len = int(wav_frame_len)

        if self.wav_frame_len != self.proj_out_dim:
            raise ValueError(
                f"wav_frame_len ({self.wav_frame_len}) must equal proj_out_dim ({self.proj_out_dim}) "
                "for reshape wav front-end."
            )

    def _wav_to_tokens(
        self,
        wav: torch.Tensor,
        mask: torch.Tensor | None = None,
        lens: torch.Tensor | None = None,
    ):
        assert wav.ndim == 2, f"Expected [B, N] wav input, got {tuple(wav.shape)}"

        bsz, num_samples = wav.shape
        frame_len = self.wav_frame_len
        pad_len = (frame_len - (num_samples % frame_len)) % frame_len

        if pad_len > 0:
            wav = F.pad(wav, (0, pad_len), value=0.0)
            if mask is not None:
                mask = F.pad(mask, (0, pad_len), value=False)

        tokens = wav.view(bsz, -1, frame_len)

        token_mask = None
        if mask is not None:
            token_mask = mask.view(bsz, -1, frame_len).any(dim=-1)

        token_lens = None
        if lens is not None:
            token_lens = (lens.to(dtype=torch.long, device=wav.device) + frame_len - 1) // frame_len

        return tokens, token_mask, token_lens

    def _tokens_to_wav(self, tokens: torch.Tensor, target_num_samples: int):
        wav = tokens.reshape(tokens.shape[0], -1)
        return wav[:, :target_num_samples]

    def _align_mel_to_tokens(self, mel: torch.Tensor, seq_len: int, name: str) -> torch.Tensor:
        if mel.ndim != 3:
            raise ValueError(f"Expected {name} [B, N, M], got {tuple(mel.shape)}")
        if mel.shape[-1] != self.mel_channels:
            raise ValueError(
                f"Expected {name} with {self.mel_channels} mel channels, "
                f"got {mel.shape[-1]}"
            )
        if mel.shape[1] < seq_len:
            mel = F.pad(mel, (0, 0, 0, seq_len - mel.shape[1]), value=0.0)
        elif mel.shape[1] > seq_len:
            mel = mel[:, :seq_len]
        return mel

    def get_input_embed(
        self,
        x,  # b n d
        cond,  # b n d
        mel,  # b n d
        mel_cond,  # b n d
        text,  # b nt
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cache: bool = True,
        audio_mask: bool["b n"] | None = None,
        inference_bottleneck_dropout: float | None = None,
        inference_bottleneck_generator: torch.Generator | None = None,
    ):
        if self.text_uncond is None or self.text_cond is None or not cache:
            if audio_mask is None:
                text_embed = self.text_embed(text, x.shape[1], drop_text=drop_text)
            else:
                batch = x.shape[0]
                seq_lens = audio_mask.sum(dim=1)  # Calculate the actual sequence length for each sample
                text_embed_list = []
                for i in range(batch):
                    text_embed_i = self.text_embed(
                        text[i].unsqueeze(0),
                        seq_len=seq_lens[i].item(),
                        drop_text=drop_text,
                    )
                    text_embed_list.append(text_embed_i[0])
                text_embed = pad_sequence(text_embed_list, batch_first=True, padding_value=0)
            if cache:
                if drop_text:
                    self.text_uncond = text_embed
                else:
                    self.text_cond = text_embed

        if cache:
            if drop_text:
                text_embed = self.text_uncond
            else:
                text_embed = self.text_cond

        x = self.input_embed(
            x,
            cond,
            mel,
            mel_cond,
            text_embed,
            drop_audio_cond=drop_audio_cond,
            audio_mask=audio_mask,
            inference_bottleneck_dropout=inference_bottleneck_dropout,
            inference_bottleneck_generator=inference_bottleneck_generator,
        )

        return x

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    def forward(
        self,
        x: float["b nw"],  # noised waveform
        cond: float["b nw"],  # masked conditioning waveform
        mel: float["b n m"],  # noised waveform
        mel_cond: float["b n m"],  # masked conditioning waveform
        text: int["b nt"],  # text
        time: float["b"] | float[""],  # time step
        mask: bool["b n"] | bool["b nw"] | None = None,
        drop_audio_cond: bool = False,  # cfg for conditioning waveform
        drop_text: bool = False,  # cfg for text
        cfg_infer: bool = False,  # cfg inference, pack cond & uncond forward
        cache: bool = False,
        lens: int["b"] | None = None,
        inference_bottleneck_dropout: float | None = None,
        inference_bottleneck_generator: torch.Generator | None = None,
        cfg_null_audio: bool = True,  # False preserves audio for a text-only null
        prompt_mask: bool["b nw"] | None = None,
        predict_mel: bool = True,
    ):
        if x.ndim != 2 or cond.ndim != 2:
            raise ValueError(f"WaveGen DiT expects raw waveform x/cond [B, N], got {x.ndim}D and {cond.ndim}D.")
        if prompt_mask is not None and (
            not torch.is_tensor(prompt_mask)
            or prompt_mask.dtype != torch.bool or prompt_mask.shape != x.shape
            or prompt_mask.device != x.device
        ):
            raise ValueError("prompt_mask must match waveform shape/device and have boolean dtype")

        target_num_samples = x.shape[1]
        x, token_mask, token_lens = self._wav_to_tokens(x, mask=mask, lens=lens)
        if prompt_mask is not None:
            prompt_mask, _, _ = self._wav_to_tokens(prompt_mask)
        cond, _, _ = self._wav_to_tokens(cond, mask=None, lens=None)
        mask = token_mask
        lens = token_lens

        batch, seq_len = x.shape[0], x.shape[1]
        mel = self._align_mel_to_tokens(mel, seq_len, "mel")
        mel_cond = self._align_mel_to_tokens(mel_cond, seq_len, "mel_cond")
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, text: text, x: noised waveform + conditioning waveform + text
        t = self.time_embed(time)

        def null_inputs():
            if prompt_mask is None:
                return x, mel
            # Joint-null drops both clean conditioning and the noised prompt;
            # text-only null keeps the reference in both input streams.
            return (
                x.masked_fill(prompt_mask, 0.0),
                mel.masked_fill(prompt_mask.any(dim=-1).unsqueeze(-1), 0.0),
            )

        if cfg_infer:  # pack cond & uncond forward: b n d -> 2b n d
            x_cond = self.get_input_embed(
                x, cond, mel, mel_cond, text, drop_audio_cond=False, drop_text=False,
                cache=cache, audio_mask=mask,
                inference_bottleneck_dropout=inference_bottleneck_dropout,
                inference_bottleneck_generator=inference_bottleneck_generator,
            )
            null_x, null_mel = null_inputs() if cfg_null_audio else (x, mel)
            x_uncond = self.get_input_embed(
                null_x, cond, null_mel, mel_cond, text, drop_audio_cond=cfg_null_audio, drop_text=True,
                cache=cache, audio_mask=mask,
                inference_bottleneck_dropout=inference_bottleneck_dropout,
                inference_bottleneck_generator=inference_bottleneck_generator,
            )
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            if drop_audio_cond and drop_text:
                x, mel = null_inputs()
            x = self.get_input_embed(
                x, cond, mel, mel_cond, text, drop_audio_cond=drop_audio_cond,
                drop_text=drop_text, cache=cache, audio_mask=mask,
                inference_bottleneck_dropout=inference_bottleneck_dropout,
                inference_bottleneck_generator=inference_bottleneck_generator,
            )

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x
            
        for i, block in enumerate(self.transformer_blocks):
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                x = torch.utils.checkpoint.checkpoint(block, x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        mel_x = x
        
        for i, wav_block in enumerate(self.wav_ddt):
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                x = torch.utils.checkpoint.checkpoint(wav_block, x, t, mask, rope, use_reentrant=False)
            else:
                x = wav_block(x, t, mask=mask, rope=rope)

        if predict_mel:
            for i, mel_block in enumerate(self.mel_ddt):
                if self.checkpoint_activations:
                    # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                    mel_x = torch.utils.checkpoint.checkpoint(mel_block, mel_x, t, mask, rope, use_reentrant=False)
                else:
                    mel_x = mel_block(mel_x, t, mask=mask, rope=rope)

        x = self.norm_out(x, t)
        output = self.proj_out(x)
        output = self._tokens_to_wav(output, target_num_samples=target_num_samples)

        mel_output = None
        if predict_mel:
            mel_x = self.mel_norm_out(mel_x, t)
            mel_output = self.mel_proj_out(mel_x)

        return (output, mel_output), None, None

DiT = DiTBackbone

__all__ = ["DiT", "DiTBackbone"]
