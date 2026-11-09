# Third-party notices

WaveGen contributions are Copyright (c) 2026 Sang-Hoon Lee. This notice
applies to the WaveGen contributions; existing third-party copyrights
and license terms are retained.

The top-level MIT grant does not replace the third-party terms below. This is
an attribution inventory, not blanket license clearance; unresolved provenance
is listed explicitly. Checkpoint and dataset licenses are separate from the
code license. This inventory is retained from the previous WaveGen export,
whose upstream review was dated 2026-09-23; a comparison tag is not proof of
the original copied revision.

## WavTTS

WaveGen reuses WavTTS code and utilities.
Source: https://github.com/cwx-worst-one/WavTTS

MIT License

Copyright (c) 2026 Wenxi Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## F5-TTS

The base code incorporates F5-TTS components, including transformer utilities
and training/data-loading patterns. Source: https://github.com/SWivid/F5-TTS

MIT License

Copyright (c) 2024 Yushen CHEN

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Hugging Face Diffusers — Apache-2.0

`src/wavegen/model/modules.py` contains adapted attention processors and an
MMDiT block, through the F5-TTS lineage. The compared upstream files carry
`Copyright 2024 The HuggingFace Team. All rights reserved.`:

- [attention_processor.py, v0.30.3](https://github.com/huggingface/diffusers/blob/v0.30.3/src/diffusers/models/attention_processor.py)
- [attention.py, v0.30.3](https://github.com/huggingface/diffusers/blob/v0.30.3/src/diffusers/models/attention.py)

The complete [Apache-2.0 license](licenses/Apache-2.0.txt) is included. These
portions were adapted for WaveGen sequence/mask/rotary handling; the copied
module also adds waveform/spectral-forcing components. The exact originating
Diffusers revision is not pinned in the inherited source.

## Other referenced model components — MIT

- `GRN` and `ConvNeXtV2Block` in `modules.py` adapt Meta's ConvNeXt-V2
  normalization/block design to one-dimensional sequences. Copyright (c) Meta
  Platforms, Inc. and affiliates. [Source](https://github.com/facebookresearch/ConvNeXt-V2/tree/main/models),
  [included MIT license](licenses/ConvNeXt-V2-MIT.txt). Upstream explicitly
  [separates MIT code from noncommercial ImageNet model weights](https://github.com/facebookresearch/ConvNeXt-V2#license);
  those image weights are not included here.
- The rotary-frequency utility cites Phil Wang's rotary-embedding-torch.
  Copyright (c) 2021 Phil Wang. [Source](https://github.com/lucidrains/rotary-embedding-torch),
  [included MIT license](licenses/rotary-embedding-torch-MIT.txt).
- `MelSpectrogramLoss` cites Descript AudioTools and DAC; WaveGen uses a
  TorchAudio implementation with waveform-span masking. [AudioTools](https://github.com/descriptinc/audiotools)
  is Copyright (c) 2023-Present, Descript; [DAC](https://github.com/descriptinc/descript-audio-codec)
  is Copyright (c) 2023-present, Descript. Included licenses:
  [AudioTools MIT](licenses/Descript-AudioTools-MIT.txt),
  [DAC MIT](licenses/Descript-Audio-Codec-MIT.txt).

## Inherited references requiring provenance review

The `ConvNeXtV2Block` comment also cites
[bfs18/e2_tts](https://github.com/bfs18/e2_tts/blob/main/rfwave/modules.py).
No root license is displayed by that repository at the audit date. The Meta
ConvNeXt-V2 license above does not establish permission for any distinct
intermediate contribution. Confirm that provenance/permission before asserting
MIT-only clearance for redistribution; this audit does not resolve it.

The inherited DiT previously cited fast-DiT for a closure used by activation
checkpointing. [fast-DiT's license is CC-BY-NC-4.0](https://github.com/chuanyangjin/fast-DiT/blob/main/LICENSE.txt).
That wrapper is not included in this release: the new package passes each
module directly to PyTorch's checkpoint API. This replacement does not establish broader
upstream license clearance; the original research checkout is unchanged.

## ECAPA speaker-similarity evaluator

`src/wavegen/eval/ecapa_tdnn.py` is adapted from Microsoft UniSpeech's speaker
verification implementation, which also credits lawlict/ECAPA-TDNN:

- https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification/models
- https://github.com/lawlict/ECAPA-TDNN

UniSpeech publishes this source under Attribution-ShareAlike 3.0 Unported:
https://github.com/microsoft/UniSpeech/blob/main/LICENSE
https://creativecommons.org/licenses/by-sa/3.0/legalcode

That source and the adapted evaluator remain subject to those terms, not the
top-level MIT grant. Changes here expose explicit feature-extractor injection
and remove implicit remote model loading. The original attribution is retained
in the source file. No upstream endorsement is implied.

The upstream attribution to lawlict is preserved; its linked repository does not
display a separate license file. Review upstream provenance and applicable model
licenses before redistributing evaluation assets. The evaluator is optional and
is not imported by the WaveGen model or inference API.
