# Third-party notices

WaveGen contributions are Copyright (c) 2026 Sang-Hoon Lee.
The repository [third-party notices](../../docs/THIRD_PARTY_NOTICES.md) and
included licenses apply to the copied generator, utility, and evaluation code.

The training recipe and spectral objective are adapted from the local WaveGen
implementation. Imports, entry points, and paths are reorganized for this package.
The generator modules retain their existing attention, normalization, and
spectral-loss attributions. `eval/ecapa_tdnn.py` retains its UniSpeech/ECAPA
attribution; it is not imported by the generator or inference API.

Checkpoint and dataset terms are separate from source-code licenses.
