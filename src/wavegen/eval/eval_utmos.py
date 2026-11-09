"""UTMOS-only convenience entry point for a complete Seed-TTS condition."""

import sys
from .eval_seedtts_testset import main


if __name__ == "__main__":
    main([*sys.argv[1:], "--eval-task", "utmos"])
