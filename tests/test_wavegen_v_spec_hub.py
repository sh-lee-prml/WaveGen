"""Hub resolution and CLI tests use mocks only; they never download a model."""

from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from wavegen_v_spec.infer.hub import resolve_hub_assets
from wavegen_v_spec.infer.infer_cli import build_parser, resolve_options
from wavegen_v_spec.infer.utils_infer import WaveGen


SHA = "1234567890abcdef1234567890abcdef12345678"
REPO = "test-owner/test-model"


class HubLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / "WaveGen_VSpec"
        self.bundle.mkdir()
        for name in ("model_1400000.safetensors", "WaveGen_VSpec.yaml", "vocab.txt"):
            (self.bundle / name).write_text("fixture", encoding="utf-8")

    def download(self, **kwargs):
        return str(self.root / kwargs["filename"])

    def options(self, arguments):
        parser = build_parser()
        with redirect_stderr(io.StringIO()):
            return resolve_options(parser.parse_args(arguments), parser)

    def test_package_lazy_export_is_the_inference_api(self):
        from wavegen_v_spec.infer import WaveGen as exported_wavegen

        self.assertIs(exported_wavegen, WaveGen)

    def test_tag_resolves_once_and_every_asset_uses_commit(self):
        with patch("huggingface_hub.model_info", return_value=SimpleNamespace(sha=SHA)) as info, \
             patch("huggingface_hub.hf_hub_download", side_effect=self.download) as download:
            assets = resolve_hub_assets(REPO, revision="release-v1", cache_dir=self.root)
        info.assert_called_once_with(REPO, revision="release-v1")
        self.assertEqual(assets.revision, SHA)
        self.assertEqual(download.call_count, 3)
        for call in download.call_args_list:
            self.assertEqual(call.kwargs["revision"], SHA)
            self.assertEqual(call.kwargs["repo_id"], REPO)
            self.assertEqual(call.kwargs["cache_dir"], self.root)
        self.assertEqual(assets.config, self.bundle / "WaveGen_VSpec.yaml")

    def test_offline_commit_does_not_request_repository_metadata(self):
        with patch("huggingface_hub.model_info") as info, \
             patch("huggingface_hub.hf_hub_download", side_effect=self.download) as download:
            assets = resolve_hub_assets(REPO, revision=SHA.upper(), local_files_only=True)
        info.assert_not_called()
        self.assertEqual(assets.revision, SHA)
        self.assertTrue(all(call.kwargs["local_files_only"] for call in download.call_args_list))

    def test_invalid_or_offline_unpinned_requests_fail_before_network(self):
        cases = [dict(revision=""), dict(revision="main", local_files_only=True),
                 dict(revision=SHA, checkpoint_filename="../model.pt"),
                 dict(revision=SHA, config_filename="C:/model.yaml"),
                 dict(revision=SHA, vocab_filename="/vocab.txt")]
        with patch("huggingface_hub.model_info") as info, patch("huggingface_hub.hf_hub_download") as download:
            for options in cases:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    resolve_hub_assets(REPO, **options)
        info.assert_not_called()
        download.assert_not_called()

    def test_invalid_resolved_commit_is_rejected(self):
        with patch("huggingface_hub.model_info", return_value=SimpleNamespace(sha="main")), \
             patch("huggingface_hub.hf_hub_download") as download, self.assertRaises(ValueError):
            resolve_hub_assets(REPO, revision="main")
        download.assert_not_called()

    def test_missing_hub_asset_does_not_fall_back_to_local_model(self):
        with patch("huggingface_hub.hf_hub_download", side_effect=FileNotFoundError("uncached")), \
             patch.object(WaveGen, "from_checkpoint") as local, self.assertRaises(FileNotFoundError):
            WaveGen.from_pretrained(REPO, revision=SHA, local_files_only=True)
        local.assert_not_called()

    def test_api_delegates_matching_assets_to_unchanged_local_loader(self):
        engine = SimpleNamespace()
        with patch("huggingface_hub.hf_hub_download", side_effect=self.download), \
             patch("huggingface_hub.model_info") as info, \
             patch.object(WaveGen, "from_checkpoint", return_value=engine) as local:
            result = WaveGen.from_pretrained(REPO, revision=SHA, device="cpu", dtype="fp32", use_ema=False)
        info.assert_not_called()
        self.assertIs(result, engine)
        self.assertEqual(result.hub_repo_id, REPO)
        self.assertEqual(result.hub_revision, SHA)
        local.assert_called_once_with(
            self.bundle / "model_1400000.safetensors", self.bundle / "vocab.txt",
            config=self.bundle / "WaveGen_VSpec.yaml", device="cpu", dtype="fp32", use_ema=False,
        )

    def test_cli_local_defaults_are_retained(self):
        options = self.options([])
        self.assertEqual(options["ckpt_file"], "ckpts/WaveGen_VSpec/model_1400000.safetensors")
        self.assertEqual(options["vocab_file"], "ckpts/WaveGen_VSpec/vocab.txt")
        self.assertEqual(options["cfg_schedule"], "scheduled_k3")
        self.assertEqual(options["checkpoint_state"], "ema")
        self.assertNotIn("repo_id", options)

    def test_cli_hub_replaces_only_packaged_local_model_defaults(self):
        options = self.options(["--repo-id", REPO, "--revision", SHA, "--local-files-only"])
        self.assertNotIn("ckpt_file", options)
        self.assertNotIn("vocab_file", options)
        self.assertEqual(options["checkpoint_filename"], "WaveGen_VSpec/model_1400000.safetensors")
        self.assertEqual(options["config_filename"], "WaveGen_VSpec/WaveGen_VSpec.yaml")
        self.assertEqual(options["vocab_filename"], "WaveGen_VSpec/vocab.txt")
        self.assertTrue(options["local_files_only"])
        self.assertEqual(options["cfg_schedule"], "scheduled_k3")
        self.assertEqual(options["cfg_strength"], 1.25)

    def test_explicit_pt_and_flat_hub_assets_override_release_defaults(self):
        options = self.options(["--ckpt-file", "previous.pt"])
        self.assertEqual(options["ckpt_file"], "previous.pt")
        hub_options = self.options([
            "--repo-id", REPO, "--revision", SHA,
            "--checkpoint-filename", "previous.pt", "--config-filename", "config.yaml",
            "--vocab-filename", "vocab.txt",
        ])
        self.assertEqual(hub_options["checkpoint_filename"], "previous.pt")
        self.assertEqual(hub_options["config_filename"], "config.yaml")
        self.assertEqual(hub_options["vocab_filename"], "vocab.txt")
        with patch("huggingface_hub.hf_hub_download", side_effect=self.download) as download, \
             patch.object(WaveGen, "from_checkpoint", return_value=SimpleNamespace()) as local:
            WaveGen.from_pretrained(REPO, revision=SHA, checkpoint_filename="previous.pt",
                                    config_filename="config.yaml", vocab_filename="vocab.txt")
        self.assertEqual([call.kwargs["filename"] for call in download.call_args_list],
                         ["previous.pt", "config.yaml", "vocab.txt"])
        local.assert_called_once_with(self.root / "previous.pt", self.root / "vocab.txt",
                                      config=self.root / "config.yaml", device=None,
                                      dtype="bf16", use_ema=True)

    def test_cli_rejects_ambiguous_or_incomplete_hub_arguments(self):
        cases = [["--repo-id", REPO], ["--revision", SHA],
                 ["--local-files-only"],
                 ["--repo-id", REPO, "--revision", SHA, "--ckpt-file", "local.pt"],
                 ["--repo-id", REPO, "--revision", SHA, "--vocab-file", "local.txt"],
                 ["--repo-id", REPO, "--revision", SHA, "--model-cfg", "local.yaml"]]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(SystemExit):
                self.options(args)

    def test_custom_hub_toml_and_rejection_of_local_overrides(self):
        config = self.root / "hub.toml"
        config.write_text(f'repo_id = "{REPO}"\nrevision = "{SHA}"\n'
                          'ref_audio = "ref.wav"\nref_text = "Reference."\ngen_text = "Target."\n',
                          encoding="utf-8")
        options = self.options(["--config", str(config)])
        self.assertEqual(options["repo_id"], REPO)
        config.write_text(config.read_text() + 'ckpt_file = "local.pt"\n', encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.options(["--config", str(config)])


if __name__ == "__main__":
    unittest.main()
