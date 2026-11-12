"""Checks for the separate fine-tuning experiment runner (no GPU needed)."""

import csv
import importlib.util
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_finetune_experiments.py"
spec = importlib.util.spec_from_file_location("run_finetune_experiments", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class TestFinetuneExperiments(unittest.TestCase):
    def test_each_attempt_uses_original_pretrained_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pretrained = root / "general_2d"
            first = runner.make_training_config({"batch_size": 1, "resume": True},
                                                pretrained, root / "attempt_01",
                                                2, 16, 1e-5, 42)
            second = runner.make_training_config({"batch_size": 1, "resume": True},
                                                 pretrained, root / "attempt_02",
                                                 5, 89, 5e-6, 42)
            self.assertEqual(first["model"], second["model"])
            self.assertEqual(first["model"], str(pretrained))
            self.assertFalse(first["resume"])
            self.assertFalse(second["resume"])
            self.assertNotEqual(first["outdir"], second["outdir"])
            self.assertEqual((first["epochs"], second["epochs"]), (2, 5))

    def test_best_validation_counts_actual_epochs(self):
        from torch.utils.tensorboard import SummaryWriter

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with SummaryWriter(str(root / "tb" / "version_0")) as writer:
                for step, loss in ((10, 0.8), (20, 0.2), (30, 0.4)):
                    writer.add_scalar("val_loss", loss, step)
            actual, best_epoch, loss = runner.best_validation(root)
            self.assertEqual((actual, best_epoch), (3, 2))
            self.assertTrue(math.isclose(loss, 0.2, rel_tol=1e-6))

    def test_metrics_and_result_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_file = root / "metrics.csv"
            metrics_file.write_text("TRA;AOGM;CT\n0.7;12;0.8\n", encoding="utf-8")
            self.assertEqual(runner.read_metrics(metrics_file),
                             {"TRA": 0.7, "CT": 0.8, "AOGM": 12.0})
            for name, ct, tra in (("low", 0.6, 0.9),
                                  ("high", 0.8, 0.7), ("higher_tra", 0.8, 0.8)):
                runner.write_json(root / name / "result.json", {
                    "experiment": name, "status": "complete", "CT": ct, "TRA": tra,
                })
            with runner.collect_results(root).open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["experiment"] for row in rows],
                             ["higher_tra", "high", "low"])
            with (root / "results_by_tra.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["experiment"] for row in rows],
                             ["low", "higher_tra", "high"])

    def test_retry_and_skip_completed_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / "finetune_config.yaml"
            config_file.write_text("epochs: 2\n", encoding="utf-8")
            model = root / "general_2d"
            model.mkdir()
            gt = root / "02_GT"
            (gt / "TRA").mkdir(parents=True)
            (gt / "TRA" / "man_track000.tif").touch()
            sequence = root / "02"
            sequence.mkdir()
            args = Namespace(config=config_file, output_root=root / "runs", seed=42,
                             mode="greedy", max_retries=1, python="python", device="cpu",
                             cache_dir=root / ".cache", trackastra_root=root)
            train_models = []

            def fake_command(command, log_file, cwd=None):
                if "train.py" in command[1]:
                    attempt = log_file.parent
                    config = runner.yaml.safe_load((attempt / "config.yaml").read_text())
                    train_models.append(config["model"])
                    if len(train_models) == 1:
                        raise RuntimeError("temporary training failure")
                    run_dir = attempt / "train"
                    run_dir.mkdir()
                    for name in ("model.pt", "config.yaml", "train_config.yaml"):
                        (run_dir / name).touch()
                else:
                    (log_file.parent / "ctc_metrics.csv").write_text(
                        "TRA;CT;AOGM\n0.9;0.8;2\n", encoding="utf-8"
                    )

            def fake_export(run_dir, sequence, gt, output, device, mode):
                output.mkdir()
                (output / "res_track.txt").touch()
                (output / "mask0000.tif").touch()

            with patch.object(runner, "run_command", fake_command), \
                 patch.object(runner, "best_validation", return_value=(2, 1, 0.25)), \
                 patch.object(runner, "export_ctc", fake_export):
                result = runner.run_experiment(args, {"epochs": 2}, model, sequence,
                                               gt, 2, 16, 1e-5, "ctc_evaluate")
                self.assertEqual(result["status"], "complete")
                self.assertEqual(train_models, [str(model), str(model)])
                self.assertEqual(result["actual_epochs"], 2)
                self.assertTrue(Path(result["ctc_result_dir"]).is_dir())
                errors = (Path(result["ctc_result_dir"]).parent / "errors.log").read_text()
                self.assertIn("temporary training failure", errors)
                skipped = runner.run_experiment(args, {"epochs": 2}, model, sequence,
                                                gt, 2, 16, 1e-5, "ctc_evaluate")
                self.assertEqual(skipped, result)
                self.assertEqual(len(train_models), 2)


if __name__ == "__main__":
    unittest.main()
