"""Output-isolation tests; no model construction, fitting or inference."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("reproduce_evaluation", ROOT / "tools/reproduce_evaluation.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EvaluationIsolationTests(unittest.TestCase):
    def fixture(self, root):
        root.mkdir()
        for name in MODULE.INPUTS:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith(".txt"):
                path.write_text("fixture")
            else:
                path.mkdir(exist_ok=True)

    def test_protected_existing_and_symlink_destinations_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "research"
            self.fixture(root)
            existing = base / "existing"
            existing.mkdir()
            alias = base / "alias"
            alias.symlink_to(root, target_is_directory=True)
            for output in (root / "new", root, base, existing, alias / "new"):
                with self.subTest(output=output), self.assertRaises((ValueError, FileExistsError)):
                    MODULE.prepare_workspace(root, output)
            self.assertFalse((root / "new").exists())

    def test_missing_inputs_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "research"
            root.mkdir()
            with self.assertRaises(FileNotFoundError):
                MODULE.prepare_workspace(root, base / "new")
            self.assertFalse((base / "new").exists())

    def test_evaluators_are_sequential_and_outputs_are_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "research"
            self.fixture(root)
            output = base / "new"
            with patch.object(MODULE.subprocess, "run") as run:
                MODULE.reproduce(root, output)
            self.assertEqual(run.call_count, 2)
            first, second = run.call_args_list
            self.assertIn("src.benchmark", first.args[0])
            self.assertIn(str(output / "historical"), first.args[0])
            self.assertIn("src.controlled_v2", second.args[0])
            self.assertEqual(first.kwargs["cwd"], output / "workspace")
            self.assertTrue(first.kwargs["check"])
            self.assertFalse((output / "workspace/results/evaluation_controlled_v2").exists())
            self.assertEqual((output / "workspace/results/controlled_v2").resolve(),
                             (root / "results/controlled_v2").resolve())

    def test_first_failure_stops_before_second_evaluator(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "research"
            self.fixture(root)
            failure = subprocess.CalledProcessError(1, ["fixture"])
            with patch.object(MODULE.subprocess, "run", side_effect=failure) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    MODULE.reproduce(root, base / "new")
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
