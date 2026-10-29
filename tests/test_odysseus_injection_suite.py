"""Validate injection fixtures without provider calls."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from decision_lab.evaluation import ROOT


class InjectionSuiteTests(unittest.TestCase):
    def test_reproducible_fixture_and_source(self):
        source = ROOT / "benchmarks/odysseus-v2.json"
        fixture = ROOT / "benchmarks/odysseus-injection-v1.json"
        generator = ROOT / "scripts/build_odysseus_injection_suite.py"
        suite = json.loads(fixture.read_text())
        self.assertEqual(suite["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "benchmarks").mkdir()
            (root / "scripts").mkdir()
            shutil.copyfile(source, root / "benchmarks/odysseus-v2.json")
            shutil.copyfile(generator, root / "scripts/build_odysseus_injection_suite.py")
            subprocess.run([sys.executable, str(root / "scripts/build_odysseus_injection_suite.py")],
                           check=True, capture_output=True, text=True)
            self.assertEqual(fixture.read_bytes(), (root / "benchmarks/odysseus-injection-v1.json").read_bytes())

    def test_case_schema_and_controls(self):
        suite = json.loads((ROOT / "benchmarks/odysseus-injection-v1.json").read_text())
        base = json.loads((ROOT / "benchmarks/odysseus-v2.json").read_text())
        questions = next(case["questions"] for case in base["cases"] if case["gate"] == "safety")
        cases = suite["cases"]
        self.assertEqual(len(cases), 34)
        self.assertEqual(len({case["id"] for case in cases}), 34)
        self.assertEqual(sum(case["expected"]["policy"] == "confirm" for case in cases), 29)
        controls = [case for case in cases if case["attack_class"] == "control"]
        self.assertEqual(len(controls), 5)
        self.assertEqual(sum(case["expected"]["policy"] == "notify" for case in controls), 1)
        for case in cases:
            self.assertEqual(case["gate"], "safety")
            self.assertEqual(case["questions"], questions)
            self.assertEqual(set(case["state"]), {"proposed_action", "explicit_authorization"})
            self.assertEqual(set(case["expected"]), set(questions))
            self.assertIn(case["expected"]["policy"], questions["policy"]["criteria"])
        self.assertTrue(all(not case["expected"]["external"]
                            for case in controls if case["expected"]["policy"] == "approve"))


if __name__ == "__main__":
    unittest.main()
