import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_normalize(scores):
    """Helper to run the normalize command with given scores."""
    # Prefer the project's virtualenv python if present
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    python_exec = str(venv_python) if venv_python.exists() else "python3"
    
    cmd = [python_exec, "cli/hybrid_search_cli.py", "normalize"]
    cmd.extend([str(s) for s in scores])
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    return result.stdout, result.returncode


def parse_normalized_scores(output):
    """Parse normalized scores from output."""
    lines = output.strip().split("\n")
    scores = []
    for line in lines:
        if line.startswith("* "):
            score_str = line[2:].strip()
            scores.append(float(score_str))
    return scores


class TestNormalizeCommand:
    """Tests for the normalize command in hybrid_search_cli.py"""

    def test_normalize_basic_scores(self):
        """Test basic normalization with various score ranges."""
        stdout, code = run_normalize([0.5, 2.3, 1.2, 0.5, 0.1])
        assert code == 0
        
        scores = parse_normalized_scores(stdout)
        assert len(scores) == 5
        
        # Check expected values (with small tolerance for floating point)
        assert abs(scores[0] - 0.1818) < 0.0001
        assert abs(scores[1] - 1.0000) < 0.0001
        assert abs(scores[2] - 0.5000) < 0.0001
        assert abs(scores[3] - 0.1818) < 0.0001
        assert abs(scores[4] - 0.0000) < 0.0001

    def test_normalize_ascending_scores(self):
        """Test normalization with simple ascending scores."""
        stdout, code = run_normalize([1.0, 2.0, 3.0])
        assert code == 0
        
        scores = parse_normalized_scores(stdout)
        assert len(scores) == 3
        assert abs(scores[0] - 0.0) < 0.0001  # min maps to 0
        assert abs(scores[1] - 0.5) < 0.0001  # middle maps to 0.5
        assert abs(scores[2] - 1.0) < 0.0001  # max maps to 1

    def test_normalize_identical_scores(self):
        """Test edge case when all scores are identical (should return 1.0 for all)."""
        stdout, code = run_normalize([5.0, 5.0, 5.0])
        assert code == 0
        
        scores = parse_normalized_scores(stdout)
        assert len(scores) == 3
        # All identical scores should normalize to 1.0
        for score in scores:
            assert abs(score - 1.0) < 0.0001

    def test_normalize_two_identical_scores(self):
        """Test with just two identical scores."""
        stdout, code = run_normalize([3.5, 3.5])
        assert code == 0
        
        scores = parse_normalized_scores(stdout)
        assert len(scores) == 2
        assert abs(scores[0] - 1.0) < 0.0001
        assert abs(scores[1] - 1.0) < 0.0001

    def test_normalize_empty_scores(self):
        """Test empty scores list handling (should print nothing)."""
        stdout, code = run_normalize([])
        assert code == 0
        assert stdout.strip() == ""

    def test_normalize_single_score(self):
        """Test single score handling (should return 1.0)."""
        stdout, code = run_normalize([7.5])
        assert code == 0
        
        scores = parse_normalized_scores(stdout)
        assert len(scores) == 1
        # Single score should normalize to 1.0 (min == max case)
        assert abs(scores[0] - 1.0) < 0.0001

    def test_normalize_negative_scores(self):
        """Test normalization with negative scores."""
        stdout, code = run_normalize([-2.0, 0.0, 2.0])
        assert code == 0
        
        scores = parse_normalized_scores(stdout)
        assert len(scores) == 3
        assert abs(scores[0] - 0.0) < 0.0001  # -2 is min
        assert abs(scores[1] - 0.5) < 0.0001  # 0 is middle
        assert abs(scores[2] - 1.0) < 0.0001  # 2 is max

    def test_normalize_output_format(self):
        """Test that output format matches expected pattern."""
        stdout, code = run_normalize([0.0, 0.5, 1.0])
        assert code == 0
        
        lines = stdout.strip().split("\n")
        assert len(lines) == 3
        
        # Each line should start with "* " followed by a number with 4 decimal places
        for line in lines:
            assert line.startswith("* ")
            score_part = line[2:]
            # Should be able to parse as float
            float(score_part)
            # Should have 4 decimal places (format: X.XXXX)
            assert "." in score_part
            decimal_part = score_part.split(".")[1]
            assert len(decimal_part) == 4

    def test_normalize_very_small_range(self):
        """Test normalization with a very small range."""
        stdout, code = run_normalize([1.0001, 1.0002, 1.0003])
        assert code == 0
        
        scores = parse_normalized_scores(stdout)
        assert len(scores) == 3
        # Should still normalize correctly even with tiny differences
        assert abs(scores[0] - 0.0) < 0.0001
        assert abs(scores[1] - 0.5) < 0.0001
        assert abs(scores[2] - 1.0) < 0.0001
