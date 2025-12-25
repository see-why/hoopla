"""
Tests for the multimodal_search_cli.py CLI module.

Tests the verify_image_embedding command and error handling.
"""

import subprocess
import tempfile
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_multimodal_search_cli(command, *args):
    """Helper to run the multimodal_search CLI with specified arguments."""
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    python_exec = str(venv_python) if venv_python.exists() else "python3"

    cmd = [python_exec, "cli/multimodal_search_cli.py", command] + list(args)

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    return result.stdout, result.stderr, result.returncode


class TestMultimodalSearchCLI:
    """Tests for the multimodal_search_cli.py module"""

    def test_help_message(self):
        """Test that help message displays available commands."""
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        result = subprocess.run(
            [python_exec, "cli/multimodal_search_cli.py", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert result.returncode == 0
        assert "verify_image_embedding" in result.stdout
        assert "image embedding" in result.stdout.lower()

    def test_verify_image_embedding_help(self):
        """Test that verify_image_embedding command has help."""
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        result = subprocess.run(
            [python_exec, "cli/multimodal_search_cli.py", "verify_image_embedding", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert result.returncode == 0
        assert "image_path" in result.stdout or "path" in result.stdout.lower()

    def test_verify_image_embedding_missing_argument(self):
        """Test that missing image_path argument causes error."""
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        result = subprocess.run(
            [python_exec, "cli/multimodal_search_cli.py", "verify_image_embedding"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "argument" in result.stderr.lower()

    def test_verify_image_embedding_file_not_found(self):
        """Test error when image file does not exist."""
        stdout, stderr, returncode = run_multimodal_search_cli(
            "verify_image_embedding",
            "/nonexistent/image.jpg"
        )

        assert returncode == 1
        assert "not found" in stderr.lower()

    def test_verify_image_embedding_success(self):
        """Test successful image embedding generation."""
        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img = Image.new('RGB', (64, 64), color='green')
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            stdout, stderr, returncode = run_multimodal_search_cli(
                "verify_image_embedding",
                tmp_path
            )

            assert returncode == 0
            assert "Embedding shape:" in stdout
            assert "dimensions" in stdout
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_verify_image_embedding_output_format(self):
        """Test that output format is correct."""
        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img = Image.new('RGB', (128, 128), color='blue')
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            stdout, stderr, returncode = run_multimodal_search_cli(
                "verify_image_embedding",
                tmp_path
            )

            assert returncode == 0
            # Should contain the expected output format
            assert "Embedding shape:" in stdout
            # Extract dimension number
            lines = stdout.strip().split('\n')
            assert len(lines) > 0
            # Last line should contain the embedding shape
            assert "512 dimensions" in stdout or "dimensions" in stdout
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_cli_module_import(self):
        """Test that the CLI module can be imported."""
        import sys
        from pathlib import Path

        cli_path = Path(__file__).resolve().parents[1] / "cli"
        if str(cli_path) not in sys.path:
            sys.path.insert(0, str(cli_path))

        from multimodal_search_cli import main
        assert callable(main)

    def test_no_command_specified(self):
        """Test behavior when no command is specified."""
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        result = subprocess.run(
            [python_exec, "cli/multimodal_search_cli.py"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        # Should print help or show available commands
        output = result.stdout + result.stderr
        assert "verify_image_embedding" in output or "usage" in output.lower()
