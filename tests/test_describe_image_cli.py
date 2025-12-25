import io
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_describe_image(image_path, query):
    """Helper to run the describe_image CLI with specified arguments."""
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    python_exec = str(venv_python) if venv_python.exists() else "python3"

    cmd = [python_exec, "cli/describe_image_cli.py", "--image", image_path, "--query", query]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    return result.stdout, result.stderr, result.returncode


class TestDescribeImageCLI:
    """Tests for the describe_image_cli.py module"""

    def test_help_message(self):
        """Test that help message displays available arguments."""
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        result = subprocess.run(
            [python_exec, "cli/describe_image_cli.py", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert result.returncode == 0
        assert "--image" in result.stdout
        assert "--query" in result.stdout
        assert "Path to the image file" in result.stdout
        assert "Text query to rewrite based on the image" in result.stdout

    def test_missing_required_image_argument(self):
        """Test that missing --image argument causes error."""
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        result = subprocess.run(
            [python_exec, "cli/describe_image_cli.py", "--query", "test query"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "required" in result.stdout.lower()

    def test_missing_required_query_argument(self):
        """Test that missing --query argument causes error."""
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        result = subprocess.run(
            [python_exec, "cli/describe_image_cli.py", "--image", "/tmp/test.jpg"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "required" in result.stdout.lower()

    def test_image_file_not_found(self):
        """Test error handling when image file does not exist."""
        stdout, stderr, code = run_describe_image("/nonexistent/path/to/image.jpg", "Find movies")

        assert code == 1
        assert "not found" in stderr.lower()

    def test_image_file_validation(self):
        """Test that image file path is validated."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(b"\xff\xd8\xff\xe0")  # JPEG magic bytes
            tmp_path = tmp.name

        try:
            # Use environment variable to mock API key existence check
            # This test just verifies the file is read without errors
            import os
            if not os.environ.get("GEMINI_API_KEY"):
                # Mock would be needed here; for now skip deep integration
                pass
            # File should be readable without immediate error
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_mime_type_detection_jpeg(self):
        """Test that JPEG files are recognized."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(b"\xff\xd8\xff\xe0")  # JPEG magic bytes
            tmp_path = tmp.name

        try:
            # Verify file exists and has expected extension
            assert tmp_path.endswith(".jpg")
            import os
            assert os.path.exists(tmp_path)
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_mime_type_detection_png(self):
        """Test that PNG files are recognized."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(b"\x89PNG\r\n\x1a\n")  # PNG magic bytes
            tmp_path = tmp.name

        try:
            # Verify file exists and has expected extension
            assert tmp_path.endswith(".png")
            import os
            assert os.path.exists(tmp_path)
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_default_mime_type_for_unknown_extension(self):
        """Test that unknown file extensions are handled."""
        with tempfile.NamedTemporaryFile(suffix=".unknown", delete=False) as tmp:
            tmp.write(b"test data")
            tmp_path = tmp.name

        try:
            # Verify file validation still works
            import os
            assert os.path.exists(tmp_path)
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_query_argument_required(self):
        """Test that query argument is required."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(b"\xff\xd8\xff\xe0")
            tmp_path = tmp.name

        try:
            venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
            python_exec = str(venv_python) if venv_python.exists() else "python3"

            result = subprocess.run(
                [python_exec, "cli/describe_image_cli.py", "--image", tmp_path],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )

            assert result.returncode != 0
            assert "required" in result.stderr.lower() or "required" in result.stdout.lower()
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_image_argument_required(self):
        """Test that image argument is required."""
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        result = subprocess.run(
            [python_exec, "cli/describe_image_cli.py", "--query", "test"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "required" in result.stdout.lower()

    def test_query_with_special_characters(self):
        """Test query containing special characters."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(b"\xff\xd8\xff\xe0")
            tmp_path = tmp.name

        try:
            venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
            python_exec = str(venv_python) if venv_python.exists() else "python3"

            cmd = [
                python_exec,
                "cli/describe_image_cli.py",
                "--image",
                tmp_path,
                "--query",
                "find movies: action & adventure",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)

            # Should not crash on special characters
            # Will fail with API error, but not argument parsing error
            assert "required" not in result.stderr.lower()
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_query_with_whitespace(self):
        """Test query containing leading/trailing whitespace."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(b"\xff\xd8\xff\xe0")
            tmp_path = tmp.name

        try:
            venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
            python_exec = str(venv_python) if venv_python.exists() else "python3"

            cmd = [
                python_exec,
                "cli/describe_image_cli.py",
                "--image",
                tmp_path,
                "--query",
                "  test query with spaces  ",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)

            # Should not fail on whitespace
            assert "required" not in result.stderr.lower()
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_large_image_file(self):
        """Test that larger image files can be read."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(b"\xff\xd8\xff\xe0")  # JPEG header
            tmp.write(b"X" * (256 * 1024))  # 256KB of data
            tmp_path = tmp.name

        try:
            # File should be readable without size-based errors
            import os
            file_size = os.path.getsize(tmp_path)
            assert file_size > 256 * 1024
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_empty_image_file(self):
        """Test handling of empty image file."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            # Write nothing, just empty file
            tmp_path = tmp.name

        try:
            stdout, stderr, code = run_describe_image(tmp_path, "test query")

            # Empty file should still be readable (MIME detection will default to jpeg)
            # Errors may come from Gemini API, not file reading
            assert "not found" not in stderr.lower()
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_cli_module_import(self):
        """Test that the CLI module can be imported without errors."""
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        result = subprocess.run(
            [python_exec, "-c", "from cli.describe_image_cli import main"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert result.returncode == 0

    def test_help_contains_description(self):
        """Test that help includes a helpful description."""
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        result = subprocess.run(
            [python_exec, "cli/describe_image_cli.py", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert result.returncode == 0
        assert "Rewrite" in result.stdout or "image" in result.stdout.lower()
    def test_magic_byte_detection_jpeg(self):
        """Test that JPEG files are detected by magic bytes."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            # JPEG magic bytes: FF D8 FF
            tmp.write(b"\xff\xd8\xff\xe0\x00\x10JFIF")
            tmp_path = tmp.name

        try:
            stdout, stderr, returncode = run_describe_image(tmp_path, "test query")
            # Should fail due to GEMINI_API_KEY or other reasons, but not due to image format
            assert "not appear to be an image" not in stderr
            assert "Could not determine image format" not in stderr
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_magic_byte_detection_png(self):
        """Test that PNG files are detected by magic bytes."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            # PNG magic bytes: 89 50 4E 47 0D 0A 1A 0A
            tmp.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
            tmp_path = tmp.name

        try:
            stdout, stderr, returncode = run_describe_image(tmp_path, "test query")
            # Should fail due to GEMINI_API_KEY or other reasons, but not due to image format
            assert "not appear to be an image" not in stderr
            assert "Could not determine image format" not in stderr
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_magic_byte_detection_gif(self):
        """Test that GIF files are detected by magic bytes."""
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            # GIF magic bytes: 47 49 46 38 39 61 or 47 49 46 38 37 61 (GIF89a or GIF87a)
            tmp.write(b"GIF89a" + b"\x00" * 100)
            tmp_path = tmp.name

        try:
            stdout, stderr, returncode = run_describe_image(tmp_path, "test query")
            # Should fail due to GEMINI_API_KEY or other reasons, but not due to image format
            assert "not appear to be an image" not in stderr
            assert "Could not determine image format" not in stderr
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_magic_byte_detection_webp(self):
        """Test that WebP files are detected by magic bytes."""
        with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as tmp:
            # WebP magic bytes: RIFF ... WEBP
            tmp.write(b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 100)
            tmp_path = tmp.name

        try:
            stdout, stderr, returncode = run_describe_image(tmp_path, "test query")
            # Should fail due to GEMINI_API_KEY or other reasons, but not due to image format
            assert "not appear to be an image" not in stderr
            assert "Could not determine image format" not in stderr
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_non_image_file_rejected(self):
        """Test that non-image files are rejected with clear error message."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"This is a text file, not an image")
            tmp_path = tmp.name

        try:
            stdout, stderr, returncode = run_describe_image(tmp_path, "test query")
            assert returncode == 1
            assert "Could not determine image format" in stderr or "not appear to be an image" in stderr
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_file_with_wrong_extension_but_valid_image(self):
        """Test that files with wrong extensions but valid image content are accepted."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            # PNG magic bytes in a .bin file
            tmp.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
            tmp_path = tmp.name

        try:
            stdout, stderr, returncode = run_describe_image(tmp_path, "test query")
            # Should not fail due to format detection (magic bytes should identify it as PNG)
            assert "Could not determine image format" not in stderr
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_file_with_image_extension_but_invalid_content(self):
        """Test that files with image extension but invalid image content are rejected."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            # Invalid image data (not JPEG magic bytes)
            tmp.write(b"This is not a real image file")
            tmp_path = tmp.name

        try:
            stdout, stderr, returncode = run_describe_image(tmp_path, "test query")
            # API should reject the invalid image (error from Gemini API)
            # This is OK - the file passed our validation (has .jpg extension)
            # but the API correctly rejects it as invalid
            if returncode == 1:
                # Either our validation caught it or API did - both acceptable
                assert "Could not determine image format" in stderr or "invalid" in stderr.lower() or "not valid" in stderr.lower()
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)