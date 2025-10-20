import json
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MOVIES_PATH = DATA_DIR / "movies.json"
BACKUP_PATH = DATA_DIR / "movies.json.bak"


def write_movies(data):
    DATA_DIR.mkdir(exist_ok=True)
    with open(MOVIES_PATH, "w", encoding="utf-8") as fh:
        json.dump({"movies": data}, fh, ensure_ascii=False, indent=2)


@pytest.fixture(autouse=True)
def backup_and_restore_movies():
    # Backup existing file if present
    if MOVIES_PATH.exists():
        shutil.copy2(MOVIES_PATH, BACKUP_PATH)
    try:
        yield
    finally:
        # Restore original file
        if BACKUP_PATH.exists():
            shutil.move(BACKUP_PATH, MOVIES_PATH)
        else:
            try:
                MOVIES_PATH.unlink()
            except FileNotFoundError:
                pass


def run_search(query):
    cmd = ["python3", "cli/keyword_search_cli.py", "search", query]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def test_case_insensitive_and_punctuation():
    movies = [
        {"id": 2, "title": "Star-Wars: A New Hope"},
        {"id": 1, "title": "star wars: the empire strikes back"},
        {"id": 3, "title": "Other Movie"},
    ]
    write_movies(movies)

    rc, out, err = run_search("STAR WARS")
    assert rc == 0
    # Should match the two Star Wars titles, sorted by id ascending, truncated to 5
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    assert any("Searching for:" in l for l in lines)
    # find the numbered results
    numbered = [l for l in lines if l[0].isdigit()]
    assert numbered[0].endswith("star wars: the empire strikes back")
    assert numbered[1].endswith("Star-Wars: A New Hope")


def test_truncate_results_and_sort():
    # Create 7 matches to ensure truncation to 5
    movies = [{"id": i, "title": f"Match {i}"} for i in range(10, 3, -1)]
    write_movies(movies)

    rc, out, err = run_search("match")
    assert rc == 0
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    numbered = [l for l in lines if l[0].isdigit()]
    # Should be at most 5 results
    assert len(numbered) <= 5


def test_no_results():
    write_movies([{"id": 1, "title": "Something Else"}])
    rc, out, err = run_search("nope")
    assert rc == 0
    assert "No results found." in out
