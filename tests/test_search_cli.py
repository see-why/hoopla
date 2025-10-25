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
    # Prefer the project's virtualenv python if present so tests run with
    # the same dependencies (e.g., nltk). Fall back to system python3.
    venv_python = (PROJECT_ROOT / ".venv" / "bin" / "python")
    python_exec = str(venv_python) if venv_python.exists() else "python3"
    # Build the index cache first
    build_cmd = [python_exec, "cli/keyword_search_cli.py", "build"]
    subprocess.run(build_cmd, capture_output=True, text=True)

    cmd = [python_exec, "cli/keyword_search_cli.py", "search", query]
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


def test_token_partial_match_great_bear():
    # Verify that a query like "Great Bear" matches a title "Big Bear"
    movies = [
        {"id": 1, "title": "Big Bear"},
        {"id": 2, "title": "Some Other"},
    ]
    write_movies(movies)

    rc, out, err = run_search("Great Bear")
    assert rc == 0
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    # find the numbered results
    numbered = [l for l in lines if l and l[0].isdigit()]
    assert any("Big Bear" in l for l in numbered)


def parse_results(out):
    import re
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    numbered = [l for l in lines if l and l[0].isdigit()]
    results = []
    for l in numbered:
        m = re.match(r"^\d+\.\s*\[(\d+)\]\s*(.*)$", l)
        if m:
            results.append((int(m.group(1)), m.group(2)))
        else:
            # fallback: include raw line
            results.append((None, l))
    return results


def test_boolean_and_or_not():
    # Setup a small set of movies to test boolean logic
    movies = [
        {"id": 1, "title": "Bear Wizard"},
        {"id": 2, "title": "Just Bear"},
        {"id": 3, "title": "Wizard Alone"},
        {"id": 4, "title": "Terror Bear"},
        {"id": 5, "title": "Cyborg Saga"},
    ]
    write_movies(movies)

    # AND: should return doc 1 only (contains both bear and wizard)
    rc, out, err = run_search("bear AND wizard")
    assert rc == 0
    res = parse_results(out)
    assert len(res) == 1
    assert res[0][0] == 1

    # NOT: bear NOT terror -> should exclude id 4, include id 1 and 2 (but truncated to up to 5)
    rc, out, err = run_search("bear NOT terror")
    assert rc == 0
    res = parse_results(out)
    # ids containing 'bear' are 1,2,4; after NOT terror remove 4 -> 1 and 2 (sorted asc for boolean)
    ids = [r[0] for r in res]
    assert 1 in ids and 2 in ids and 4 not in ids

    # OR: bear OR cyborg -> should return ids that contain either term (1,2,4,5) -> sorted asc -> 1,2,4,5 (truncated to 5)
    rc, out, err = run_search("bear OR cyborg")
    assert rc == 0
    res = parse_results(out)
    ids = [r[0] for r in res]
    # expect 1 and 2 and 4 and 5 present (may be truncated to <=5)
    for want in (1, 2, 4, 5):
        assert want in ids
