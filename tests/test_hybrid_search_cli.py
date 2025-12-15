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


def run_weighted_search(query, alpha=None, limit=None):
    """Helper to run the weighted-search command."""
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    python_exec = str(venv_python) if venv_python.exists() else "python3"
    
    cmd = [python_exec, "cli/hybrid_search_cli.py", "weighted-search", query]
    if alpha is not None:
        cmd.extend(["--alpha", str(alpha)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    return result.stdout, result.stderr, result.returncode


def parse_weighted_search_results(output):
    """Parse weighted search results from output.
    
    Returns a list of dicts with keys: rank, title, hybrid_score, bm25_score, semantic_score, description
    """
    results = []
    lines = output.strip().split("\n")
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for rank line (e.g., "1. Paddington")
        if line and line[0].isdigit() and ". " in line:
            rank_str, title = line.split(". ", 1)
            rank = int(rank_str)
            
            # Next line should be hybrid score
            i += 1
            if i < len(lines) and "Hybrid Score:" in lines[i]:
                hybrid_score = float(lines[i].split("Hybrid Score:")[1].strip())
                
                # Next line should be BM25 and Semantic scores
                i += 1
                if i < len(lines) and "BM25:" in lines[i] and "Semantic:" in lines[i]:
                    parts = lines[i].strip().split(",")
                    bm25_score = float(parts[0].split("BM25:")[1].strip())
                    semantic_score = float(parts[1].split("Semantic:")[1].strip())
                    
                    # Next line should be description
                    i += 1
                    description = lines[i].strip() if i < len(lines) else ""
                    
                    results.append({
                        "rank": rank,
                        "title": title,
                        "hybrid_score": hybrid_score,
                        "bm25_score": bm25_score,
                        "semantic_score": semantic_score,
                        "description": description
                    })
        
        i += 1
    
    return results


class TestWeightedSearchCommand:
    """Tests for the weighted-search command in hybrid_search_cli.py"""

    def test_weighted_search_alpha_0_5_default(self):
        """Test weighted search with default alpha (0.5) - balanced hybrid."""
        stdout, stderr, code = run_weighted_search("bear paddington", limit=5)
        assert code == 0
        assert "Top 5 results" in stdout
        assert "alpha=0.5" in stdout
        
        results = parse_weighted_search_results(stdout)
        assert len(results) == 5
        
        # Verify all results have valid scores
        for result in results:
            assert 0.0 <= result["hybrid_score"] <= 1.0
            assert 0.0 <= result["bm25_score"] <= 1.0
            assert 0.0 <= result["semantic_score"] <= 1.0
            assert result["title"]
            assert result["description"]
        
        # Results should be ordered by hybrid score descending
        for i in range(len(results) - 1):
            assert results[i]["hybrid_score"] >= results[i + 1]["hybrid_score"]

    def test_weighted_search_alpha_0_0_semantic_only(self):
        """Test weighted search with alpha=0.0 (100% semantic search)."""
        stdout, stderr, code = run_weighted_search("bear paddington", alpha=0.0, limit=3)
        assert code == 0
        assert "alpha=0.0" in stdout
        
        results = parse_weighted_search_results(stdout)
        assert len(results) == 3
        
        # With alpha=0, hybrid score should equal semantic score
        for result in results:
            # Allow small floating point difference
            assert abs(result["hybrid_score"] - result["semantic_score"]) < 0.001

    def test_weighted_search_alpha_1_0_bm25_only(self):
        """Test weighted search with alpha=1.0 (100% BM25 keyword search)."""
        stdout, stderr, code = run_weighted_search("bear paddington", alpha=1.0, limit=3)
        assert code == 0
        assert "alpha=1.0" in stdout
        
        results = parse_weighted_search_results(stdout)
        assert len(results) == 3
        
        # With alpha=1, hybrid score should equal BM25 score
        for result in results:
            # Allow small floating point difference
            assert abs(result["hybrid_score"] - result["bm25_score"]) < 0.001

    def test_weighted_search_alpha_0_2_favor_semantic(self):
        """Test weighted search with alpha=0.2 (favor semantic search)."""
        stdout, stderr, code = run_weighted_search("bear paddington", alpha=0.2, limit=5)
        assert code == 0
        assert "alpha=0.2" in stdout
        
        results = parse_weighted_search_results(stdout)
        assert len(results) == 5
        
        # Verify hybrid score calculation: hybrid = 0.2 * bm25 + 0.8 * semantic
        for result in results:
            expected = 0.2 * result["bm25_score"] + 0.8 * result["semantic_score"]
            assert abs(result["hybrid_score"] - expected) < 0.001

    def test_weighted_search_alpha_0_8_favor_bm25(self):
        """Test weighted search with alpha=0.8 (favor BM25 keyword search)."""
        stdout, stderr, code = run_weighted_search("bear paddington", alpha=0.8, limit=5)
        assert code == 0
        assert "alpha=0.8" in stdout
        
        results = parse_weighted_search_results(stdout)
        assert len(results) == 5
        
        # Verify hybrid score calculation: hybrid = 0.8 * bm25 + 0.2 * semantic
        for result in results:
            expected = 0.8 * result["bm25_score"] + 0.2 * result["semantic_score"]
            assert abs(result["hybrid_score"] - expected) < 0.001

    def test_weighted_search_different_limits(self):
        """Test weighted search respects different limit values."""
        for limit in [1, 3, 10]:
            stdout, stderr, code = run_weighted_search("space adventure", limit=limit)
            assert code == 0
            
            results = parse_weighted_search_results(stdout)
            assert len(results) == limit
            assert f"Top {limit} results" in stdout

    def test_weighted_search_output_format(self):
        """Test that output format is correct."""
        stdout, stderr, code = run_weighted_search("british bear", alpha=0.5, limit=3)
        assert code == 0
        
        # Check header format
        assert "Top 3 results for query: 'british bear'" in stdout
        assert "alpha=0.5" in stdout
        
        results = parse_weighted_search_results(stdout)
        assert len(results) == 3
        
        # Check that each result has all required fields
        for result in results:
            assert result["rank"] > 0
            assert result["title"] != ""
            assert result["hybrid_score"] >= 0.0
            assert result["bm25_score"] >= 0.0
            assert result["semantic_score"] >= 0.0
            # Description should be truncated (approximately 100 chars or less including "...")
            if len(result["description"]) > 3:
                assert len(result["description"]) <= 105  # Allow some tolerance

    def test_weighted_search_score_normalization(self):
        """Test that scores are properly normalized to 0-1 range."""
        stdout, stderr, code = run_weighted_search("action movie", alpha=0.5, limit=10)
        assert code == 0
        
        results = parse_weighted_search_results(stdout)
        assert len(results) == 10
        
        # All scores should be in [0, 1] range
        for result in results:
            assert 0.0 <= result["hybrid_score"] <= 1.0
            assert 0.0 <= result["bm25_score"] <= 1.0
            assert 0.0 <= result["semantic_score"] <= 1.0
        
        # At least one result should have a normalized score near 1.0 (top result)
        max_hybrid = max(r["hybrid_score"] for r in results)
        assert max_hybrid >= 0.9  # Top score should be close to 1.0

    def test_weighted_search_result_ordering(self):
        """Test that results are correctly ordered by hybrid score."""
        stdout, stderr, code = run_weighted_search("comedy film", alpha=0.5, limit=15)
        assert code == 0
        
        results = parse_weighted_search_results(stdout)
        assert len(results) == 15
        
        # Verify strict descending order
        hybrid_scores = [r["hybrid_score"] for r in results]
        assert hybrid_scores == sorted(hybrid_scores, reverse=True)
        
        # Verify ranks are sequential
        for i, result in enumerate(results):
            assert result["rank"] == i + 1

    def test_weighted_search_specific_movies(self):
        """Test that specific expected movies appear in results."""
        stdout, stderr, code = run_weighted_search("British Bear", alpha=0.8, limit=25)
        assert code == 0
        
        results = parse_weighted_search_results(stdout)
        titles = [r["title"] for r in results]
        
        # These movies should appear based on the query
        assert "Paddington" in titles
        assert "The Duchess" in titles
        assert "The Great Bear" in titles

    def test_weighted_search_empty_query(self):
        """Test weighted search with empty or very short query."""
        # Python argparse requires at least the query argument, so empty string should work
        stdout, stderr, code = run_weighted_search("", limit=5)
        # Should either return results or handle gracefully
        # The actual behavior depends on implementation, but shouldn't crash
        assert code == 0 or "error" in stderr.lower()

    def test_weighted_search_unusual_query(self):
        """Test weighted search with unusual query characters."""
        stdout, stderr, code = run_weighted_search("@#$%", limit=3)
        # Should handle gracefully, either returning results or empty
        assert code == 0
        # May have no results for special characters
        assert "No results found" in stdout or "Top" in stdout

    def test_weighted_search_long_query(self):
        """Test weighted search with a very long query."""
        long_query = "adventure action thriller suspense drama comedy romance sci-fi fantasy"
        stdout, stderr, code = run_weighted_search(long_query, alpha=0.5, limit=5)
        assert code == 0
        
        results = parse_weighted_search_results(stdout)
        assert len(results) == 5

    def test_weighted_search_alpha_boundaries(self):
        """Test weighted search with alpha at exact boundaries."""
        # Test alpha = 0.0
        stdout1, stderr1, code1 = run_weighted_search("test", alpha=0.0, limit=2)
        assert code1 == 0
        
        # Test alpha = 1.0
        stdout2, stderr2, code2 = run_weighted_search("test", alpha=1.0, limit=2)
        assert code2 == 0
        
        # Both should return valid results
        results1 = parse_weighted_search_results(stdout1)
        results2 = parse_weighted_search_results(stdout2)
        assert len(results1) <= 2
        assert len(results2) <= 2

    def test_weighted_search_alpha_validation_negative(self):
        """Test that negative alpha values are rejected."""
        stdout, stderr, code = run_weighted_search("test", alpha=-0.5, limit=2)
        assert code == 1  # Should exit with error code
        assert "alpha must be between 0.0 and 1.0" in stderr
        assert "-0.5" in stderr

    def test_weighted_search_alpha_validation_too_high(self):
        """Test that alpha values > 1.0 are rejected."""
        stdout, stderr, code = run_weighted_search("test", alpha=1.5, limit=2)
        assert code == 1  # Should exit with error code
        assert "alpha must be between 0.0 and 1.0" in stderr
        assert "1.5" in stderr

    def test_weighted_search_alpha_validation_way_out_of_range(self):
        """Test that wildly out-of-range alpha values are rejected."""
        # Test very large value
        stdout1, stderr1, code1 = run_weighted_search("test", alpha=100.0, limit=2)
        assert code1 == 1
        assert "alpha must be between 0.0 and 1.0" in stderr1
        
        # Test very negative value
        stdout2, stderr2, code2 = run_weighted_search("test", alpha=-50.0, limit=2)
        assert code2 == 1
        assert "alpha must be between 0.0 and 1.0" in stderr2

    def test_weighted_search_large_limit(self):
        """Test weighted search with large limit values."""
        # This should work without performance issues due to MAX_EXPANDED_LIMIT cap
        stdout, stderr, code = run_weighted_search("action", alpha=0.5, limit=100)
        assert code == 0
        
        results = parse_weighted_search_results(stdout)
        # Should return up to 100 results (or fewer if less are available)
        assert len(results) <= 100
        assert len(results) > 0

    def test_weighted_search_very_large_limit(self):
        """Test weighted search with very large limit to verify capping behavior."""
        # With limit=1000, expanded would be 500,000 without cap
        # But MAX_EXPANDED_LIMIT should cap it to 10,000
        stdout, stderr, code = run_weighted_search("movie", alpha=0.5, limit=1000)
        assert code == 0
        
        results = parse_weighted_search_results(stdout)
        # Should still complete successfully
        assert len(results) <= 1000
        assert len(results) > 0

    def test_weighted_search_limit_validation_zero(self):
        """Test that zero limit is rejected."""
        stdout, stderr, code = run_weighted_search("test", alpha=0.5, limit=0)
        assert code == 1  # Should exit with error code
        assert "limit must be a positive integer" in stderr
        assert "0" in stderr

    def test_weighted_search_limit_validation_negative(self):
        """Test that negative limit values are rejected."""
        stdout, stderr, code = run_weighted_search("test", alpha=0.5, limit=-5)
        assert code == 1  # Should exit with error code
        assert "limit must be a positive integer" in stderr
        assert "-5" in stderr

    def test_weighted_search_limit_validation_large_negative(self):
        """Test that large negative limit values are rejected."""
        stdout, stderr, code = run_weighted_search("test", alpha=0.5, limit=-1000)
        assert code == 1
        assert "limit must be a positive integer" in stderr

    def test_weighted_search_deterministic_results(self):
        """Test that weighted search returns consistent results across multiple runs."""
        query = "space adventure"
        alpha = 0.5
        limit = 10
        
        # Run the search multiple times
        results_list = []
        for _ in range(3):
            stdout, stderr, code = run_weighted_search(query, alpha=alpha, limit=limit)
            assert code == 0
            results = parse_weighted_search_results(stdout)
            results_list.append(results)
        
        # All runs should produce identical results (same titles in same order)
        for i in range(1, len(results_list)):
            assert len(results_list[i]) == len(results_list[0])
            for j in range(len(results_list[0])):
                assert results_list[i][j]["title"] == results_list[0][j]["title"]
                assert results_list[i][j]["rank"] == results_list[0][j]["rank"]
                # Scores should also be identical
                assert abs(results_list[i][j]["hybrid_score"] - results_list[0][j]["hybrid_score"]) < 0.0001
                assert abs(results_list[i][j]["bm25_score"] - results_list[0][j]["bm25_score"]) < 0.0001
                assert abs(results_list[i][j]["semantic_score"] - results_list[0][j]["semantic_score"]) < 0.0001


class TestWeightedSearchMethodValidation:
    """Tests for method-level validation in HybridSearch.weighted_search()"""

    def test_method_alpha_validation_negative(self):
        """Test that the weighted_search method rejects negative alpha values."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        import pytest
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        
        # Should raise ValueError for negative alpha
        with pytest.raises(ValueError, match="alpha must be between 0.0 and 1.0"):
            hs.weighted_search("test", alpha=-0.5, limit=5)

    def test_method_alpha_validation_too_high(self):
        """Test that the weighted_search method rejects alpha values > 1.0."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        import pytest
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        
        # Should raise ValueError for alpha > 1.0
        with pytest.raises(ValueError, match="alpha must be between 0.0 and 1.0"):
            hs.weighted_search("test", alpha=1.5, limit=5)

    def test_method_alpha_validation_boundaries(self):
        """Test that the weighted_search method accepts boundary values 0.0 and 1.0."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        
        # Should accept alpha = 0.0
        results1 = hs.weighted_search("test", alpha=0.0, limit=2)
        assert len(results1) <= 2
        
        # Should accept alpha = 1.0
        results2 = hs.weighted_search("test", alpha=1.0, limit=2)
        assert len(results2) <= 2


def run_rrf_search(query, k=None, limit=None):
    """Helper to run the rrf-search command."""
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    python_exec = str(venv_python) if venv_python.exists() else "python3"
    
    cmd = [python_exec, "cli/hybrid_search_cli.py", "rrf-search", query]
    if k is not None:
        cmd.extend(["--k", str(k)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    return result.stdout, result.stderr, result.returncode


def run_rrf_search_with_enhance(query, enhance_method, k=None, limit=None, env=None):
    """Helper to run the rrf-search command with --enhance flag."""
    # Prefer the project's virtualenv python if present
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    python_exec = str(venv_python) if venv_python.exists() else "python3"
    
    cmd = [python_exec, "cli/hybrid_search_cli.py", "rrf-search", query]
    cmd.extend(["--enhance", enhance_method])
    if k is not None:
        cmd.extend(["--k", str(k)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, env=env)
    return result.stdout, result.stderr, result.returncode


def parse_rrf_search_results(output):
    """Parse RRF search results from output.
    
    Returns a list of dicts with keys: rank, title, rrf_score, bm25_rank, semantic_rank, description
    """
    results = []
    lines = output.strip().split("\n")
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for rank line (e.g., "1. Paddington")
        if line and line[0].isdigit() and ". " in line:
            rank_str, title = line.split(". ", 1)
            rank = int(rank_str)
            
            # Next line should be RRF score
            i += 1
            if i < len(lines) and "RRF Score:" in lines[i]:
                rrf_score = float(lines[i].split("RRF Score:")[1].strip())
                
                # Next line should be BM25 Rank and Semantic Rank
                i += 1
                if i < len(lines) and "BM25 Rank:" in lines[i] and "Semantic Rank:" in lines[i]:
                    parts = lines[i].strip().split(",")
                    bm25_str = parts[0].split("BM25 Rank:")[1].strip()
                    semantic_str = parts[1].split("Semantic Rank:")[1].strip()
                    bm25_rank = int(bm25_str) if bm25_str != "None" else None
                    semantic_rank = int(semantic_str) if semantic_str != "None" else None
                    
                    # Next line should be description
                    i += 1
                    description = lines[i].strip() if i < len(lines) else ""
                    
                    results.append({
                        "rank": rank,
                        "title": title,
                        "rrf_score": rrf_score,
                        "bm25_rank": bm25_rank,
                        "semantic_rank": semantic_rank,
                        "description": description
                    })
        
        i += 1
    
    return results


class TestRRFSearchCommand:
    """Tests for the rrf-search command in hybrid_search_cli.py"""

    def test_rrf_search_default_parameters(self):
        """Test RRF search with default k (60) and limit (5)."""
        stdout, stderr, code = run_rrf_search("magical bear")
        assert code == 0
        assert "Top 5 results" in stdout
        assert "k=60" in stdout
        
        results = parse_rrf_search_results(stdout)
        assert len(results) == 5
        
        # Verify all results have valid scores and ranks
        for result in results:
            assert result["rrf_score"] > 0.0
            assert result["bm25_rank"] > 0
            assert result["semantic_rank"] > 0
            assert result["title"]
            assert result["description"]
        
        # Results should be ordered by RRF score descending
        for i in range(len(results) - 1):
            assert results[i]["rrf_score"] >= results[i + 1]["rrf_score"]

    def test_rrf_search_custom_k(self):
        """Test RRF search with custom k value."""
        stdout, stderr, code = run_rrf_search("magical bear", k=30, limit=3)
        assert code == 0
        assert "k=30" in stdout
        assert "Top 3 results" in stdout
        
        results = parse_rrf_search_results(stdout)
        assert len(results) == 3
        
        # Verify RRF scores are calculated correctly (higher k = higher scores)
        for result in results:
            assert result["rrf_score"] > 0.0

    def test_rrf_search_custom_limit(self):
        """Test RRF search with custom limit value."""
        stdout, stderr, code = run_rrf_search("bear", limit=10)
        assert code == 0
        assert "Top 10 results" in stdout
        
        results = parse_rrf_search_results(stdout)
        assert len(results) == 10

    def test_rrf_search_small_limit(self):
        """Test RRF search with limit=1."""
        stdout, stderr, code = run_rrf_search("paddington", limit=1)
        assert code == 0
        
        results = parse_rrf_search_results(stdout)
        assert len(results) == 1
        
        # Should have one result with valid data
        assert results[0]["rrf_score"] > 0.0
        assert results[0]["bm25_rank"] > 0
        assert results[0]["semantic_rank"] > 0

    def test_rrf_search_output_format(self):
        """Test that RRF search output format is correct."""
        stdout, stderr, code = run_rrf_search("bear", k=60, limit=3)
        assert code == 0
        
        results = parse_rrf_search_results(stdout)
        assert len(results) == 3
        
        for i, result in enumerate(results, start=1):
            # Rank should match position
            assert result["rank"] == i
            
            # RRF score should be formatted to 3 decimal places
            # (we can verify this by checking the parsed float is reasonable)
            assert 0.0 < result["rrf_score"] < 1.0
            
            # Ranks should be positive integers
            assert isinstance(result["bm25_rank"], int)
            assert isinstance(result["semantic_rank"], int)
            assert result["bm25_rank"] > 0
            assert result["semantic_rank"] > 0

    def test_rrf_search_rank_information(self):
        """Test that RRF search displays rank information from both methods."""
        stdout, stderr, code = run_rrf_search("paddington bear", limit=5)
        assert code == 0
        
        results = parse_rrf_search_results(stdout)
        assert len(results) == 5
        
        # Each result should have ranks from both BM25 and semantic search
        for result in results:
            assert "bm25_rank" in result
            assert "semantic_rank" in result
            # Ranks should be different for most results (different ranking methods)
            # We don't assert inequality as some docs might rank the same

    def test_rrf_search_deterministic(self):
        """Test that RRF search produces deterministic results."""
        stdout1, stderr1, code1 = run_rrf_search("magical", k=60, limit=5)
        stdout2, stderr2, code2 = run_rrf_search("magical", k=60, limit=5)
        
        assert code1 == 0
        assert code2 == 0
        
        results1 = parse_rrf_search_results(stdout1)
        results2 = parse_rrf_search_results(stdout2)
        
        # Results should be identical
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1["title"] == r2["title"]
            assert abs(r1["rrf_score"] - r2["rrf_score"]) < 0.0001
            assert r1["bm25_rank"] == r2["bm25_rank"]
            assert r1["semantic_rank"] == r2["semantic_rank"]

    def test_rrf_search_k_validation_negative(self):
        """Test that CLI rejects negative k values."""
        stdout, stderr, code = run_rrf_search("test", k=-10, limit=5)
        assert code == 1  # Should exit with error
        assert "Error: k must be a positive integer" in stderr
        assert "got -10" in stderr

    def test_rrf_search_k_validation_zero(self):
        """Test that CLI rejects k=0."""
        stdout, stderr, code = run_rrf_search("test", k=0, limit=5)
        assert code == 1  # Should exit with error
        assert "Error: k must be a positive integer" in stderr
        assert "got 0" in stderr

    def test_rrf_search_k_validation_positive(self):
        """Test that CLI accepts positive k values."""
        stdout, stderr, code = run_rrf_search("bear", k=100, limit=2)
        assert code == 0
        assert "k=100" in stdout
        
        results = parse_rrf_search_results(stdout)
        assert len(results) == 2

    def test_rrf_search_limit_validation_negative(self):
        """Test that CLI rejects negative limit values."""
        stdout, stderr, code = run_rrf_search("test", k=60, limit=-5)
        assert code == 1  # Should exit with error
        assert "Error: limit must be a positive integer" in stderr
        assert "got -5" in stderr

    def test_rrf_search_limit_validation_zero(self):
        """Test that CLI rejects limit=0."""
        stdout, stderr, code = run_rrf_search("test", k=60, limit=0)
        assert code == 1  # Should exit with error
        assert "Error: limit must be a positive integer" in stderr
        assert "got 0" in stderr

    def test_rrf_search_limit_validation_positive(self):
        """Test that CLI accepts positive limit values."""
        stdout, stderr, code = run_rrf_search("bear", k=60, limit=7)
        assert code == 0
        assert "Top 7 results" in stdout
        
        results = parse_rrf_search_results(stdout)
        assert len(results) == 7

    def test_rrf_search_large_limit(self):
        """Test RRF search with a large limit value."""
        stdout, stderr, code = run_rrf_search("bear", k=60, limit=50)
        assert code == 0
        
        results = parse_rrf_search_results(stdout)
        # May return fewer results if dataset doesn't have 50 matching docs
        assert len(results) > 0
        assert len(results) <= 50

    def test_rrf_search_different_k_values(self):
        """Test that different k values produce different RRF scores."""
        stdout1, stderr1, code1 = run_rrf_search("magical bear", k=10, limit=3)
        stdout2, stderr2, code2 = run_rrf_search("magical bear", k=100, limit=3)
        
        assert code1 == 0
        assert code2 == 0
        
        results1 = parse_rrf_search_results(stdout1)
        results2 = parse_rrf_search_results(stdout2)
        
        # Same documents should be returned (same query)
        assert len(results1) == 3
        assert len(results2) == 3
        
        # But RRF scores should be different (different k values)
        # Lower k should produce higher scores
        for r1, r2 in zip(results1, results2):
            # Allow for possibility of same doc appearing
            if r1["title"] == r2["title"]:
                # Score with k=10 should be higher than k=100
                assert r1["rrf_score"] > r2["rrf_score"]

    def test_rrf_search_query_variations(self):
        """Test RRF search with different query types."""
        # Single word query
        stdout1, stderr1, code1 = run_rrf_search("bear", limit=3)
        assert code1 == 0
        results1 = parse_rrf_search_results(stdout1)
        assert len(results1) == 3
        
        # Multi-word query
        stdout2, stderr2, code2 = run_rrf_search("magical bear adventure", limit=3)
        assert code2 == 0
        results2 = parse_rrf_search_results(stdout2)
        assert len(results2) == 3


class TestRRFSearchMethodValidation:
    """Tests for method-level validation in HybridSearch.rrf_search()"""

    def test_method_k_validation_negative(self):
        """Test that the rrf_search method rejects negative k values."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        import pytest
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        
        # Should raise ValueError for negative k
        with pytest.raises(ValueError, match="k must be a positive integer"):
            hs.rrf_search("test", k=-10, limit=5)

    def test_method_k_validation_zero(self):
        """Test that the rrf_search method rejects k=0."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        import pytest
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        
        # Should raise ValueError for k=0
        with pytest.raises(ValueError, match="k must be a positive integer"):
            hs.rrf_search("test", k=0, limit=5)

    def test_method_k_validation_positive(self):
        """Test that the rrf_search method accepts positive k values."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        
        # Should accept positive k values
        results = hs.rrf_search("test", k=50, limit=2)
        assert len(results) <= 2
        
        # Verify result format
        for doc_id, scores in results:
            assert "rrf" in scores
            assert "bm25_rank" in scores
            assert "semantic_rank" in scores

    def test_method_limit_validation_negative(self):
        """Test that the rrf_search method rejects negative limit values."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        import pytest
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        
        # Should raise ValueError for negative limit
        with pytest.raises(ValueError, match="limit must be a positive integer"):
            hs.rrf_search("test", k=60, limit=-5)

    def test_method_limit_validation_zero(self):
        """Test that the rrf_search method rejects limit=0."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        import pytest
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        
        # Should raise ValueError for limit=0
        with pytest.raises(ValueError, match="limit must be a positive integer"):
            hs.rrf_search("test", k=60, limit=0)

    def test_method_k_validation_negative(self):
        """Test that the rrf_search method rejects negative k values."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        import pytest
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        
        # Should raise ValueError for negative k
        with pytest.raises(ValueError, match="k must be a positive integer"):
            hs.rrf_search("test", k=-10, limit=5)

    def test_method_k_validation_zero(self):
        """Test that the rrf_search method rejects k=0."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        import pytest
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        
        # Should raise ValueError for k=0
        with pytest.raises(ValueError, match="k must be a positive integer"):
            hs.rrf_search("test", k=0, limit=5)

    def test_method_k_validation_positive(self):
        """Test that the rrf_search method accepts positive k values."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        
        # Should not raise for positive k values
        results = hs.rrf_search("test", k=100, limit=5)
        assert len(results) <= 5


class TestRRFSearchMethod:
    """Tests for the rrf_search method implementation."""

    def test_rrf_search_basic_functionality(self):
        """Test basic RRF search returns results in correct format."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        results = hs.rrf_search("magical bear", k=60, limit=5)
        
        # Should return results
        assert len(results) > 0
        assert len(results) <= 5
        
        # Check result format: list of (doc_id, scores_dict) tuples
        for doc_id, scores in results:
            assert isinstance(doc_id, int)
            assert isinstance(scores, dict)
            assert "rrf" in scores
            assert "bm25_rank" in scores
            assert "semantic_rank" in scores
            assert isinstance(scores["rrf"], float)
            assert scores["rrf"] > 0.0

    def test_rrf_search_different_k_values(self):
        """Test that different k values affect RRF scores."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        
        # Get results with different k values
        results_k30 = hs.rrf_search("bear", k=30, limit=5)
        results_k60 = hs.rrf_search("bear", k=60, limit=5)
        results_k100 = hs.rrf_search("bear", k=100, limit=5)
        
        # All should return results
        assert len(results_k30) > 0
        assert len(results_k60) > 0
        assert len(results_k100) > 0
        
        # Lower k values should produce higher RRF scores (since 1/(k+rank) is larger)
        # Check first result's RRF score
        score_k30 = results_k30[0][1]["rrf"]
        score_k60 = results_k60[0][1]["rrf"]
        score_k100 = results_k100[0][1]["rrf"]
        
        assert score_k30 > score_k60
        assert score_k60 > score_k100

    def test_rrf_search_result_ordering(self):
        """Test that results are ordered by RRF score descending."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        results = hs.rrf_search("adventure", k=60, limit=10)
        
        assert len(results) > 1
        
        # Check that RRF scores are in descending order
        rrf_scores = [scores["rrf"] for _, scores in results]
        assert rrf_scores == sorted(rrf_scores, reverse=True)

    def test_rrf_search_score_calculation(self):
        """Test that RRF scores are calculated correctly."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        k = 60
        results = hs.rrf_search("paddington", k=k, limit=3)
        
        assert len(results) > 0
        
        # For each result, verify RRF score calculation
        for doc_id, scores in results:
            rrf_score = scores["rrf"]
            bm25_rank = scores["bm25_rank"]
            semantic_rank = scores["semantic_rank"]
            
            # Calculate expected RRF score
            expected_rrf = 0.0
            if bm25_rank is not None:
                expected_rrf += 1.0 / (k + bm25_rank)
            if semantic_rank is not None:
                expected_rrf += 1.0 / (k + semantic_rank)
            
            # Should match within floating point tolerance
            assert abs(rrf_score - expected_rrf) < 0.0001

    def test_rrf_search_rank_information(self):
        """Test that rank information is correctly included in results."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        results = hs.rrf_search("bear", k=60, limit=5)
        
        assert len(results) > 0
        
        # Check that ranks are positive integers or None
        for doc_id, scores in results:
            bm25_rank = scores["bm25_rank"]
            semantic_rank = scores["semantic_rank"]
            
            # At least one rank should be present for each result
            assert bm25_rank is not None or semantic_rank is not None
            
            # If present, ranks should be positive integers
            if bm25_rank is not None:
                assert isinstance(bm25_rank, int)
                assert bm25_rank > 0
            if semantic_rank is not None:
                assert isinstance(semantic_rank, int)
                assert semantic_rank > 0

    def test_rrf_search_deterministic_results(self):
        """Test that RRF search produces deterministic results."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        
        # Run the same search multiple times
        results1 = hs.rrf_search("fantasy adventure", k=60, limit=5)
        results2 = hs.rrf_search("fantasy adventure", k=60, limit=5)
        results3 = hs.rrf_search("fantasy adventure", k=60, limit=5)
        
        # All runs should produce identical results
        assert len(results1) == len(results2) == len(results3)
        
        for (doc_id1, scores1), (doc_id2, scores2), (doc_id3, scores3) in zip(results1, results2, results3):
            assert doc_id1 == doc_id2 == doc_id3
            assert abs(scores1["rrf"] - scores2["rrf"]) < 0.0001
            assert abs(scores1["rrf"] - scores3["rrf"]) < 0.0001

    def test_rrf_search_empty_query(self):
        """Test RRF search with empty query raises ValueError."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        import pytest
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        
        # Empty query should raise ValueError from semantic search
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            hs.rrf_search("", k=60, limit=5)

    def test_rrf_search_rare_query(self):
        """Test RRF search with a query that yields few results."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        # Use a very specific query that should match few documents
        results = hs.rrf_search("xyzabc123nonexistent", k=60, limit=5)
        
        # Should return a list (may be empty or have few results)
        assert isinstance(results, list)
        
        # All results should have valid format
        for doc_id, scores in results:
            assert "rrf" in scores
            assert "bm25_rank" in scores
            assert "semantic_rank" in scores

    def test_rrf_search_limit_behavior(self):
        """Test that limit parameter correctly restricts results."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        
        # Test various limits
        results_3 = hs.rrf_search("adventure", k=60, limit=3)
        results_5 = hs.rrf_search("adventure", k=60, limit=5)
        results_10 = hs.rrf_search("adventure", k=60, limit=10)
        
        assert len(results_3) <= 3
        assert len(results_5) <= 5
        assert len(results_10) <= 10
        
        # Smaller limits should be subsets of larger limits (same top results)
        if len(results_3) >= 3 and len(results_5) >= 5:
            # First 3 results should match
            for i in range(3):
                assert results_3[i][0] == results_5[i][0]

    def test_rrf_search_both_searches_contribute(self):
        """Test that both BM25 and semantic search contribute to results."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        results = hs.rrf_search("magical bear adventure", k=60, limit=10)
        
        assert len(results) > 0
        
        # Check that we have results with different rank patterns
        # Some documents should appear in both searches
        both_searches = []
        only_bm25 = []
        only_semantic = []
        
        for doc_id, scores in results:
            if scores["bm25_rank"] is not None and scores["semantic_rank"] is not None:
                both_searches.append(doc_id)
            elif scores["bm25_rank"] is not None:
                only_bm25.append(doc_id)
            elif scores["semantic_rank"] is not None:
                only_semantic.append(doc_id)
        
        # Should have at least some documents in both searches
        # (This is query-dependent but likely for this query)
        assert len(both_searches) > 0

    def test_rrf_search_large_k_value(self):
        """Test RRF search with large k value."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        results = hs.rrf_search("bear", k=1000, limit=5)
        
        assert len(results) > 0
        
        # With large k, RRF scores should be smaller
        for doc_id, scores in results:
            assert scores["rrf"] > 0.0
            assert scores["rrf"] < 0.1  # Should be quite small with k=1000

    def test_rrf_search_small_k_value(self):
        """Test RRF search with small k value."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        results = hs.rrf_search("bear", k=1, limit=5)
        
        assert len(results) > 0
        
        # With small k, RRF scores should be larger
        for doc_id, scores in results:
            assert scores["rrf"] > 0.0

    def test_rrf_search_expansion_factor(self):
        """Test that expansion factor allows better fusion."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        
        # Even with small limit, should get good results due to expansion
        results = hs.rrf_search("magical bear", k=60, limit=1)
        
        assert len(results) == 1
        
        # Should have rank information from expanded searches
        doc_id, scores = results[0]
        assert "bm25_rank" in scores
        assert "semantic_rank" in scores

    def test_rrf_search_returns_document_ids(self):
        """Test that RRF search returns valid document IDs."""
        from cli.lib.semantic_search import load_movies_dataset
        from cli.hybrid_search_cli import HybridSearch
        
        docs, exc, _ = load_movies_dataset()
        assert exc is None
        
        hs = HybridSearch(docs)
        results = hs.rrf_search("adventure", k=60, limit=5)
        
        assert len(results) > 0
        
        # Collect all document IDs from dataset
        all_doc_ids = {doc.get("id") for doc in docs}
        
        # All returned doc IDs should be valid
        for doc_id, scores in results:
            assert doc_id in all_doc_ids


class TestRRFSearchEnhancement:
    """Tests for the --enhance functionality in rrf-search command."""

    def test_enhance_spell_with_correct_query(self):
        """Test --enhance spell with correctly spelled query (no changes expected)."""
        stdout, stderr, code = run_rrf_search_with_enhance("magical bear", "spell", limit=3)
        
        # Should succeed even if API key is missing (will fallback to original query)
        assert code == 0
        
        # Should have results
        assert "Top 3 results" in stdout
        
        # If spell correction worked and query was correct, should not see "Enhanced query"
        # OR it will be present but show no change, OR we'll see a warning about API failure
        # Any of these outcomes is acceptable for a correctly spelled query

    def test_enhance_spell_with_typo_query(self):
        """Test --enhance spell with query containing typo."""
        # Use "paddingon" instead of "paddington"
        stdout, stderr, code = run_rrf_search_with_enhance("paddingon", "spell", limit=3)
        
        # Should succeed (may use original query if API fails)
        assert code == 0
        
        # Should have results (even if using original query as fallback)
        assert "results for query:" in stdout.lower()

    def test_enhance_spell_missing_api_key(self):
        """Test --enhance spell when GEMINI_API_KEY is not set."""
        import os
        
        # Skip this test if .env file exists with API key
        # (load_dotenv will load it regardless of environment variables)
        dotenv_path = PROJECT_ROOT / ".env"
        if dotenv_path.exists():
            with open(dotenv_path) as f:
                if "GEMINI_API_KEY" in f.read():
                    pytest.skip("Skipping test - GEMINI_API_KEY exists in .env file")
        
        # Create minimal environment without GEMINI_API_KEY
        # Keep PATH to find python executable
        env = {"PATH": os.environ.get("PATH", "")}
        
        stdout, stderr, code = run_rrf_search_with_enhance("magical bear", "spell", limit=3, env=env)
        
        # Should exit with error code 1 when API key is missing
        assert code == 1
        
        # Should have error message about missing API key
        assert "GEMINI_API_KEY" in stderr

    def test_enhance_spell_api_failure_fallback(self):
        """Test that spell correction falls back gracefully on API failure."""
        # This test will work even if API fails (rate limit, network error, etc.)
        stdout, stderr, code = run_rrf_search_with_enhance("bear", "spell", limit=2)
        
        # Should succeed (uses fallback)
        assert code == 0
        
        # Should have results
        assert "results for query:" in stdout.lower()
        
        # If API failed, should see warning in stderr OR if it succeeded, should see results
        # Either way, the command should complete successfully

    def test_enhance_spell_output_format(self):
        """Test that enhanced query output follows expected format."""
        stdout, stderr, code = run_rrf_search_with_enhance("briish bear", "spell", limit=2)
        
        # Should succeed
        assert code == 0
        
        # If enhancement worked, should see the enhancement message
        # Format: "Enhanced query (spell): 'original' -> 'corrected'"
        # But this is optional if API fails or query is already correct
        
        # Should always have search results
        assert "results for query:" in stdout.lower()

    def test_enhance_spell_with_different_k_values(self):
        """Test --enhance spell works with different k parameter values."""
        stdout, stderr, code = run_rrf_search_with_enhance("bear", "spell", k=30, limit=2)
        
        assert code == 0
        assert "k=30" in stdout

    def test_enhance_spell_preserves_search_functionality(self):
        """Test that spell enhancement doesn't break normal RRF search."""
        # Run search with enhancement
        stdout_enhanced, stderr_enhanced, code_enhanced = run_rrf_search_with_enhance(
            "paddington", "spell", limit=3
        )
        
        # Run search without enhancement
        stdout_normal, stderr_normal, code_normal = run_rrf_search(
            "paddington", limit=3
        )
        
        # Both should succeed
        assert code_enhanced == 0
        assert code_normal == 0
        
        # Both should return results
        results_enhanced = parse_rrf_search_results(stdout_enhanced)
        results_normal = parse_rrf_search_results(stdout_normal)
        
        # Should have same number of results
        assert len(results_enhanced) == len(results_normal)
        
        # For a correctly spelled query, results should be similar
        # (may differ slightly if enhancement added/removed whitespace)

    def test_enhance_invalid_choice(self):
        """Test that invalid --enhance choice is rejected."""
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        python_exec = str(venv_python) if venv_python.exists() else "python3"
        
        cmd = [python_exec, "cli/hybrid_search_cli.py", "rrf-search", "bear", "--enhance", "invalid"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        
        # Should fail with non-zero exit code
        assert result.returncode != 0
        
        # Should have error message about invalid choice
        assert "invalid" in result.stderr.lower() or "choice" in result.stderr.lower()

    def test_enhance_rewrite_basic_functionality(self):
        """Test basic rewrite enhancement functionality with vague query."""
        stdout, stderr, code = run_rrf_search_with_enhance("action movie", "rewrite", limit=3)
        
        assert code == 0
        # Should have some output
        assert len(stdout) > 0
        
        # Should have search results
        results = parse_rrf_search_results(stdout)
        assert len(results) > 0

    def test_enhance_rewrite_output_format(self):
        """Test that rewrite enhancement shows proper output format."""
        stdout, stderr, code = run_rrf_search_with_enhance("space movie", "rewrite", limit=2)
        
        assert code == 0
        
        # Check for enhancement output format
        # Should contain either:
        # 1. "Enhanced query (rewrite):" if query was rewritten
        # 2. Normal search output if query was already good
        if "enhanced query" in stdout.lower():
            assert "rewrite" in stdout.lower()
            assert "space movie" in stdout.lower()  # Original query
            assert "->" in stdout  # Transformation arrow

    def test_enhance_rewrite_fallback_on_api_failure(self):
        """Test that rewrite falls back to original query if API fails."""
        # Use a query that should work even without enhancement
        stdout, stderr, code = run_rrf_search_with_enhance("inception", "rewrite", limit=2)
        
        # Should succeed even if API fails
        assert code == 0
        
        # Should have search results (either with enhanced or original query)
        results = parse_rrf_search_results(stdout)
        assert len(results) > 0

    def test_enhance_rewrite_validates_length(self):
        """Test that rewrite validates enhanced query length."""
        # Test with a very short query that might get expanded
        stdout, stderr, code = run_rrf_search_with_enhance("car", "rewrite", limit=2)
        
        # Should succeed
        assert code == 0
        
        # If rewriting occurred, check it's within acceptable limits
        if "enhanced query" in stdout.lower():
            # Extract the enhanced query from output
            lines = stdout.split('\n')
            for line in lines:
                if "enhanced query" in line.lower() and "->" in line:
                    # Parse: Enhanced query (rewrite): 'car' -> 'enhanced text'
                    parts = line.split("->")
                    if len(parts) == 2:
                        enhanced = parts[1].strip().strip("'\"")
                        # Should respect validation: max(5x original, 200 chars)
                        max_allowed = max(len("car") * 5, 200)
                        assert len(enhanced) <= max_allowed

    def test_enhance_rewrite_preserves_search_functionality(self):
        """Test that rewrite enhancement doesn't break normal RRF search."""
        # Run search with rewrite enhancement
        stdout_enhanced, stderr_enhanced, code_enhanced = run_rrf_search_with_enhance(
            "batman", "rewrite", limit=3
        )
        
        # Run search without enhancement
        stdout_normal, stderr_normal, code_normal = run_rrf_search(
            "batman", limit=3
        )
        
        # Both should succeed
        assert code_enhanced == 0
        assert code_normal == 0
        
        # Both should return results
        results_enhanced = parse_rrf_search_results(stdout_enhanced)
        results_normal = parse_rrf_search_results(stdout_normal)
        
        # Both should have results
        assert len(results_enhanced) > 0
        assert len(results_normal) > 0

    def test_enhance_rewrite_with_different_k_values(self):
        """Test --enhance rewrite works with different k parameter values."""
        stdout, stderr, code = run_rrf_search_with_enhance("comedy", "rewrite", k=25, limit=2)
        
        assert code == 0
        assert "k=25" in stdout

    def test_enhance_rewrite_specific_query(self):
        """Test rewrite enhancement makes queries more specific."""
        # Use a vague query that could be improved
        stdout, stderr, code = run_rrf_search_with_enhance("old movie", "rewrite", limit=3)
        
        assert code == 0
        
        # Should have search results
        results = parse_rrf_search_results(stdout)
        assert len(results) > 0
        
        # If enhancement occurred, the enhanced query should be present
        # But this is optional if API fails or query is already acceptable
        assert "results for query:" in stdout.lower()

    def test_enhance_rewrite_vs_spell_difference(self):
        """Test that rewrite and spell produce different behaviors."""
        query = "scifi"
        
        # Run with spell enhancement (should fix typo: scifi -> sci-fi or sci fi)
        stdout_spell, _, code_spell = run_rrf_search_with_enhance(query, "spell", limit=2)
        
        # Run with rewrite enhancement (should expand to more specific query)
        stdout_rewrite, _, code_rewrite = run_rrf_search_with_enhance(query, "rewrite", limit=2)
        
        # Both should succeed
        assert code_spell == 0
        assert code_rewrite == 0
        
        # Both should have results
        assert len(parse_rrf_search_results(stdout_spell)) > 0
        assert len(parse_rrf_search_results(stdout_rewrite)) > 0
        
        # If both enhanced, they should mention their respective methods
        if "enhanced query" in stdout_spell.lower():
            assert "spell" in stdout_spell.lower()
        if "enhanced query" in stdout_rewrite.lower():
            assert "rewrite" in stdout_rewrite.lower()

    def test_enhance_expand_basic_functionality(self):
        """Test basic expand enhancement functionality."""
        stdout, stderr, code = run_rrf_search_with_enhance("bear movie", "expand", limit=3)
        
        assert code == 0
        # Should have some output
        assert len(stdout) > 0
        
        # Should have search results
        results = parse_rrf_search_results(stdout)
        assert len(results) > 0

    def test_enhance_expand_output_format(self):
        """Test that expand enhancement shows proper output format."""
        stdout, stderr, code = run_rrf_search_with_enhance("horror", "expand", limit=2)
        
        assert code == 0
        
        # Check for enhancement output format
        # Should contain either:
        # 1. "Enhanced query (expand):" if query was expanded
        # 2. Normal search output if expansion wasn't needed
        if "enhanced query" in stdout.lower():
            assert "expand" in stdout.lower()
            assert "horror" in stdout.lower()  # Original query should be in output
            assert "->" in stdout  # Transformation arrow

    def test_enhance_expand_appends_to_original(self):
        """Test that expand appends terms to original query rather than replacing."""
        stdout, stderr, code = run_rrf_search_with_enhance("action", "expand", limit=2)
        
        assert code == 0
        
        # If expansion occurred, the output should show the original + expansion
        if "enhanced query" in stdout.lower():
            lines = stdout.split('\n')
            for line in lines:
                if "enhanced query" in line.lower() and "->" in line:
                    # Extract enhanced query
                    parts = line.split("->")
                    if len(parts) == 2:
                        enhanced = parts[1].strip().strip("'\"")
                        # Enhanced query should start with original query
                        assert enhanced.startswith("action"), f"Enhanced query '{enhanced}' should start with 'action'"

    def test_enhance_expand_validates_length(self):
        """Test that expand validates enhanced query length."""
        # Test with a short query
        stdout, stderr, code = run_rrf_search_with_enhance("war", "expand", limit=2)
        
        # Should succeed
        assert code == 0
        
        # If expansion occurred, check it's within acceptable limits
        if "enhanced query" in stdout.lower():
            lines = stdout.split('\n')
            for line in lines:
                if "enhanced query" in line.lower() and "->" in line:
                    # Parse: Enhanced query (expand): 'war' -> 'war battle conflict...'
                    parts = line.split("->")
                    if len(parts) == 2:
                        enhanced = parts[1].strip().strip("'\"")
                        # Should respect validation: max(6x original, 250 chars)
                        max_allowed = max(len("war") * 6, 250)
                        assert len(enhanced) <= max_allowed

    def test_enhance_expand_fallback_on_api_failure(self):
        """Test that expand falls back to original query if API fails."""
        # Use a query that should work even without enhancement
        stdout, stderr, code = run_rrf_search_with_enhance("comedy", "expand", limit=2)
        
        # Should succeed even if API fails
        assert code == 0
        
        # Should have search results (either with enhanced or original query)
        results = parse_rrf_search_results(stdout)
        assert len(results) > 0

    def test_enhance_expand_preserves_search_functionality(self):
        """Test that expand enhancement doesn't break normal RRF search."""
        # Run search with expand enhancement
        stdout_enhanced, stderr_enhanced, code_enhanced = run_rrf_search_with_enhance(
            "thriller", "expand", limit=3
        )
        
        # Run search without enhancement
        stdout_normal, stderr_normal, code_normal = run_rrf_search(
            "thriller", limit=3
        )
        
        # Both should succeed
        assert code_enhanced == 0
        assert code_normal == 0
        
        # Both should return results
        results_enhanced = parse_rrf_search_results(stdout_enhanced)
        results_normal = parse_rrf_search_results(stdout_normal)
        
        # Both should have results
        assert len(results_enhanced) > 0
        assert len(results_normal) > 0

    def test_enhance_expand_with_different_k_values(self):
        """Test --enhance expand works with different k parameter values."""
        stdout, stderr, code = run_rrf_search_with_enhance("adventure", "expand", k=30, limit=2)
        
        assert code == 0
        assert "k=30" in stdout

    def test_enhance_expand_adds_related_terms(self):
        """Test that expand adds related terms to make search more comprehensive."""
        # Use a specific query
        stdout, stderr, code = run_rrf_search_with_enhance("scary", "expand", limit=3)
        
        assert code == 0
        
        # Should have search results
        results = parse_rrf_search_results(stdout)
        assert len(results) > 0
        
        # If enhancement occurred, the enhanced query should be present
        assert "results for query:" in stdout.lower()

    def test_enhance_expand_vs_rewrite_difference(self):
        """Test that expand appends while rewrite replaces."""
        query = "bear"
        
        # Run with expand (should append related terms)
        stdout_expand, _, code_expand = run_rrf_search_with_enhance(query, "expand", limit=2)
        
        # Run with rewrite (should replace with more specific query)
        stdout_rewrite, _, code_rewrite = run_rrf_search_with_enhance(query, "rewrite", limit=2)
        
        # Both should succeed
        assert code_expand == 0
        assert code_rewrite == 0
        
        # Both should have results
        assert len(parse_rrf_search_results(stdout_expand)) > 0
        assert len(parse_rrf_search_results(stdout_rewrite)) > 0
        
        # Check their respective methods are mentioned
        if "enhanced query" in stdout_expand.lower():
            assert "expand" in stdout_expand.lower()
            # For expand, original query should be at the start
            lines = stdout_expand.split('\n')
            for line in lines:
                if "enhanced query" in line.lower() and "->" in line:
                    parts = line.split("->")
                    if len(parts) == 2:
                        enhanced = parts[1].strip().strip("'\"")
                        assert enhanced.startswith(query), f"Expand should append, not replace"
        
        if "enhanced query" in stdout_rewrite.lower():
            assert "rewrite" in stdout_rewrite.lower()


def run_rrf_search_with_rerank(query, rerank_method, k=None, limit=None, env=None):
    """Helper to run the rrf-search command with --rerank-method flag."""
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    python_exec = str(venv_python) if venv_python.exists() else "python3"
    
    cmd = [python_exec, "cli/hybrid_search_cli.py", "rrf-search", query]
    cmd.extend(["--rerank-method", rerank_method])
    if k is not None:
        cmd.extend(["--k", str(k)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, env=env)
    return result.stdout, result.stderr, result.returncode


class TestRRFSearchReranking:
    """Tests for the --rerank-method functionality in rrf-search command."""

    def test_rerank_individual_basic_functionality(self):
        """Test basic individual reranking functionality."""
        stdout, stderr, code = run_rrf_search_with_rerank("bear movie", "individual", limit=3)
        
        assert code == 0
        # Should have output
        assert len(stdout) > 0
        
        # Should show reranking message
        assert "Reranking" in stdout and "results to return top 3" in stdout
        
        # Should have results
        assert "LLM Reranked Results" in stdout

    def test_rerank_individual_output_format(self):
        """Test that individual reranking shows proper output format."""
        stdout, stderr, code = run_rrf_search_with_rerank("comedy", "individual", limit=2)
        
        assert code == 0
        
        # Should have reranking message
        assert "Reranking" in stdout and "results to return top 2" in stdout
        
        # Should have LLM reranking results header
        assert "LLM Reranked Results" in stdout and "for 'comedy'" in stdout
        
        # Should show Rerank Score
        assert "Rerank Score:" in stdout
        
        # Should show RRF Score
        assert "RRF Score:" in stdout

    def test_rerank_individual_missing_api_key(self):
        """Test --rerank-method individual when GEMINI_API_KEY is not set."""
        import os
        
        # Skip this test if .env file exists with API key
        dotenv_path = PROJECT_ROOT / ".env"
        if dotenv_path.exists():
            with open(dotenv_path) as f:
                if "GEMINI_API_KEY" in f.read():
                    pytest.skip("Skipping test - GEMINI_API_KEY exists in .env file")
        
        # Create minimal environment without GEMINI_API_KEY
        env = {"PATH": os.environ.get("PATH", "")}
        
        stdout, stderr, code = run_rrf_search_with_rerank("action", "individual", limit=2, env=env)
        
        # Should exit with error code 1 when API key is missing
        assert code == 1
        
        # Should have error message about missing API key
        assert "GEMINI_API_KEY" in stderr

    def test_rerank_individual_with_different_limits(self):
        """Test --rerank-method individual works with different limit values."""
        stdout, stderr, code = run_rrf_search_with_rerank("thriller", "individual", limit=5)
        
        assert code == 0
        assert "Reranking" in stdout and "results to return top 5" in stdout

    def test_rerank_individual_with_different_k_values(self):
        """Test --rerank-method individual works with different k parameter values."""
        stdout, stderr, code = run_rrf_search_with_rerank("adventure", "individual", k=30, limit=2)
        
        assert code == 0
        assert "k=30" in stdout
        assert "Reranking" in stdout and "results to return top 2" in stdout

    def test_rerank_individual_score_format(self):
        """Test that rerank scores are formatted correctly (X.XXX/10)."""
        stdout, stderr, code = run_rrf_search_with_rerank("horror", "individual", limit=2)
        
        assert code == 0
        
        # Look for rerank score in output
        if "Rerank Score:" in stdout:
            # Should be formatted as X.XXX/10
            lines = stdout.split('\n')
            for line in lines:
                if "Rerank Score:" in line:
                    # Extract score part (e.g., "10.000/10")
                    score_part = line.split("Rerank Score:")[1].strip()
                    assert "/10" in score_part
                    # Should have 3 decimal places
                    score_value = score_part.split("/10")[0].strip()
                    parts = score_value.split(".")
                    if len(parts) == 2:
                        assert len(parts[1]) == 3  # 3 decimal places

    def test_rerank_individual_gathers_more_results(self):
        """Test that individual reranking gathers 5x more results initially."""
        # This test verifies the behavior indirectly by checking that reranking produces results
        # In practice, with limit=2, it should gather 10 results to rerank
        stdout, stderr, code = run_rrf_search_with_rerank("family", "individual", limit=2)
        
        assert code == 0
        # Should have reranked results
        assert "Rerank Score:" in stdout
        
        # Count the number of results shown (should be 2)
        result_count = stdout.count("Rerank Score:")
        assert result_count <= 2  # Should show at most the limit

    def test_rerank_individual_preserves_search_functionality(self):
        """Test that individual reranking doesn't break normal RRF search."""
        # Run search with reranking
        stdout_reranked, stderr_reranked, code_reranked = run_rrf_search_with_rerank(
            "animation", "individual", limit=3
        )
        
        # Run search without reranking
        stdout_normal, stderr_normal, code_normal = run_rrf_search(
            "animation", limit=3
        )
        
        # Both should succeed
        assert code_reranked == 0
        assert code_normal == 0
        
        # Both should have results
        assert "results for query:" in stdout_reranked.lower() or "LLM Reranked Results" in stdout_reranked
        assert "results for query:" in stdout_normal.lower()

    def test_rerank_individual_invalid_choice(self):
        """Test that invalid --rerank-method choice is rejected."""
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        python_exec = str(venv_python) if venv_python.exists() else "python3"
        
        cmd = [python_exec, "cli/hybrid_search_cli.py", "rrf-search", "bear", "--rerank-method", "invalid"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        
        # Should fail with non-zero exit code
        assert result.returncode != 0
        
        # Should have error message about invalid choice
        assert "invalid" in result.stderr.lower() or "choice" in result.stderr.lower()

    def test_rerank_individual_handles_api_errors(self):
        """Test that individual reranking handles API errors gracefully."""
        # This test verifies the feature continues even if some API calls fail
        # We can't easily simulate API failures, but we can verify the feature works end-to-end
        stdout, stderr, code = run_rrf_search_with_rerank("mystery", "individual", limit=2)
        
        # Should complete successfully or show warnings but still return results
        # Either way, should not crash
        assert code == 0 or "Warning:" in stderr

