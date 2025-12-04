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
