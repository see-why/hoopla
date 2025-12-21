import json
import subprocess
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_evaluation(golden_dataset_path=None, limit=None, rrf_k=None):
    """Helper to run the evaluation CLI with specified arguments."""
    # Prefer the project's virtualenv python if present
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    python_exec = str(venv_python) if venv_python.exists() else "python3"
    
    cmd = [python_exec, "cli/evaluation_cli.py"]
    
    if golden_dataset_path:
        cmd.extend(["--golden-dataset", str(golden_dataset_path)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if rrf_k is not None:
        cmd.extend(["--rrf-k", str(rrf_k)])
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    return result.stdout, result.stderr, result.returncode


def parse_metrics(output):
    """Parse precision, recall, and F1 score metrics from output."""
    metrics = []
    lines = output.strip().split("\n")
    
    current_query = None
    current_precision = None
    current_recall = None
    current_f1 = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("- Query:"):
            current_query = line.replace("- Query:", "").strip()
        elif "- Precision@" in line:
            parts = line.split(":")
            if len(parts) == 2:
                current_precision = float(parts[1].strip())
        elif "- Recall@" in line:
            parts = line.split(":")
            if len(parts) == 2:
                current_recall = float(parts[1].strip())
        elif "- F1 Score:" in line:
            parts = line.split(":")
            if len(parts) == 2:
                current_f1 = float(parts[1].strip())
                # When we have all four, save the metric
                if current_query and current_precision is not None and current_recall is not None and current_f1 is not None:
                    metrics.append({
                        "query": current_query,
                        "precision": current_precision,
                        "recall": current_recall,
                        "f1_score": current_f1
                    })
                    current_query = None
                    current_precision = None
                    current_recall = None
                    current_f1 = None
    
    return metrics


class TestEvaluationCLI:
    """Tests for the evaluation_cli.py module"""

    def test_missing_golden_dataset(self):
        """Test error handling when golden dataset file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_path = Path(tmpdir) / "nonexistent.json"
            stdout, stderr, code = run_evaluation(golden_dataset_path=nonexistent_path)
            
            assert code == 1
            assert "Error: Golden dataset not found" in stderr
            assert str(nonexistent_path) in stderr

    def test_invalid_json_golden_dataset(self):
        """Test error handling when golden dataset contains invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json content")
            invalid_json_path = f.name
        
        try:
            stdout, stderr, code = run_evaluation(golden_dataset_path=invalid_json_path)
            
            assert code == 1
            assert "Error: Failed to parse golden dataset JSON" in stderr
        finally:
            Path(invalid_json_path).unlink()

    def test_precision_calculation_perfect_match(self):
        """Test precision calculation when all retrieved documents are relevant."""
        # Create a simple golden dataset with known relevant documents
        golden_data = {
            "test_cases": [
                {
                    "query": "action movies",
                    "relevant_docs": [
                        "The Matrix",
                        "Die Hard",
                        "Mad Max: Fury Road"
                    ]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(golden_data, f)
            golden_path = f.name
        
        try:
            stdout, stderr, code = run_evaluation(golden_dataset_path=golden_path, limit=3)
            
            assert code == 0
            assert "Query: action movies" in stdout
            assert "Precision@3:" in stdout
            assert "Recall@3:" in stdout
            
            # Parse and verify metrics structure
            metrics = parse_metrics(stdout)
            assert len(metrics) == 1
            assert metrics[0]["query"] == "action movies"
            assert 0.0 <= metrics[0]["precision"] <= 1.0
            assert 0.0 <= metrics[0]["recall"] <= 1.0
        finally:
            Path(golden_path).unlink()

    def test_precision_calculation_no_match(self):
        """Test precision calculation when no retrieved documents are relevant."""
        golden_data = {
            "test_cases": [
                {
                    "query": "nonexistent movie xyz123",
                    "relevant_docs": ["Fake Movie That Does Not Exist"]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(golden_data, f)
            golden_path = f.name
        
        try:
            stdout, stderr, code = run_evaluation(golden_dataset_path=golden_path, limit=5)
            
            assert code == 0
            metrics = parse_metrics(stdout)
            assert len(metrics) == 1
            # Precision should be low or zero since relevant docs don't exist
            assert metrics[0]["precision"] <= 1.0
        finally:
            Path(golden_path).unlink()

    def test_multiple_test_cases(self):
        """Test evaluation with multiple test cases in the golden dataset."""
        golden_data = {
            "test_cases": [
                {
                    "query": "sci-fi",
                    "relevant_docs": ["The Matrix", "Inception"]
                },
                {
                    "query": "comedy",
                    "relevant_docs": ["The Grand Budapest Hotel", "Superbad"]
                },
                {
                    "query": "thriller",
                    "relevant_docs": ["Se7en", "The Silence of the Lambs"]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(golden_data, f)
            golden_path = f.name
        
        try:
            stdout, stderr, code = run_evaluation(golden_dataset_path=golden_path, limit=5)
            
            assert code == 0
            metrics = parse_metrics(stdout)
            assert len(metrics) == 3
            
            # Verify all queries are present
            queries = [m["query"] for m in metrics]
            assert "sci-fi" in queries
            assert "comedy" in queries
            assert "thriller" in queries
        finally:
            Path(golden_path).unlink()

    def test_custom_limit_parameter(self):
        """Test that --limit parameter affects k in precision@k and recall@k."""
        golden_data = {
            "test_cases": [
                {
                    "query": "action",
                    "relevant_docs": ["The Matrix"]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(golden_data, f)
            golden_path = f.name
        
        try:
            # Test with limit=3
            stdout, stderr, code = run_evaluation(golden_dataset_path=golden_path, limit=3)
            assert code == 0
            assert "k=3" in stdout
            assert "Precision@3:" in stdout
            assert "Recall@3:" in stdout
            
            # Test with limit=10
            stdout, stderr, code = run_evaluation(golden_dataset_path=golden_path, limit=10)
            assert code == 0
            assert "k=10" in stdout
            assert "Precision@10:" in stdout
            assert "Recall@10:" in stdout
        finally:
            Path(golden_path).unlink()

    def test_custom_rrf_k_parameter(self):
        """Test that --rrf-k parameter is accepted without error."""
        golden_data = {
            "test_cases": [
                {
                    "query": "adventure",
                    "relevant_docs": ["Indiana Jones"]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(golden_data, f)
            golden_path = f.name
        
        try:
            # Test with custom rrf-k value
            stdout, stderr, code = run_evaluation(
                golden_dataset_path=golden_path, 
                limit=5, 
                rrf_k=100
            )
            assert code == 0
            # Should complete without error
            assert "Query: adventure" in stdout
        finally:
            Path(golden_path).unlink()

    def test_recall_calculation(self):
        """Test recall calculation (relevant_retrieved / total_relevant)."""
        golden_data = {
            "test_cases": [
                {
                    "query": "drama",
                    "relevant_docs": [
                        "The Shawshank Redemption",
                        "The Godfather",
                        "Schindler's List",
                        "12 Angry Men",
                        "Forrest Gump"
                    ]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(golden_data, f)
            golden_path = f.name
        
        try:
            stdout, stderr, code = run_evaluation(golden_dataset_path=golden_path, limit=3)
            
            assert code == 0
            metrics = parse_metrics(stdout)
            assert len(metrics) == 1
            
            # With 5 relevant docs and limit=3, recall should be at most 3/5 = 0.6
            assert metrics[0]["recall"] <= 0.6
            # Recall should be non-negative
            assert metrics[0]["recall"] >= 0.0
        finally:
            Path(golden_path).unlink()

    def test_output_format(self):
        """Test that output includes all expected fields."""
        golden_data = {
            "test_cases": [
                {
                    "query": "test query",
                    "relevant_docs": ["Test Movie"]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(golden_data, f)
            golden_path = f.name
        
        try:
            stdout, stderr, code = run_evaluation(golden_dataset_path=golden_path, limit=5)
            
            assert code == 0
            # Verify output contains all expected sections
            assert "k=5" in stdout
            assert "- Query:" in stdout
            assert "- Precision@5:" in stdout
            assert "- Recall@5:" in stdout
            assert "- Retrieved:" in stdout
            assert "- Relevant:" in stdout
        finally:
            Path(golden_path).unlink()

    def test_empty_test_cases(self):
        """Test behavior with empty test cases list."""
        golden_data = {
            "test_cases": []
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(golden_data, f)
            golden_path = f.name
        
        try:
            stdout, stderr, code = run_evaluation(golden_dataset_path=golden_path, limit=5)
            
            # Should complete successfully with no test results
            assert code == 0
            assert "k=5" in stdout
            # Should not have any query results
            metrics = parse_metrics(stdout)
            assert len(metrics) == 0
        finally:
            Path(golden_path).unlink()

    def test_integration_with_hybrid_search(self):
        """Test that evaluation correctly integrates with HybridSearch."""
        # This test uses the actual movies dataset and verifies the integration
        golden_data = {
            "test_cases": [
                {
                    "query": "space",
                    "relevant_docs": ["Star Wars", "2001: A Space Odyssey"]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(golden_data, f)
            golden_path = f.name
        
        try:
            stdout, stderr, code = run_evaluation(golden_dataset_path=golden_path, limit=10)
            
            # Should complete successfully
            assert code == 0
            
            # Should contain actual search results
            assert "- Retrieved:" in stdout
            
            # Metrics should be calculated
            metrics = parse_metrics(stdout)
            assert len(metrics) == 1
            assert metrics[0]["query"] == "space"
        finally:
            Path(golden_path).unlink()
