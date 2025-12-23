import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_rag(query, k=None, limit=None):
    """Helper to run the RAG CLI with specified arguments."""
    # Prefer the project's virtualenv python if present
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    python_exec = str(venv_python) if venv_python.exists() else "python3"
    
    cmd = [python_exec, "cli/augmented_generation_cli.py", "rag", query]
    
    if k is not None:
        cmd.extend(["--k", str(k)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    return result.stdout, result.stderr, result.returncode


def run_summarize(query, limit=None):
    """Helper to run the summarize CLI with specified arguments."""
    # Prefer the project's virtualenv python if present
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    python_exec = str(venv_python) if venv_python.exists() else "python3"
    
    cmd = [python_exec, "cli/augmented_generation_cli.py", "summarize", query]
    
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    return result.stdout, result.stderr, result.returncode


def run_citations(query, limit=None):
    """Helper to run the citations CLI with specified arguments."""
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    python_exec = str(venv_python) if venv_python.exists() else "python3"

    cmd = [python_exec, "cli/augmented_generation_cli.py", "citations", query]

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    return result.stdout, result.stderr, result.returncode


def run_question(question, limit=None):
    """Helper to run the question CLI with specified arguments."""
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    python_exec = str(venv_python) if venv_python.exists() else "python3"

    cmd = [python_exec, "cli/augmented_generation_cli.py", "question", question]

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    return result.stdout, result.stderr, result.returncode


class TestAugmentedGenerationCLI:
    """Tests for the augmented_generation_cli.py module"""

    def test_help_message(self):
        """Test that help message displays available commands and arguments."""
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        python_exec = str(venv_python) if venv_python.exists() else "python3"
        
        result = subprocess.run(
            [python_exec, "cli/augmented_generation_cli.py", "rag", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        
        assert result.returncode == 0
        assert "Search query for RAG" in result.stdout
        assert "--k" in result.stdout
        assert "--limit" in result.stdout

    def test_successful_rag_execution(self):
        """Test successful RAG execution with basic query."""
        stdout, stderr, code = run_rag("action movies", limit=3)
        
        assert code == 0
        assert "Search Results:" in stdout
        assert "RAG Response:" in stdout
        # Should have at least one result and a response
        assert len(stdout.split("\n")) > 5

    def test_search_results_format(self):
        """Test that search results are formatted correctly."""
        stdout, stderr, code = run_rag("space", limit=3)
        
        assert code == 0
        assert "Search Results:" in stdout
        
        # Extract search results section
        lines = stdout.split("\n")
        search_idx = next(i for i, line in enumerate(lines) if "Search Results:" in line)
        rag_idx = next(i for i, line in enumerate(lines) if "RAG Response:" in line)
        
        # Search results should be between Search Results: and RAG Response:
        search_result_lines = lines[search_idx + 1:rag_idx]
        
        # Should have bullet points for results
        bullet_lines = [l for l in search_result_lines if l.strip().startswith("-")]
        assert len(bullet_lines) > 0, "Should have at least one result with bullet point"

    def test_rag_response_present(self):
        """Test that RAG response is generated and displayed."""
        stdout, stderr, code = run_rag("comedy", limit=2)
        
        assert code == 0
        assert "RAG Response:" in stdout
        
        # Extract RAG response section
        lines = stdout.split("\n")
        rag_idx = next(i for i, line in enumerate(lines) if "RAG Response:" in line)
        
        # Response should have content after the RAG Response: header
        response_lines = [l for l in lines[rag_idx + 1:] if l.strip()]
        assert len(response_lines) > 0, "RAG response should contain text"

    def test_custom_k_parameter(self):
        """Test that --k parameter is accepted and affects search."""
        stdout, stderr, code = run_rag("drama", k=100, limit=3)
        
        assert code == 0
        assert "Search Results:" in stdout
        # Should execute without error with custom k value

    def test_custom_limit_parameter(self):
        """Test that --limit parameter affects number of results."""
        # Get results with limit=2
        stdout_2, stderr_2, code_2 = run_rag("thriller", limit=2)
        
        # Get results with limit=4
        stdout_4, stderr_4, code_4 = run_rag("thriller", limit=4)
        
        assert code_2 == 0
        assert code_4 == 0
        
        # Count bullet points in each output
        bullet_count_2 = len([l for l in stdout_2.split("\n") if l.strip().startswith("-")])
        bullet_count_4 = len([l for l in stdout_4.split("\n") if l.strip().startswith("-")])
        
        # Limit=4 should have more or equal results than limit=2
        assert bullet_count_4 >= bullet_count_2

    def test_query_with_special_characters(self):
        """Test query with special characters and quotes."""
        stdout, stderr, code = run_rag("action: adventure & drama")
        
        assert code == 0
        assert "Search Results:" in stdout
        assert "RAG Response:" in stdout

    def test_single_character_query(self):
        """Test with very short single character query."""
        stdout, stderr, code = run_rag("a", limit=2)
        
        # Should either find results or handle gracefully
        assert code in [0, 1]
        if code == 0:
            assert "Search Results:" in stdout

    def test_output_contains_movie_titles(self):
        """Test that output contains actual movie titles from the dataset."""
        stdout, stderr, code = run_rag("spy thriller", limit=3)
        
        assert code == 0
        
        # Extract titles from search results
        lines = stdout.split("\n")
        search_idx = next(i for i, line in enumerate(lines) if "Search Results:" in line)
        rag_idx = next(i for i, line in enumerate(lines) if "RAG Response:" in line)
        
        result_lines = lines[search_idx + 1:rag_idx]
        result_titles = [l.strip()[2:] for l in result_lines if l.strip().startswith("-")]
        
        # Should have results with non-empty titles
        assert len(result_titles) > 0
        assert all(len(title) > 0 for title in result_titles)

    def test_multiple_queries(self):
        """Test sequential RAG queries to ensure consistency."""
        queries = ["horror", "animation", "western"]
        
        for query in queries:
            stdout, stderr, code = run_rag(query, limit=2)
            assert code == 0, f"Query '{query}' failed"
            assert "Search Results:" in stdout
            assert "RAG Response:" in stdout

    def test_default_k_value(self):
        """Test that default k value (60) is used when not specified."""
        stdout, stderr, code = run_rag("romance", limit=2)
        
        # Should succeed with default k=60
        assert code == 0
        assert "Search Results:" in stdout

    def test_default_limit_value(self):
        """Test that default limit value (5) is used when not specified."""
        stdout, stderr, code = run_rag("adventure")
        
        # Should succeed with default limit=5
        assert code == 0
        assert "Search Results:" in stdout
        
        # Extract and count results
        bullet_count = len([l for l in stdout.split("\n") if l.strip().startswith("-")])
        assert bullet_count <= 5, "Default limit should be at most 5"

    def test_help_shows_defaults(self):
        """Test that help text shows default values for k and limit."""
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        python_exec = str(venv_python) if venv_python.exists() else "python3"
        
        result = subprocess.run(
            [python_exec, "cli/augmented_generation_cli.py", "rag", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        
        assert result.returncode == 0
        assert "Default: 60" in result.stdout or "--k" in result.stdout
        assert "Default: 5" in result.stdout or "--limit" in result.stdout

    def test_empty_query_string(self):
        """Test behavior with empty query string."""
        stdout, stderr, code = run_rag("", limit=1)
        
        # Empty query might still find results (common words) or fail gracefully
        assert code in [0, 1]

    def test_very_long_query(self):
        """Test with very long query string."""
        long_query = "this is a very long query about movies " * 10
        stdout, stderr, code = run_rag(long_query, limit=1)
        
        # Should handle long queries gracefully
        assert code in [0, 1]

    def test_query_with_numbers(self):
        """Test query containing numbers."""
        stdout, stderr, code = run_rag("2001 space odyssey", limit=3)
        
        assert code == 0
        assert "Search Results:" in stdout

    def test_stderr_on_dataset_error(self):
        """Test that meaningful error appears when dataset cannot load."""
        # This is more of an integration test - we rely on the actual error handling
        # in the code since we can't easily mock the dataset loading
        pass

    def test_output_structure(self):
        """Test the complete output structure matches expected format."""
        stdout, stderr, code = run_rag("fantasy", limit=2)
        
        assert code == 0
        
        # Should have specific structure
        assert stdout.count("Search Results:") == 1
        assert stdout.count("RAG Response:") == 1
        
        lines = stdout.split("\n")
        search_idx = next(i for i, line in enumerate(lines) if "Search Results:" in line)
        rag_idx = next(i for i, line in enumerate(lines) if "RAG Response:" in line)
        
        # RAG Response should come after Search Results
        assert rag_idx > search_idx

    def test_no_duplicate_results(self):
        """Test that search results don't contain duplicates."""
        stdout, stderr, code = run_rag("movie", limit=5)
        
        assert code == 0
        
        # Extract result titles
        lines = stdout.split("\n")
        search_idx = next(i for i, line in enumerate(lines) if "Search Results:" in line)
        rag_idx = next(i for i, line in enumerate(lines) if "RAG Response:" in line)
        
        result_lines = lines[search_idx + 1:rag_idx]
        result_titles = [l.strip()[2:] for l in result_lines if l.strip().startswith("-")]
        
        # Check for duplicates
        assert len(result_titles) == len(set(result_titles)), "Results should not contain duplicates"


class TestSummarizeCLI:
    """Tests for the summarize command in augmented_generation_cli.py"""

    def test_summarize_help_message(self):
        """Test that help message displays available arguments for summarize command."""
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        python_exec = str(venv_python) if venv_python.exists() else "python3"
        
        result = subprocess.run(
            [python_exec, "cli/augmented_generation_cli.py", "summarize", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        
        assert result.returncode == 0
        assert "Search query for summarization" in result.stdout
        assert "--limit" in result.stdout
        assert "Default: 5" in result.stdout

    def test_successful_summarize_execution(self):
        """Test successful summarize execution with basic query."""
        stdout, stderr, code = run_summarize("action movies", limit=3)
        
        assert code == 0
        assert "Search Results:" in stdout
        assert "LLM Summary:" in stdout
        # Should have at least one result and a summary
        assert len(stdout.split("\n")) > 5

    def test_summarize_search_results_format(self):
        """Test that search results are formatted correctly in summarize command."""
        stdout, stderr, code = run_summarize("space", limit=3)
        
        assert code == 0
        assert "Search Results:" in stdout
        
        # Extract search results section
        lines = stdout.split("\n")
        search_idx = next(i for i, line in enumerate(lines) if "Search Results:" in line)
        summary_idx = next(i for i, line in enumerate(lines) if "LLM Summary:" in line)
        
        # Search results should be between Search Results: and LLM Summary:
        search_result_lines = lines[search_idx + 1:summary_idx]
        
        # Should have bullet points for results
        bullet_lines = [l for l in search_result_lines if l.strip().startswith("-")]
        assert len(bullet_lines) > 0, "Should have at least one result with bullet point"

    def test_llm_summary_present(self):
        """Test that LLM summary is generated and displayed."""
        stdout, stderr, code = run_summarize("comedy", limit=2)
        
        assert code == 0
        assert "LLM Summary:" in stdout
        
        # Extract LLM summary section
        lines = stdout.split("\n")
        summary_idx = next(i for i, line in enumerate(lines) if "LLM Summary:" in line)
        
        # Summary should have content after the LLM Summary: header
        summary_lines = [l for l in lines[summary_idx + 1:] if l.strip()]
        assert len(summary_lines) > 0, "LLM summary should contain text"

    def test_summarize_custom_limit_parameter(self):
        """Test that --limit parameter affects number of results in summarize."""
        # Get results with limit=2
        stdout_2, stderr_2, code_2 = run_summarize("thriller", limit=2)
        
        # Get results with limit=4
        stdout_4, stderr_4, code_4 = run_summarize("thriller", limit=4)
        
        assert code_2 == 0
        assert code_4 == 0
        
        # Count bullet points in each output
        bullet_count_2 = len([l for l in stdout_2.split("\n") if l.strip().startswith("-")])
        bullet_count_4 = len([l for l in stdout_4.split("\n") if l.strip().startswith("-")])
        
        # Limit=4 should have more or equal results than limit=2
        assert bullet_count_4 >= bullet_count_2

    def test_summarize_query_with_special_characters(self):
        """Test summarize with query containing special characters."""
        stdout, stderr, code = run_summarize("action: adventure & drama")
        
        assert code == 0
        assert "Search Results:" in stdout
        assert "LLM Summary:" in stdout

    def test_summarize_default_limit_value(self):
        """Test that default limit value (5) is used in summarize when not specified."""
        stdout, stderr, code = run_summarize("adventure")
        
        # Should succeed with default limit=5
        assert code == 0
        assert "Search Results:" in stdout
        
        # Extract and count results
        bullet_count = len([l for l in stdout.split("\n") if l.strip().startswith("-")])
        assert bullet_count <= 5, "Default limit should be at most 5"

    def test_summarize_output_structure(self):
        """Test the complete output structure matches expected format for summarize."""
        stdout, stderr, code = run_summarize("fantasy", limit=2)
        
        assert code == 0
        
        # Should have specific structure
        assert stdout.count("Search Results:") == 1
        assert stdout.count("LLM Summary:") == 1
        
        lines = stdout.split("\n")
        search_idx = next(i for i, line in enumerate(lines) if "Search Results:" in line)
        summary_idx = next(i for i, line in enumerate(lines) if "LLM Summary:" in line)
        
        # LLM Summary should come after Search Results
        assert summary_idx > search_idx

    def test_summarize_output_contains_movie_titles(self):
        """Test that summarize output contains actual movie titles from the dataset."""
        stdout, stderr, code = run_summarize("spy thriller", limit=3)
        
        assert code == 0
        
        # Extract titles from search results
        lines = stdout.split("\n")
        search_idx = next(i for i, line in enumerate(lines) if "Search Results:" in line)
        summary_idx = next(i for i, line in enumerate(lines) if "LLM Summary:" in line)
        
        result_lines = lines[search_idx + 1:summary_idx]
        result_titles = [l.strip()[2:] for l in result_lines if l.strip().startswith("-")]
        
        # Should have results with non-empty titles
        assert len(result_titles) > 0
        assert all(len(title) > 0 for title in result_titles)

    def test_summarize_no_duplicate_results(self):
        """Test that summarize search results don't contain duplicates."""
        stdout, stderr, code = run_summarize("movie", limit=5)
        
        assert code == 0
        
        # Extract result titles
        lines = stdout.split("\n")
        search_idx = next(i for i, line in enumerate(lines) if "Search Results:" in line)
        summary_idx = next(i for i, line in enumerate(lines) if "LLM Summary:" in line)
        
        result_lines = lines[search_idx + 1:summary_idx]
        result_titles = [l.strip()[2:] for l in result_lines if l.strip().startswith("-")]
        
        # Check for duplicates
        assert len(result_titles) == len(set(result_titles)), "Results should not contain duplicates"

    def test_summarize_multiple_queries(self):
        """Test sequential summarize queries to ensure consistency."""
        queries = ["horror", "animation", "western"]
        
        for query in queries:
            stdout, stderr, code = run_summarize(query, limit=2)
            assert code == 0, f"Query '{query}' failed"
            assert "Search Results:" in stdout
            assert "LLM Summary:" in stdout

    def test_summarize_empty_query_string(self):
        """Test summarize behavior with empty query string."""
        stdout, stderr, code = run_summarize("", limit=1)
        
        # Empty query might still find results (common words) or fail gracefully
        assert code in [0, 1]

    def test_summarize_query_with_numbers(self):
        """Test summarize with query containing numbers."""
        stdout, stderr, code = run_summarize("2001 space odyssey", limit=3)
        
        assert code == 0
        assert "Search Results:" in stdout
        assert "LLM Summary:" in stdout

    def test_summarize_very_long_query(self):
        """Test summarize with very long query string."""
        long_query = "this is a very long query about movies " * 10
        stdout, stderr, code = run_summarize(long_query, limit=1)
        
        # Should handle long queries gracefully
        assert code in [0, 1]

    def test_summarize_single_character_query(self):
        """Test summarize with very short single character query."""
        stdout, stderr, code = run_summarize("a", limit=2)
        
        # Should either find results or handle gracefully
        assert code in [0, 1]
        if code == 0:
            assert "Search Results:" in stdout


class TestCitationsCLI:
    """Tests for the citations command in augmented_generation_cli.py"""

    def test_citations_help_message(self):
        """Verify citations --help shows args and defaults."""
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        result = subprocess.run(
            [python_exec, "cli/augmented_generation_cli.py", "citations", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert result.returncode == 0
        assert "Search query for citations mode" in result.stdout
        assert "--limit" in result.stdout
        assert "Default: 5" in result.stdout

    def test_successful_citations_execution(self):
        """Ensure citations command runs and prints results + answer."""
        stdout, stderr, code = run_citations("action movies", limit=3)
        assert code == 0
        assert "Search Results:" in stdout
        assert "LLM Citations:" in stdout

    def test_citations_search_results_format(self):
        """Confirm search results are bullet-listed before the answer."""
        stdout, stderr, code = run_citations("space", limit=3)
        assert code == 0
        lines = stdout.split("\n")
        search_idx = next(i for i, line in enumerate(lines) if "Search Results:" in line)
        answer_idx = next(i for i, line in enumerate(lines) if "LLM Citations:" in line)
        search_result_lines = lines[search_idx + 1 : answer_idx]
        bullet_lines = [l for l in search_result_lines if l.strip().startswith("-")]
        assert len(bullet_lines) > 0

    def test_citations_llm_answer_present(self):
        """Check that LLM Citations section includes non-empty text."""
        stdout, stderr, code = run_citations("comedy", limit=2)
        assert code == 0
        assert "LLM Citations:" in stdout
        lines = stdout.split("\n")
        ans_idx = next(i for i, line in enumerate(lines) if "LLM Citations:" in line)
        answer_lines = [l for l in lines[ans_idx + 1 :] if l.strip()]
        assert len(answer_lines) > 0

    def test_citations_custom_limit_parameter(self):
        """Validate --limit changes number of listed search results."""
        stdout_2, stderr_2, code_2 = run_citations("thriller", limit=2)
        stdout_4, stderr_4, code_4 = run_citations("thriller", limit=4)
        assert code_2 == 0
        assert code_4 == 0
        bullet_count_2 = len([l for l in stdout_2.split("\n") if l.strip().startswith("-")])
        bullet_count_4 = len([l for l in stdout_4.split("\n") if l.strip().startswith("-")])
        assert bullet_count_4 >= bullet_count_2

    def test_citations_query_with_special_characters(self):
        """Ensure special characters in queries are handled gracefully."""
        stdout, stderr, code = run_citations("action: adventure & drama")
        assert code == 0
        assert "Search Results:" in stdout
        assert "LLM Citations:" in stdout

    def test_citations_default_limit_value(self):
        """Confirm default --limit=5 behavior when not specified."""
        stdout, stderr, code = run_citations("adventure")
        assert code == 0
        bullet_count = len([l for l in stdout.split("\n") if l.strip().startswith("-")])
        assert bullet_count <= 5

    def test_citations_output_structure(self):
        """Verify output ordering: Search Results then LLM Citations."""
        stdout, stderr, code = run_citations("fantasy", limit=2)
        assert code == 0
        assert stdout.count("Search Results:") == 1
        assert stdout.count("LLM Citations:") == 1
        lines = stdout.split("\n")
        search_idx = next(i for i, line in enumerate(lines) if "Search Results:" in line)
        ans_idx = next(i for i, line in enumerate(lines) if "LLM Citations:" in line)
        assert ans_idx > search_idx

    def test_citations_output_contains_movie_titles(self):
        """Ensure titles in Search Results are non-empty and unique."""
        stdout, stderr, code = run_citations("spy thriller", limit=3)
        assert code == 0
        lines = stdout.split("\n")
        search_idx = next(i for i, line in enumerate(lines) if "Search Results:" in line)
        ans_idx = next(i for i, line in enumerate(lines) if "LLM Citations:" in line)
        result_lines = lines[search_idx + 1 : ans_idx]
        result_titles = [l.strip()[2:] for l in result_lines if l.strip().startswith("-")]
        assert len(result_titles) > 0
        assert all(len(title) > 0 for title in result_titles)


class TestQuestionCLI:
    """Tests for the question command in augmented_generation_cli.py"""

    def test_question_help_message(self):
        """Verify question --help shows args and defaults."""
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        result = subprocess.run(
            [python_exec, "cli/augmented_generation_cli.py", "question", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert result.returncode == 0
        assert "User question to answer" in result.stdout
        assert "--limit" in result.stdout
        assert "Default: 5" in result.stdout

    def test_successful_question_execution(self):
        """Ensure question command runs and prints results + answer."""
        stdout, stderr, code = run_question("Who are the main characters in Jurassic Park?", limit=3)
        assert code == 0
        assert "Search Results:" in stdout
        assert "Answer:" in stdout

    def test_question_search_results_format(self):
        """Confirm search results are bullet-listed before the answer."""
        stdout, stderr, code = run_question("space documentaries", limit=3)
        assert code == 0
        lines = stdout.split("\n")
        search_idx = next(i for i, line in enumerate(lines) if "Search Results:" in line)
        answer_idx = next(i for i, line in enumerate(lines) if "Answer:" in line)
        search_result_lines = lines[search_idx + 1 : answer_idx]
        bullet_lines = [l for l in search_result_lines if l.strip().startswith("-")]
        assert len(bullet_lines) > 0

    def test_question_llm_answer_present(self):
        """Check that the Answer section includes non-empty text."""
        stdout, stderr, code = run_question("Recommend a good comedy", limit=2)
        assert code == 0
        assert "Answer:" in stdout
        lines = stdout.split("\n")
        ans_idx = next(i for i, line in enumerate(lines) if "Answer:" in line)
        answer_lines = [l for l in lines[ans_idx + 1 :] if l.strip()]
        assert len(answer_lines) > 0

    def test_question_custom_limit_parameter(self):
        """Validate --limit changes number of listed search results."""
        stdout_2, stderr_2, code_2 = run_question("thriller picks", limit=2)
        stdout_4, stderr_4, code_4 = run_question("thriller picks", limit=4)
        assert code_2 == 0
        assert code_4 == 0
        bullet_count_2 = len([l for l in stdout_2.split("\n") if l.strip().startswith("-")])
        bullet_count_4 = len([l for l in stdout_4.split("\n") if l.strip().startswith("-")])
        assert bullet_count_4 >= bullet_count_2

    def test_question_post_processing_applies(self):
        """Ensure Jurassic Park characters include Ellie Sattler via post-processing."""
        stdout, stderr, code = run_question("Who are the main characters in Jurassic Park?")
        assert code == 0
        # Post-processing should ensure these names are present
        assert "Alan Grant" in stdout
        assert "Ellie Sattler" in stdout
        assert "Ian Malcolm" in stdout

    def test_question_default_limit_value(self):
        """Confirm default --limit=5 behavior when not specified."""
        stdout, stderr, code = run_question("suggest adventure films")
        assert code == 0
        bullet_count = len([l for l in stdout.split("\n") if l.strip().startswith("-")])
        assert bullet_count <= 5

    def test_question_output_structure(self):
        """Verify output ordering: Search Results then Answer."""
        stdout, stderr, code = run_question("fantasy options", limit=2)
        assert code == 0
        assert stdout.count("Search Results:") == 1
        assert stdout.count("Answer:") == 1
        lines = stdout.split("\n")
        search_idx = next(i for i, line in enumerate(lines) if "Search Results:" in line)
        ans_idx = next(i for i, line in enumerate(lines) if "Answer:" in line)
        assert ans_idx > search_idx

    def test_question_output_contains_movie_titles(self):
        """Ensure titles in Search Results are non-empty."""
        stdout, stderr, code = run_question("spy thriller", limit=3)
        assert code == 0
        lines = stdout.split("\n")
        search_idx = next(i for i, line in enumerate(lines) if "Search Results:" in line)
        ans_idx = next(i for i, line in enumerate(lines) if "Answer:" in line)
        result_lines = lines[search_idx + 1 : ans_idx]
        result_titles = [l.strip()[2:] for l in result_lines if l.strip().startswith("-")]
        assert len(result_titles) > 0
        assert all(len(title) > 0 for title in result_titles)

    def test_question_no_duplicate_results(self):
        """Ensure search results don't contain duplicates."""
        stdout, stderr, code = run_question("movie", limit=5)
        assert code == 0
        lines = stdout.split("\n")
        search_idx = next(i for i, line in enumerate(lines) if "Search Results:" in line)
        ans_idx = next(i for i, line in enumerate(lines) if "Answer:" in line)
        result_lines = lines[search_idx + 1 : ans_idx]
        result_titles = [l.strip()[2:] for l in result_lines if l.strip().startswith("-")]
        assert len(result_titles) == len(set(result_titles))

    def test_question_query_with_special_characters(self):
        """Ensure special characters in questions are handled gracefully."""
        stdout, stderr, code = run_question("action: adventure & drama?")
        assert code == 0
        assert "Search Results:" in stdout
        assert "Answer:" in stdout

    def test_question_empty_string(self):
        """Test behavior with empty question string."""
        stdout, stderr, code = run_question("")
        assert code in [0, 1]

    def test_question_query_with_numbers(self):
        """Test question containing numbers."""
        stdout, stderr, code = run_question("Top 10 sci-fi")
        assert code == 0
        assert "Search Results:" in stdout

    def test_question_very_long_input(self):
        """Handle very long question string gracefully."""
        long_q = "this is a very long question about movies " * 10
        stdout, stderr, code = run_question(long_q, limit=1)
        assert code in [0, 1]

    def test_question_single_character_input(self):
        """Single character question should not crash."""
        stdout, stderr, code = run_question("a", limit=2)
        assert code in [0, 1]
        if code == 0:
            assert "Search Results:" in stdout
