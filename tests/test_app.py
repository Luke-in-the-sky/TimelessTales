import pytest
from unittest.mock import patch
from app import LiteratureAdaptationApp, main


@pytest.fixture
def app(mocker):
    # Create an instance of LiteratureAdaptationApp for testing
    return LiteratureAdaptationApp(
        gutenberg_query="Sample Query",
        target_age=10,
        api_key="sample_api_key",
        model_type="gpt-3.5-turbo",
        model_id="sample_model_id",
    )


def test_output_result_console(app, capsys):
    # Test output_result method for console output
    app.output_result("Sample Text", filepath=None)
    captured = capsys.readouterr()
    assert captured.out == "Sample Text\n"


def test_output_result_file(app, tmp_path):
    # Test output_result method for file output
    test_file_path = tmp_path / "test_output.txt"
    app.output_result("Sample Text", filepath=str(test_file_path))
    assert test_file_path.read_text() == "Sample Text"


def test_run(app, mocker):
    m_GutApi = mocker.patch("app.ProjectGutenbergAPI")
    m_GutApi.return_value.download_most_popular_book_from_query.return_value = (
        "Sample Book Content"
    )
    mock_summarizer = mocker.patch("app.TextSummarizer")
    mock_summarizer.return_value.summarize.return_value = "Sample Summary"

    mock_llm_compose = mocker.patch("app.LargeLanguageModelAPI.compose_prompt")
    mock_llm_compose.return_value = "Mock composed prompt"

    app.run()

if __name__ == "__main__":
    pytest.main()
