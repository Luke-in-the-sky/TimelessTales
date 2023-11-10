import pytest
from unittest.mock import patch
from main import LiteratureAdaptationApp  # Replace 'your_module' with the actual module name

@pytest.fixture
def app(mocker):
    # Create an instance of LiteratureAdaptationApp for testing
    return LiteratureAdaptationApp(
        gutenberg_query="Sample Query",
        target_age=10,
        api_key="sample_api_key",
        model_type="gpt-3.5-turbo",
        model_id="sample_model_id"
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
    # Mock the ProjectGutenbergAPI and LargeLanguageModelAPI calls
    mock_project_gutenberg_api = mocker.patch('gutenberg_api.ProjectGutenbergAPI')
    mock_project_gutenberg_api.return_value.download_most_popular_book_from_query.return_value = "Sample Book Content"

    mock_summarizer = mocker.patch('summarizer.TextSummarizer')
    mock_summarizer_instance = mock_summarizer.return_value
    mock_summarizer_instance.summarize.return_value = "Sample Summary"

    app.run()

    # Verify that the relevant methods were called with the expected arguments
    mock_project_gutenberg_api.assert_called_once()
    mock_project_gutenberg_api.return_value.download_most_popular_book_from_query.assert_called_once_with("Sample Query")
    mock_summarizer.assert_called_once_with(language_model=app.llm, custom_system_prompt=None)
    mock_summarizer_instance.summarize.assert_called_once_with("Sample Book Content")

if __name__ == "__main__":
    pytest.main()
