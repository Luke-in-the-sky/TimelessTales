import pytest
from unittest.mock import Mock
from llm_api import LargeLanguageModelAPI
from summarizer import StringSplitter, TextSummarizer


@pytest.fixture
def mock_llm_api():
    mock_api = Mock(spec=LargeLanguageModelAPI)
    mock_api.infer = Mock(return_value="Mocked summary")
    mock_api.max_context_length = 4000  # Set the max_context_length attribute
    return mock_api


def test_string_splitter_init():
    with pytest.raises(ValueError):
        # Test if max_chunk_length is a positive integer
        StringSplitter(max_chunk_length=-1)


def test_string_splitter_split_string():
    splitter = StringSplitter(max_chunk_length=10)
    input_string = "Line1\nLine2\nLine3"
    expected_chunks = ["Line1\n", "Line2\n", "Line3"]
    chunks = splitter.split_string(input_string)
    assert chunks == expected_chunks

    splitter = StringSplitter(max_chunk_length=45)
    input_string = "Your\n\n\nvery long string\n\nwith newline\ncharacters"
    chunks = splitter.split_string(input_string)
    expected_chunks = ["Your\n\n\n", "very long string\n\nwith newline\ncharacters"]
    assert chunks == expected_chunks


def test_text_summarizer_init(mock_llm_api):
    summarizer = TextSummarizer(mock_llm_api)
    assert summarizer.max_length == mock_llm_api.max_context_length


def test_text_summarizer_set_max_length(mock_llm_api):
    summarizer = TextSummarizer(mock_llm_api)
    summarizer.set_max_length(100)
    assert summarizer.max_length == 100


def test_text_summarizer_summarize(mock_llm_api):
    summarizer = TextSummarizer(mock_llm_api)
    message = "This is a long text to be summarized."
    summary = summarizer.summarize(message)
    expected_summary = "Mocked summary"
    assert summary == expected_summary


def test_text_summarizer_multi_chunk(mock_llm_api):
    summarizer = TextSummarizer(mock_llm_api, max_length=20)
    input_text = "This is a long test input text with multiple chunks."
    summary = summarizer.summarize(input_text)
    expected_summary = " ".join(
        [
            "Mocked summary",
            "Mocked summary",
            "Mocked summary",
        ]
    )
    assert summary == expected_summary


# Run the tests
if __name__ == "__main__":
    pytest.main()
