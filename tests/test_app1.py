import pytest
from unittest.mock import patch
from your_module import StorySummarizer

# Replace "your_module" with the actual module name where your code is located


@pytest.mark.parametrize(
    "max_tokens_per_chunk, text_length, expected_chunks",
    [
        (100, 500, 5),  # Test case: 500 tokens, 100 tokens per chunk
        (50, 500, 10),  # Test case: 500 tokens, 50 tokens per chunk
        (200, 600, 3),  # Test case: 600 tokens, 200 tokens per chunk
    ],
)
def test_chunking_in_summarization(max_tokens_per_chunk, text_length, expected_chunks):
    config = {
        "summarizer": {
            "model_name": "mock_model_name",
            "max_tokens_per_chunk": max_tokens_per_chunk,
            "chapter_separator": None,
        }
    }

    # Create an instance of StorySummarizer with the provided configuration
    with patch(
        "your_module.SummarizerModelInterface.pipeline",
        side_effect=MockSummarizationPipeline,
    ):
        summarizer = StorySummarizer(**config["summarizer"])

        # Generate a sample text with the specified length
        sample_text = " ".join(["word"] * text_length)

        # Call the summarize_chapter method
        summarized_text = summarizer.summarize_chapter(sample_text)

        # Split the summarized text into chunks using whitespace as the separator
        chunked_summaries = summarized_text.split()

        # Check if the number of chunks in the summarized text matches the expected number
        assert len(chunked_summaries) == expected_chunks
