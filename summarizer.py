from llm_api import LargeLanguageModelAPI

prompt__summarize_narrative = """
following is a narrative. I want to tell this story to a 5 year old, but the text is too long.
Instead, we want to make a new narrative, where the story is the same but the language is appropriate
for a 5 year old and the story is divided into five to eight chapters, where each chapter is just
long enough for a bedtime story

"""


class StringSplitter:
    """
    A class that provides a method to split a string into chunks based on the longest sequence
    of newline characters ('\\n'). The chunks are created such that each is smaller than or
    equal to a specified maximum chunk length. If a chunk exceeds the maximum length and cannot
    be split on a newline sequence, it is split at the maximum length.
    Usage:
        >>> splitter = StringSplitter(max_chunk_length=45)
        >>> input_string = "Your\\n\\n\\nvery long string\\n\\nwith newline\\ncharacters"
        >>> chunks = splitter.split_string(input_string)
        >>> print(chunks)
        ['Your\\n\\n\\n', 'very long string\\n\\nwith newline\\ncharacters']
    """

    def __init__(self, max_chunk_length):
        if not isinstance(max_chunk_length, int) or max_chunk_length <= 0:
            raise ValueError("max_chunk_length must be a positive integer.")
        self.max_chunk_length = max_chunk_length

    def split_string(self, input_string):
        """
        Split a string into chunks based on the longest sequence of '\n' characters
        while ensuring each chunk is smaller than max_chunk_length.
        """
        if not isinstance(input_string, str):
            raise ValueError("input_string must be a string.")

        chunks = []
        current_string = input_string
        while len(current_string) > self.max_chunk_length:
            split_index = self._find_split_index(current_string)
            # Split the string at the index and add the left part to chunks
            chunks.append(current_string[:split_index])
            # Update current_string to the remaining part
            current_string = current_string[split_index:]
        # Add the last part if there's any
        if current_string:
            chunks.append(current_string)

        return chunks

    def _find_split_index(self, string):
        # Initialize the longest sequence count and index
        longest_sequence_count = 0
        split_index = max_length = self.max_chunk_length

        # Iterate over the string up to the max_length to find the longest sequence of '\n'
        current_sequence_count = 0
        for i, char in enumerate(string[:max_length]):
            if char == "\n":
                current_sequence_count += 1
                # If the current sequence is longer than the longest found, update the longest count and index
                if current_sequence_count > longest_sequence_count:
                    longest_sequence_count = current_sequence_count
                    split_index = i + 1  # Include the last '\n' in the current sequence
            else:
                current_sequence_count = 0  # Reset count if the sequence is broken

        # If no sequence is found, default to split at max_length
        if longest_sequence_count == 0:
            return max_length

        return split_index


class TextSummarizer:
    """
    A class that encapsulates the functionality required to summarize a long piece of text using a
    large language model, either via an API or a downloaded model from Hugging Face.
    """

    def __init__(
        self,
        language_model: LargeLanguageModelAPI,
        custom_system_prompt: str = None,
        max_length=None,
    ):
        """
        Initializes the TextSummarizer instance with a language model, a system prompt, and a maximum length.
        """
        self.llm = language_model
        self.system_prompt = (
            custom_system_prompt
            if custom_system_prompt
            else prompt__summarize_narrative
            # TODO: we might have a more standard system prompt and concat here
            # the custom one just to express style preferences
        )
        self.set_max_length(max_length or language_model.max_context_length)

    def set_max_length(self, max_length):
        """
        Sets the maximum length of text to summarize and initializes a StringSplitter instance.
        If max_length < length of the text to be summarize, we will chunk things down into sizes of max_length
        """
        self.max_length = max_length
        self.string_splitter = StringSplitter(max_length)

    def summarize(
        self,
        text: str,
    ) -> str:
        """
        Summarizes the provided text using the language model.

        If the concatenated system prompt and text exceeds the maximum length, the text is split into chunks
        using the `StringSplitter.split_text` method. Each chunk is then summarized individually, and the results
        are concatenated to form the final summary.
        """
        # Concatenate the prompt with the text
        full_text = self.llm.compose_prompt(
            message=text, system_message=self.system_prompt
        )

        # Check if the length of the text is within the maximum length of chuncks we want to summarize
        if len(text) <= self.max_length:
            return self.llm.infer(
                full_text
            )  # infer on `full_text`, but have the if on `text`
        else:
            # If the text exceeds the limit, split it into manageable chunks
            chunks = self.string_splitter.split_string(text)
            full_chunks = [
                self.llm.compose_prompt(
                    message=chunk, system_message=self.system_prompt
                )
                for chunk in chunks
            ]

            # Summarize each chunk individually, then combine them into the final summary
            summaries = [self.llm.infer(full) for full in full_chunks]
            return " ".join(summaries)

    # TODO: we might want to add a method to expand summaries abstractively a little bit, so that
    # `summarize` gives the outline, but the new method expands on individual chapters a bit more


# testing

t =  TextSummarizer(
        language_model=LargeLanguageModelAPI(
            model_type="hf_seq2seq",
            model_id="pszemraj/led-base-book-summary",
            ),
        custom_system_prompt="summarize this:",
        max_length=4000,
    )
t.summarize('''
            Basic Usage
I recommend using encoder_no_repeat_ngram_size=3 when calling the pipeline object, as it enhances the summary quality by encouraging the use of new vocabulary and crafting an abstractive summary.

Create the pipeline object:

import torch
from transformers import pipeline

hf_name = "pszemraj/led-base-book-summary"

summarizer = pipeline(
    "summarization",
    hf_name,
    device=0 if torch.cuda.is_available() else -1,
)

Feed the text into the pipeline object:

wall_of_text = "your words here"

result = summarizer(
    wall_of_text,
    min_length=8,
    max_length=256,
    no_repeat_ngram_size=3,
    encoder_no_repeat_ngram_size=3,
    repetition_penalty=3.5,
    num_beams=4,
    do_sample=False,
    early_stopping=True,
)
print(result[0]["generated_text"])

Simplified Usage with TextSum
To streamline the process of using this and other models, I've developed a Python package utility named textsum. This package offers simple interfaces for applying summarization models to text documents of arbitrary length.

Install TextSum:

pip install textsum

Then use it in Python with this model:
''')
