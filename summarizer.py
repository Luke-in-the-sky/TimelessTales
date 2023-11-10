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

    Attributes:
        max_chunk_length (int): The maximum allowed length of each chunk.

    This class can be used to process long strings where it is important to preserve newline
    grouping and maintain readability, such as when formatting text for display in a limited
    space or ensuring consistent data chunks for processing.

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
          if char == '\n':
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

    Attributes:
        language_model (object): An instance of the large language model used for summarization.
        system_prompt (str): A hardcoded system prompt to be used for initiating the summary.
        max_length (int): The maximum length of text that the model can handle in one go.

    Methods:
        summarize(text: str) -> str: Returns a summary of the provided text.
    """

    def __init__(self, language_model: LargeLanguageModelAPI, custom_system_prompt: str = None, max_length=None):
        """
        Initializes the TextSummarizer instance with a language model, a system prompt, and a maximum length.

        Parameters:
            language_model (object): An instance of the large language model used for summarization.
            system_prompt (str): A hardcoded system prompt to be concatenated with the text to be summarized.
            if None, a default system prompt will be used.
            max_length (int): the size of the chunck of text we will create mini-summaries for.
                smaller sizes mean we will end up with a more detailed summary, larger sizes mean we will end up
                with a more high-level summary. Defaults to the largest context that the LLM model can handle.
        """
        self.llm = language_model
        self.system_prompt = system_prompt if system_prompt else prompt__summarize_narrative
        self.set_max_length(max_length or language_model.max_context_length)

    def set_max_length(self, max_length):
        """
        Sets the maximum length of text that the model can handle and initializes a StringSplitter instance.

        Parameters:
            max_length (int): The maximum length of text that the model can handle.
        """
        self.max_length = max_length
        self.string_splitter = StringSplitter(max_length)


    def summarize(self, text: str) -> str:
        """
        Summarizes the provided text using the language model.

        If the concatenated system prompt and text exceeds the maximum length, the text is split into chunks
        using the `StringSplitter.split_text` method. Each chunk is then summarized individually, and the results
        are concatenated to form the final summary.

        Parameters:
            text (str): The long text to be summarized.

        Returns:
            str: A summary of the input text.
        """
        try:
            # Concatenate the prompt with the text
            full_text = self.system_prompt + "\n" + text

            # Check if the length of the text is within the maximum length limit
            if len(full_text) <= self.max_length:
                # If the text is within the limit, directly infer the summary
                return self.llm.infer(full_text)
            else:
                # If the text exceeds the limit, split it into manageable chunks
                chunks = self.string_splitter.split_string(full_text)

                # Summarize each chunk individually
                summaries = [self.llm.infer(chunk) for chunk in chunks]

                # Combine the individual summaries into the final summary
                final_summary = ' '.join(summaries)
                return final_summary

    # TODO: we might want to add a method to expand summaries abstractively a little bit, so that
    # `summarize` gives the outline, but the new method expands on individual chapters a bit more
