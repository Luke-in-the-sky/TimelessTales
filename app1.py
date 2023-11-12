import torch
from dataclasses import dataclass
import openai
from transformers import pipeline


@dataclass
class Platforms:
    OPENAI: str = "openai"
    HUGGINGFACE: str = "huggingface"


# Example configuration
config = {
    "summarizer": {
        "model_name": "pszemraj/led-base-book-summary",
        "max_tokens_per_chunk": 4096,
        "max_chunk_summary_length": 128,
        "chapter_separator": "\n" * 5,
    },
    # "simplifier": {
    #     "source": Platforms.OPENAI,
    #     "simplification_prompt": "Simplify this text for a child:",
    #     "openai_api_key": "your-openai-api-key",
    #     "openai_model": "text-davinci-002",
    #     "huggingface_model_name": "gpt2"
    # }
}


# Summarizer Model Interface
class SummarizerModelInterface:
    def __init__(self, model_name):
        self.pipeline = pipeline(
            "summarization",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
        )

    def tokenize(self, text):
        # Note: don't return the `pt` tensor, just the list of tokens
        return self.pipeline.tokenizer.encode(text, add_special_tokens=True)

    def decode(self, tokens):
        return self.pipeline.tokenizer.decode(tokens)

    def generate_output(self, input_text, max_length=130, min_length=30):
        return self.pipeline(
            input_text, max_length=max_length, min_length=min_length, truncation=True
        )[0]["summary_text"]


# Story Summarizer
class StorySummarizer:
    def __init__(
        self,
        model_name,
        max_tokens_per_chunk,
        chapter_separator=None,
        max_chunk_summary_length=128,
    ):
        self.model_interface = SummarizerModelInterface(model_name)
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.max_chunk_summary_length = max_chunk_summary_length
        self.chapter_separator = chapter_separator

    def chunk_tokens(self, tokens):
        """
        TODO: we want a better tokenization strategy here
        where the length of the chunks is consistent
        and there is option for a bit of overlap between tokens
        """
        for i in range(0, len(tokens), self.max_tokens_per_chunk):
            yield tokens[i : i + self.max_tokens_per_chunk]

    def summarize_chapter(self, chapter_text):
        print("tokenizing chapter")
        chapter_tokens = self.model_interface.tokenize(chapter_text)
        print(f" ..done: {len(chapter_tokens)=}")
        if len(chapter_tokens) > self.max_tokens_per_chunk:
            chunk_summaries = []
            for chunk in self.chunk_tokens(chapter_tokens):
                chunk_text = self.model_interface.decode(chunk)
                print(f"{len(chunk)=}, {len(chunk_text)=}")
                chunk_summary = self.model_interface.generate_output(
                    chunk_text,
                    max_length=min(len(chunk), self.max_chunk_summary_length),
                )
                chunk_summaries.append(chunk_summary)
            return " ".join(chunk_summaries)
        else:
            return self.model_interface.generate_output(
                chapter_text, max_length=self.max_chunk_summary_length
            )

    def summarize(self, text):
        if self.chapter_separator:
            chapters = text.split(self.chapter_separator)
            chapter_summaries = [
                self.summarize_chapter(chapter) for chapter in chapters
            ]
            return " ".join(chapter_summaries)
        else:
            print("-- no chapters")
            return self.summarize_chapter(text)


# Text Simplifier
class TextSimplifier:
    def __init__(
        self,
        source,
        simplification_prompt="Simplify this text for a child:",
        openai_api_key=None,
        openai_model="text-davinci-002",
        huggingface_model_name="gpt2",
    ):
        self.source = source
        self.simplification_prompt = simplification_prompt

        if self.source == Platforms.OPENAI:
            openai.api_key = openai_api_key
            self.model = openai_model
        elif self.source == Platforms.HUGGINGFACE:
            self.model = pipeline(
                "text-generation",
                model=huggingface_model_name,
                device=0 if torch.cuda.is_available() else -1,
            )

    def simplify(self, text):
        prompt = self.simplification_prompt + text
        if self.source == Platforms.OPENAI:
            response = openai.Completion.create(
                engine=self.model, prompt=prompt, max_tokens=100
            )
            return response.choices[0].text.strip()
        elif self.source == Platforms.HUGGINGFACE:
            return self.model(prompt)[0]["generated_text"]


# Main Processing Function
def process_story(story_text, config):
    summarizer = StorySummarizer(**config["summarizer"])
    print("passing text")
    processed = summarizer.summarize(story_text)

    if "simplifier" in config:
        simplifier = TextSimplifier(**config["simplifier"])
        processed = simplifier.simplify(processed)

    return processed
