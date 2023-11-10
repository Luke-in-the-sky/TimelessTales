import argparse
from typing import Optional
from gutenberg_api import ProjectGutenbergAPI
from llm_api import LargeLanguageModelAPI


class LiteratureAdaptationApp:
    """
    The main application class for creating kids adaptations of literature classics.
    """

    def __init__(
        self,
        gutenberg_query: str,
        target_age: Optional[int],
        api_key: str,
        model_type: str,
        model_id: str,
    ):
        self.gutenberg_query = gutenberg_query
        self.target_age = target_age

        # ML params
        self.api_key = api_key
        self.llm = LargeLanguageModelAPI(model_type, model_id, api_key=None)

    def run(self):
        """
        The main method to run the app's workflow.
        """
        api = ProjectGutenbergAPI()
        book_content = api.download_most_popular_book_from_query(gutenberg_query)

        # TODO: add functionality to specify custom preference for style
        summarizer = TextSummarizer(language_model=self.llm, custom_system_prompt=None)
        summary = summarizer.summarize(book_content)

        # TODO
        # formatter = MarkdownFormatter()
        # markdown_text = formatter.format_to_markdown(summary)

        self.output_result(summary)

    def output_result(self, text: str, filepath: Optional[str]):
        """
        Outputs the result to the terminal or saves it to a file based on user input.
        """
        if filepath is not None:
            with open(filepath, "w") as f:
                f.write(text)
        else:
            print(text)


def main():
    parser = argparse.ArgumentParser(
        description="Create kids adaptations of literature classics."
    )
    parser.add_argument("title", type=str, help="The title of the book")
    parser.add_argument("author", type=str, help="The author of the book")
    parser.add_argument(
        "--age", type=int, help="The target age of the kids", default=None
    )
    parser.add_argument("--api_key", type=str, help="OpenAI API key", required=True)
    args = parser.parse_args()

    app = LiteratureAdaptationApp(args.title, args.author, args.age, args.api_key)
    app.run()


if __name__ == "__main__":
    main()
