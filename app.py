import argparse
from typing import Optional
from gutenberg_api import ProjectGutenbergAPI
from llm_api import LargeLanguageModelAPI
from summarizer import TextSummarizer


class LiteratureAdaptationApp:
    """
    The main application class for creating kids adaptations of literature classics.
    """

    def __init__(
        self,
        gutenberg_query: str,
        api_key: str,
        model_type: str,
        model_id: str,
        target_age: Optional[int] = None,
        output_file_path: Optional[str] = None,
    ):
        self.gutenberg_query = gutenberg_query
        self.target_age = target_age
        self.output_file_path = output_file_path

        # ML params
        self.api_key = api_key
        self.llm = LargeLanguageModelAPI(model_type, model_id, api_key=None)

    def run(self):
        """
        The main method to run the app's workflow.
        """
        api = ProjectGutenbergAPI()
        book_content = api.download_most_popular_book_from_query(self.gutenberg_query)

        # TODO: add functionality to specify custom preference for style
        summarizer = TextSummarizer(language_model=self.llm, custom_system_prompt=None)
        summary = summarizer.summarize(book_content)

        # TODO
        # formatter = MarkdownFormatter()
        # markdown_text = formatter.format_to_markdown(summary)

        self.output_result(summary, self.output_file_path)

    def output_result(self, text: str, filepath: Optional[str] = None):
        """
        Outputs the result to the terminal or saves it to a file based on user input.
        """
        if filepath is not None:
            with open(filepath, "w") as f:
                f.write(text)
        else:
            print(text)


def parse_args_from_cli():
    parser = argparse.ArgumentParser(
        description="Create kids adaptations of literature classics."
    )
    parser.add_argument(
        "gutenberg_query", type=str, help="The query to search for in Project Gutenberg"
    )
    parser.add_argument(
        "--target_age",
        type=int,
        help="The target age of the kids",
        default=None,
        required=False,
    )
    parser.add_argument("--api_key", type=str, help="OpenAI API key", required=False)
    parser.add_argument(
        "--model_type", type=str, help="hugging_face or open_ai", required=False
    )
    parser.add_argument(
        "--model_id", type=str, help="name of the model", required=False
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        help="filepath for saving the output of the summarization",
        required=False,
    )

    print(parser.parse_args())
    return parser.parse_args()


def main():
    args = parse_args_from_cli()

    app = LiteratureAdaptationApp(**vars(args))
    app.run()


if __name__ == "__main__":
    main()
