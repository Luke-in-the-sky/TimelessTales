from typing import List
import re
import requests
from bs4 import BeautifulSoup


class ProjectGutenbergAPIError(Exception):
    """Custom exception class for Project Gutenberg API errors."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ProjectGutenbergAPI:
    """
    A simple API client class for Project Gutenberg that allows searching for books
    and downloading their content.

    Attributes:
        base_url (str): The base URL for the Project Gutenberg search functionality.
    """

    def __init__(self):
        self.base_url = "https://www.gutenberg.org/ebooks/search/?query="

    def search_books(self, query, top_n=1) -> List[dict]:
        """
        Searches for books on Project Gutenberg based on a search query.

        Parameters:
            query (str): The search term used to query Project Gutenberg.
            top_n (int): The number of top search results to return. Defaults to 1.

        Returns:
            list: A list of dictionaries, where each dictionary contains information
                  about a book, including its title, author, and link to its Gutenberg page.
        """
        search_url = self.base_url + query
        response = requests.get(search_url)
        soup = BeautifulSoup(response.content, "html.parser")
        books = []
        for row in soup.find_all("li", class_="booklink"):
            if top_n == 0:
                break
            link = row.find("a", class_="link")
            book_link = "https://www.gutenberg.org" + link.get("href")
            book_title = row.find("span", class_="title").get_text().strip()
            author = row.find("span", class_="subtitle").get_text().strip()
            books.append({"title": book_title, "author": author, "link": book_link})
            top_n -= 1
        return books

    def download_book_content(self, book_link, text_only=True) -> str:
        """
        Downloads the content of a book from Project Gutenberg.

        Parameters:
            book_link (str): The link to the book's Gutenberg page.
            text_only (bool): If True, retrieves plain text content. Otherwise,
                              retrieves the content as originally provided.
                              Defaults to True.

        Returns:
            str: The text content of the book if successful.

        Raises:
            ProjectGutenbergAPIError: An error occurred while attempting to
                                      download the book content.
        """
        try:
            if text_only:
                book_id_match = re.search(r"gutenberg\.org/ebooks/(\d+)/?$", book_link)
                if book_id_match:
                    book_id = book_id_match.group(1)
                    content_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
                    response = requests.get(content_url)
                else:
                    raise ProjectGutenbergAPIError(
                        "Failed to extract book_id from book_link"
                    )
            else:
                response = requests.get(book_link)

            if response.status_code == 200:
                return response.text
            else:
                raise ProjectGutenbergAPIError("Failed to download book content")
        except requests.exceptions.RequestException as e:
            raise ProjectGutenbergAPIError("Failed to retrieve book content") from e

    def download_most_popular_book_from_query(self, query, text_only=True) -> str:
        """
        Downloads the most popular book from Project Gutenberg based on a search query.

        Parameters:
            query (str): The search term used to query Project Gutenberg.
            text_only (bool): If True, retrieves plain text content. Otherwise,
                              retrieves the content as originally provided.
                              Defaults to True.

        Returns:
            str: The text content of the book if successful.
        """
        search_results = self.search_books(query, top_n=1)
        if search_results:
            book_link = search_results[0]["link"]
            book_content = self.download_book_content(book_link, text_only=text_only)
            return book_content
        else:
            raise ProjectGutenbergAPIError("No search results found")
