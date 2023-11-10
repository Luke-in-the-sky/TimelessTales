import pytest
from unittest.mock import patch, Mock
from gutenberg_api import ProjectGutenbergAPI, ProjectGutenbergAPIError

@pytest.fixture
def api():
    return ProjectGutenbergAPI()

@patch('requests.get')
def test_search_books(mock_get, api):
    mock_response = Mock()
    mock_response.content = b'<html><li class="booklink"><a class="link" href="/ebooks/12345">Book Link</a><span class="title">Book Title</span><span class="subtitle">Author</span></li></html>'
    mock_get.return_value = mock_response

    result = api.search_books("test", top_n=1)
    assert result == [{'title': 'Book Title', 'author': 'Author', 'link': 'https://www.gutenberg.org/ebooks/12345'}]

@patch('requests.get')
def test_download_book_content_text_only(mock_get, api):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "Book Content"
    mock_get.return_value = mock_response

    result = api.download_book_content("https://www.gutenberg.org/ebooks/12345", text_only=True)
    assert result == "Book Content"

@patch('requests.get')
def test_download_book_content_original(mock_get, api):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "Book HTML Content"
    mock_get.return_value = mock_response

    result = api.download_book_content("https://www.gutenberg.org/ebooks/12345", text_only=False)
    assert result == "Book HTML Content"

@patch.object(ProjectGutenbergAPI, 'search_books')
@patch.object(ProjectGutenbergAPI, 'download_book_content')
def test_download_most_popular_book_from_query(mock_download_book_content, mock_search_books, api):
    mock_search_books.return_value = [{'title': 'Book Title', 'author': 'Author', 'link': 'https://www.gutenberg.org/ebooks/12345'}]
    mock_download_book_content.return_value = "Book Content"

    result = api.download_most_popular_book_from_query("test", text_only=True)
    assert result == "Book Content"
