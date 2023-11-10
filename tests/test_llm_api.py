import pytest
from unittest.mock import patch, Mock
from llm_api import LargeLanguageModelAPI

@pytest.fixture
def huggingface_api():
    with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
         patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
        api = LargeLanguageModelAPI('huggingface', 'model_id')
        yield api, mock_tokenizer, mock_model

def test_set_model_id_huggingface(huggingface_api):
    api, mock_tokenizer, mock_model = huggingface_api
    mock_tokenizer.assert_called_with('model_id')
    mock_model.assert_called_with('model_id')
    assert api.tokenizer == mock_tokenizer.return_value
    assert api.model == mock_model.return_value

def test_infer_openai():
    # TODO
    pass

def test_infer_invalid_model_type():
    api = LargeLanguageModelAPI('invalid_type', 'model_id')
    with pytest.raises(ValueError):
        api.infer('input_text')
