import pytest
from unittest.mock import patch, MagicMock
from llm_api import (
    LargeLanguageModelAPI,
    SupportedLLMTypes,
    LargeLanguageModelAPIError,
    prompt_templates,
)


@pytest.fixture
def mock_transformers():
    with patch("llm_api.pipeline") as mock_pipeline, patch(
        "llm_api.AutoTokenizer.from_pretrained"
    ) as mock_tokenizer, patch(
        "llm_api.AutoModelForCausalLM.from_pretrained"
    ) as mock_model, patch.dict(
        prompt_templates,
        {"dummy_model": "{message}", "new_dummy_model": "{system_message} {message}"},
    ):
        mock_pipeline.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        yield mock_tokenizer, mock_model


# Test Initialization
@pytest.mark.parametrize(
    "model_type",
    [
        (SupportedLLMTypes.HF_SEQ2SEQ),
        (SupportedLLMTypes.HF_CAUSAL),
        (SupportedLLMTypes.OA_API),
    ],
)
def test_initialization(model_type, mock_transformers):
    api = LargeLanguageModelAPI(model_type=model_type, model_id="dummy_model")
    assert api.model_type == model_type


# Test set_model_id Method
def test_set_model_id(mock_transformers):
    mock_tokenizer, mock_model = mock_transformers
    api = LargeLanguageModelAPI(
        model_type=SupportedLLMTypes.HF_CAUSAL, model_id="dummy_model"
    )
    api.set_model_id("new_dummy_model")
    mock_tokenizer.assert_called_with("new_dummy_model")
    mock_model.assert_called_with("new_dummy_model")


# Test compose_prompt Method
def test_compose_prompt(mock_transformers):
    api = LargeLanguageModelAPI(
        model_type=SupportedLLMTypes.HF_SEQ2SEQ, model_id="dummy_model"
    )
    prompt = api.compose_prompt(message="Hello World")
    assert prompt == "Hello World"

    api = LargeLanguageModelAPI(
        model_type=SupportedLLMTypes.HF_SEQ2SEQ, model_id="new_dummy_model"
    )
    prompt = api.compose_prompt(message="Hello World", system_message="System Message")
    assert prompt == "System Message Hello World"

    with pytest.raises(ValueError):
        api = LargeLanguageModelAPI(
            model_type=SupportedLLMTypes.HF_SEQ2SEQ, model_id="unknown_model"
        )
        api.compose_prompt(message="Hello")


# Test Model Inference Methods
# TODO
