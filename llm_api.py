from dataclasses import dataclass


prompt_templates = {
    "pszemraj/led-base-book-summary": "{message}",
    "teknium/Mistral-Trismegistus-7B": "{system_message}\nUSER: {message}\nASSISTANT:",
}


class LargeLanguageModelAPIError(Exception):
    """Custom exception class for LargeLanguageModelAPI errors."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class LargeLanguageModelAPI:
    """
    A unified API class to interface with different large language models.

    This class can instantiate an interface for either a local Hugging Face model
    or a remote OpenAI GPT model based on the provided arguments at instantiation.
    For Hugging Face models, the model and tokenizer are instantiated only once
    when a new model_id is set. The actual inference call will utilize the specified
    model to generate the output.
    """

    def __init__(self, model_type, model_id, api_key=None, max_context_length=4000):
        """
        Initializes the API interface with the given model type and identifier.
        """
        self.model_type = model_type
        self.api_key = api_key
        self._model_id = None
        self.max_context_length = max_context_length
        self.set_model_id(model_id)

    def set_model_id(self, model_id):
        """
        Sets a new model identifier for Hugging Face models and re-instantiates the tokenizer and model.
        """
        if self.model_type == "huggingface" and self._model_id != model_id:
            self._model_id = model_id
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id)

    def compose_prompt(
        self, message: str, system_message: str = None, full_text: str = None
    ):
        if full_text is not None:
            return full_text
        try:
            full_text = prompt_templates[self._model_id].format(
                message=message, system_message=system_message
            )
        except KeyError:
            raise ValueError("Don't know the prompt format for this model")

    def infer(self, text: str = None):
        """
        Generates an inference from the chosen large language model based on the input text.
        routing between huggingface and openai
        """
        # TODO: we might want to catch situations where the text is too long for the LLM
        if self.model_type == "huggingface":
            return self._hf_local_model_inference(full_text)
        elif self.model_type == "openai":
            return self._oai_remote_model_inference(full_text)
        else:
            raise ValueError("Invalid model type specified.")

    def _hf_local_model_inference(self, text):
        """
        Generates an inference using a local Hugging Face model.

        Parameters:
            text (str): The input text to be processed by the model.

        Returns:
            str: The generated text from the model.
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(0)
        out = self.model.generate(**inputs, max_new_tokens=300)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def _oai_remote_model_inference(self, text):
        """
        Generates an inference using OpenAI's GPT model remotely.

        Parameters:
            text (str): The input text to be processed by the model.

        Returns:
            str: The generated text from the model.
        """
        if not self.api_key:
            raise ValueError("API key is required for OpenAI model inference.")

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": text,
                }
            ],
            model="gpt-3.5-turbo",
        )

        return chat_completion.choices[0].text.strip()
