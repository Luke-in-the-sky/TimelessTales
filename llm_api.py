class LargeLanguageModelAPI:
    """
    A unified API class to interface with different large language models.

    This class can instantiate an interface for either a local Hugging Face model
    or a remote OpenAI GPT model based on the provided arguments at instantiation.
    For Hugging Face models, the model and tokenizer are instantiated only once
    when a new model_id is set. The actual inference call will utilize the specified
    model to generate the output.

    Attributes:
        model_type (str): Type of model to use ('huggingface' or 'openai').
        model_id (str): The model identifier for Hugging Face or engine name for OpenAI.
        api_key (str, optional): The API key for OpenAI's model. Not required for local Hugging Face models.
        tokenizer (object, optional): Tokenizer instance for the Hugging Face model.
        model (object, optional): Model instance for the Hugging Face model.
    """

    def __init__(self, model_type, model_id, api_key=None, max_context_length=4000):
        """
        Initializes the API interface with the given model type and identifier.

        Parameters:
            model_type (str): The type of model, either 'huggingface' for a local model
                              or 'openai' for a remote model.
            model_id (str): The identifier for the Hugging Face model or the engine name for OpenAI.
            api_key (str, optional): The API key for OpenAI. Required if using an OpenAI model.
        """
        self.model_type = model_type
        self.api_key = api_key
        self._model_id = None
        self.tokenizer = None
        self.model = None
        self.max_context_length = max_context_length
        self.set_model_id(model_id)

    def set_model_id(self, model_id):
        """
        Sets a new model identifier for Hugging Face models and re-instantiates the tokenizer and model.

        Parameters:
            model_id (str): The identifier for the Hugging Face model.
        """
        if self.model_type == 'huggingface' and self._model_id != model_id:
            self._model_id = model_id
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id)

    def infer(self, text, temperature=0.7):
        """
        Generates an inference from the chosen large language model based on the input text.

        Parameters:
            text (str): The input text to be processed by the model.
            temperature (float, optional): Controls the randomness of the output for OpenAI models.
                                           Ignored for Hugging Face models. Defaults to 0.7.

        Returns:
            str: The generated text from the model.
        """
        if self.model_type == 'huggingface':
            return self._hf_local_model_inference(text)
        elif self.model_type == 'openai':
            return self._oai_remote_model_inference(text, temperature)
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

    def _oai_remote_model_inference(self, text, temperature):
        """
        Generates an inference using OpenAI's GPT model remotely.

        Parameters:
            text (str): The input text to be processed by the model.
            temperature (float): Controls the randomness of the output.

        Returns:
            str: The generated text from the model.
        """
        import openai

        if not self.api_key:
            raise ValueError("API key is required for OpenAI model inference.")

        openai.api_key = self.api_key

        response = openai.Completion.create(
            engine=self._model_id,
            prompt=text,
            max_tokens=50,
            temperature=temperature
        )

        return response.choices[0].text.strip()
