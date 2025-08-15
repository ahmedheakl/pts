from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Diffusion:
    """
    Adapter for your diffusion text model.
    Replace `generate()` with a real call to your model (local or service).
    """
    def __init__(self, model_id: str, device: str, max_new_tokens: int = 256, impl: str = "local"):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.impl = impl

        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="auto"
        ).to(device)

    def generate(self, prompt: str, **kwargs) -> Dict:
        """
        Generate text using the diffusion model.
        """
        # Tokenize the input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate output
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, **kwargs)

        # Decode the output tokens
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "text": text,
            "metadata": {
                "model_id": self.model_id,
                "impl": self.impl,
                "device": self.device,
                "max_new_tokens": self.max_new_tokens
            }
        }