from typing import Dict, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LLM:
    def __init__(self, model_id: str, device: str = "cuda:0", max_new_tokens: int = 512,
                 temperature: float = 0.2, do_sample: bool = False, **kwargs):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
        

    def generate(self, system_prompt: Optional[str], user_prompt: str) -> Dict:
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        else:
            messages = [{"role": "user", "content": user_prompt}]

        # Apply chat template and tokenize
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")

        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode the output tokens
        generated_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        return {
            "text": generated_text.strip(),
            "metadata": {
                "model_id": self.model_id,
                "device": self.device,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "do_sample": self.do_sample
            }
        }