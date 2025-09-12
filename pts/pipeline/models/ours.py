from typing import Dict

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import numpy as np
from .modeling_llada import LLaDAModelLM


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                y = model(x_)
                hidden_states = y.hidden_states
                logits = y.logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                y = model(x)
                hidden_states = y.hidden_states
                logits = y.logits
                

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
    return x, hidden_states

class OurDualModel:
    def __init__(self, diff_id: str, llm_id: str, max_new_latents: int = 128, 
                 max_new_tokens=128, temperature: float = 0, do_sample: bool = False, **kwargs):
        self.diff_id = diff_id
        self.llm_id = llm_id
        self.max_new_latents = max_new_latents
        self.max_new_tokens = max_new_tokens
        
        self.temperature = temperature
        self.do_sample = do_sample
         
        self.diff_tokenizer = AutoTokenizer.from_pretrained(diff_id, trust_remote_code=True)
        self.diff_model = LLaDAModelLM.from_pretrained(
            diff_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to("cuda").eval()
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_id, use_fast=True)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_id, 
            torch_dtype=torch.bfloat16, 
        ).to("cuda").eval()

    def generate_plan(self, prompt: str, **kwargs) -> Dict:
        print(f"Generating Plan for {prompt}")
        m = [{"role": "user", "content": prompt}]
        prompt = self.diff_tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = self.diff_tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).to("cuda").unsqueeze(0)
        input_length = input_ids.shape[1]
        text, latents = generate(self.diff_model, input_ids, steps=128, gen_length=self.max_new_latents, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
        latents = latents[:, input_length:, :]
        text = text[:, input_length:]
        text = self.diff_tokenizer.decode(text[0], skip_special_tokens=True)
        print(text)
        return {
            "latents": latents,
            "text": text,
            "metadata": {
                "diff_id": self.diff_id,
                "max_new_latents": self.max_new_latents
            }
        }
        
    def generate_answer(self, user_prompt: str, latents=None, system_prompt=None, **kwargs) -> Dict:
        print(f"{user_prompt}\n{'='*50}")
        messages = []
        if system_prompt: 
            messages += [{"role": "system", "content": system_prompt}]
        messages += [{"role": "user", "content": user_prompt}]
        latent_token = "<|reserved_special_token_3|>"
        prompt = self.llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        if latents is not None:
            prompt = latent_token * 128 + prompt
            latents = latents.to("cuda")
        inputs = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
        ).to("cuda")
        input_len = inputs["input_ids"].shape[-1]
        kwargs = {}
        if latents is not None:
            kwargs['latents'] = latents
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.llm_tokenizer.eos_token_id,
                **kwargs
            )

        generated_text = self.llm_tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        return {
            "text": generated_text.strip(),
            "metadata": {
                "llm_id": self.llm_id,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "do_sample": self.do_sample
            }
        }