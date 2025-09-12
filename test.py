from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import numpy as np
from tqdm import tqdm

model_id = "/l/users/abdulrahman.mahmoud/heakl/PTS/checkpoints/llama3b-pixart-4bs-2grad-35klatents-lora/final_merged"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda").eval()

data_path = "/l/users/abdulrahman.mahmoud/heakl/PTS/outputs/train_5x/arc_challenge_20250905_225026_latents.npy"
data = np.load(data_path, allow_pickle=True)
latent_token = "<|reserved_special_token_3|>"
acc = 0
for idx in tqdm(range(len(data))):
    sample = data[idx]
    prompt = sample['prompt']
    latents = sample['latents']
    answer = sample['answer']
    messages = [
        {"role": "user", "content": prompt}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    # prompt = latent_token * 128 + prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to("cuda")
    input_len = inputs["input_ids"].shape[-1]
    # latents = torch.tensor(latents, device="cuda:0", dtype=torch.bfloat16).unsqueeze(0)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=None,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            min_new_tokens=2,
            # latents=latents,
            top_p=None,
            use_cache=False,
        )

    generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    acc += int(generated_text[0] == answer[0])
    
acc = round(acc*100 / len(data), 2)
print(acc)