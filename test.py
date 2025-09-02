from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import numpy as np

model_id = "/l/users/abdulrahman.mahmoud/heakl/PTS/checkpoints/llama3b-pixart-8bs/checkpoint-1750"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

data_path = "outputs/train/arc_easy_20250824_224600_latents.npy"
data = np.load(data_path, allow_pickle=True)

for idx in range(len(data)):
    sample = data[idx]
    prompt = sample['prompt']
    latents = sample['latents']
    answer = sample['answer']
    messages = [
        {"role": "user", "content": prompt}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    input_len = inputs["input_ids"].shape[-1]
    latents = torch.tensor(latents, device="cuda", dtype=torch.bfloat16).unsqueeze(0)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=None,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            min_new_tokens=2,
            latents=latents,
            top_p=None,
            use_cache=False,
        )

    generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    print(answer, generated_text)
    if idx > 5: break
