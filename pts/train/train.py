from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from glob import glob
from datasets import Dataset, Features, Value, Array2D, concatenate_datasets
from tabulate import tabulate
import os
import numpy as np
from tqdm import tqdm


model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = "<|finetune_right_pad_id|>"
tokenizer.padding_side = "right"
model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
model.init_latent_projector()


LLM_TEMPLATE = """You are an expert in solving multiple-choice questions.
Given the following plan or reasoning, please solve the question. If the plan contains any explicit answer or option letter, ignore it and solve from the hints + question only.
{question}"""

run_name = "llama3b-pixart-8bs"
data_folder = "outputs/train"
batch_size = 8
gradient_accumulation = 1
num_epochs = 3
learning_rate = 3e-4
weight_decay = 0.001
warmup_steps = 300
save_steps = 875
output_dir = f"/l/users/abdulrahman.mahmoud/heakl/PTS/checkpoints/{run_name}"
eval_size_per_type = 10
os.makedirs(output_dir, exist_ok=True)
ds_root = "/l/users/abdulrahman.mahmoud/heakl/PTS/cached_datasets"
cached_train_data_path = f"{ds_root}/train_data_{os.path.basename(model_id)}"
cached_eval_data_path = f"{ds_root}/eval_data_{os.path.basename(model_id)}"


if os.path.exists(cached_train_data_path) and os.path.exists(cached_eval_data_path):
    train_data = Dataset.load_from_disk(cached_train_data_path)
    eval_data = Dataset.load_from_disk(cached_eval_data_path)
    print(f"Loaded cached data: {len(train_data)} training samples, {len(eval_data)} eval samples.")
else:
    data_files = glob(f"{data_folder}/*.npy")
    questions, answers, latents, plans, ds_types = [], [], [], [], []
    for ds_path in data_files:
        raw_data = np.load(ds_path, allow_pickle=True)
        dname = "_".join(os.path.basename(ds_path).split("_")[-4::-1][::-1])
        for d in tqdm(raw_data, desc=f"Reading {dname}"):
            questions.append(d["prompt"])
            answers.append(d["answer"])
            latents.append(d["latents"])
            plans.append(d["plan"])
            ds_types.append(d["dataset"])
    print(f"Loaded {len(questions)} samples from {len(data_files)} files.")

    data = []
    eval_data = {}
    repeated, skipped, total = 0, 0, 0
    split_pattern = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    for question, answer, latent, plan, ds_type in tqdm(zip(questions, answers, latents, plans, ds_types), desc="Preparing data"):
        conversations = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        msg = tokenizer.apply_chat_template(conversations, tokenize=False)
        prompt, completion = msg.split(split_pattern)
        prompt += split_pattern
        latent = latent.astype(np.float16, copy=False)
        sample = {"prompt": prompt, "completion": completion, "ds_type": ds_type, "latents": latent}
        if len(eval_data.get(ds_type, [])) < eval_size_per_type:
            eval_data[ds_type] = eval_data.get(ds_type, [])
            eval_data[ds_type].append(sample)
        else:
            data.append(sample)

    features = Features({
        "prompt": Value("string"),
        "completion": Value("string"),
        "ds_type": Value("string"),
        "latents": Array2D(shape=(128, 4096), dtype="float16"),
    })
    def to_sharded_dataset(rows, features, shard_size=128):
        shards = []
        for i in tqdm(range(0, len(rows), shard_size), desc="Sharing data"):
            shard = rows[i:i+shard_size]
            for s in shard:
                s["latents"] = s["latents"].astype(np.float16, copy=False)
            shards.append(Dataset.from_list(shard, features=features))
        return concatenate_datasets(shards) if len(shards) > 1 else shards[0]
    del raw_data, questions, answers, latents, plans, ds_types
    eval_data = [item for sublist in eval_data.values() for item in sublist]
    eval_data  = Dataset.from_list(eval_data, features=features)
    eval_data.save_to_disk(cached_eval_data_path)
    train_data = to_sharded_dataset(data, features, shard_size=128)
    train_data.save_to_disk(cached_train_data_path)
    
    print(f"Training on {len(train_data)} samples, evaluating on {len(eval_data)} samples.")

# train_data = train_data.select(range(10))
training_args = SFTConfig(
    do_train=True,
    do_eval=False,
    do_predict=False,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation,
    eval_strategy="no",
    num_train_epochs=num_epochs,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    logging_steps=2,
    save_steps=save_steps,
    output_dir=output_dir,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_steps=warmup_steps,
    fp16=False,
    bf16=True,
    dataloader_num_workers=2,
    dataset_num_proc=2,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    run_name=run_name,
    report_to="wandb",
    prediction_loss_only=False,
)
for p in model.model.parameters():
    p.requires_grad = False
for p in model.latents_projector.parameters():
    p.requires_grad = True
for p in model.lm_head.parameters():
    p.requires_grad = False

stat = []
for i, (name, p) in enumerate(model.named_parameters()):
    stat.append([i, name, p.shape, p.requires_grad])
print(tabulate(stat, headers=["idx", "name", "shape", "trainable"]))

trainer = SFTTrainer(
    args=training_args,
    model=model,
    train_dataset=train_data,
    processing_class=tokenizer,
)
trainer.train()