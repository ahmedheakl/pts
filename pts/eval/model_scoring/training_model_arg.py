# Download and preprocess the IBM Argument Quality Ranking dataset for Qwen3-4B-Instruct fine-tuning
import os
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
from peft import LoraConfig, get_peft_model, TaskType

def get_argument_quality_data():
	# Load the dataset from Hugging Face
	dataset = load_dataset("ibm-research/argument_quality_ranking_30k", "argument_quality_ranking", split="train")
	# Convert to pandas DataFrame for easier manipulation
	df = pd.DataFrame(dataset.to_dict())
	# Select only 'argument' as input and 'WA' as output
	df = df[["argument", "WA"]]
	return df



# --- Fine-tuning Qwen3-4B-Instruct ---

def format_for_supervised_finetuning(df):
	# Format as instruction tuning: input = argument, output = WA
	return [
		{
			"input": row["argument"],
			"output": str(row["WA"])
		}
		for _, row in df.iterrows()
	]

class ArgumentQualityDataset(torch.utils.data.Dataset):
	def __init__(self, data, tokenizer, max_length=512):
		self.data = data
		self.tokenizer = tokenizer
		self.max_length = max_length

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		item = self.data[idx]
		prompt = f"Rate this essay from 0 to 100: {item['input']}"
		target = str(float(item['output']) * 100)
		inputs = self.tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors="pt", padding="max_length")
		labels = self.tokenizer(target, truncation=True, max_length=32, return_tensors="pt", padding="max_length")
		input_ids = inputs.input_ids.squeeze()
		label_ids = labels.input_ids.squeeze()
		# Concatenate prompt and target for causal LM
		full_input = torch.cat([input_ids, label_ids])
		# Mask prompt tokens for loss
		labels = torch.cat([
			torch.full_like(input_ids, -100),  # ignore prompt
			label_ids
		])
		return {
			"input_ids": full_input,
			"labels": labels
		}

def train_qwen3_4b_instruct():
	model_name = "Qwen/Qwen3-4B-Instruct-2507"
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(model_name)

	# --------------- Optimization ---------------
	# Disable cache to save memory
	model.config.use_cache = False
	# Enable gradient checkpointing
	model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
	# LoRA configuration
	peft_cfg = LoraConfig(
		task_type=TaskType.CAUSAL_LM,
		r=16,
		lora_alpha=32,
		lora_dropout=0.05,
		target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
	)
	# ------------------------------------------
 
	model = get_peft_model(model, peft_cfg)
	df = pd.read_csv("argument_quality_train.csv")
	data = format_for_supervised_finetuning(df)
	print("Sample formatted data:", data[:2])  # Print first 2 samples for verification
	dataset = ArgumentQualityDataset(data, tokenizer)
	training_args = TrainingArguments(
		output_dir="./qwen3-4b-arg-quality",
		per_device_train_batch_size=1,
		gradient_accumulation_steps=4,
		num_train_epochs=1,
		logging_steps=10,
		save_steps=100,
		bf16=True, fp16=False,
		gradient_checkpointing=True,
		optim="paged_adamw_8bit",
		max_grad_norm=0.5,
		report_to="none",
		)
	# training_args = TrainingArguments(
    # output_dir="./qwen3-4b-arg-quality",
    # per_device_train_batch_size=1,           #  
    # gradient_accumulation_steps=8,           # keep effective batch ~8
    # num_train_epochs=1,
    # logging_steps=10,
    # save_steps=100,
    # fp16=True,                               # keep fp16
    # gradient_checkpointing=True,             # big saver
    # optim="paged_adamw_8bit",                # bitsandbytes
    # report_to="none",
    # max_grad_norm=0.5
	# )
 
	data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) #Inputs are dynamically padded to the maximum length of a batch if they are not all of the same length.
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=dataset,
		data_collator=data_collator,
	)
	trainer.train()
	model.save_pretrained("./qwen3-4b-arg-quality")
	tokenizer.save_pretrained("./qwen3-4b-arg-quality")




if __name__ == "__main__":
	df = get_argument_quality_data()
	print("Sample data:", df.head())
	# Save preprocessed data for training
	df.to_csv("argument_quality_train.csv", index=False)
	train_qwen3_4b_instruct()
