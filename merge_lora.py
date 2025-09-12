from transformers import LlamaForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from argparse import ArgumentParser

def main(args):
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        
    ).to("cuda").eval()
    # files = os.listdir(args.lora_path)
    # for file in files:
    #     if os.path.isdir(os.path.join(args.lora_path, file)):
    #         args.lora_path = os.path.join(args.lora_path, file)
    #         break
    model = PeftModel.from_pretrained(model, args.lora_path)
    merged_model = model.merge_and_unload()
    run_name = args.lora_path.split("/")[-1]
    output_path = f"/l/users/abdulrahman.mahmoud/heakl/PTS/checkpoints/llama3b-pixart-4bs-2grad-35klatents-lora/merged_checkpoint-{run_name}"
    merged_model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(output_path)
    print(f"Model merged and saved to {output_path}")


    
if __name__ == "__main__":
    base_model_path = "meta-llama/Llama-3.2-3B-Instruct"
    lora_path = "/l/users/abdulrahman.mahmoud/heakl/PTS/checkpoints/llama3b-pixart-4bs-2grad-35klatents-lora/checkpoint-4340"
    parser = ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model", type=str, default=base_model_path, help="Path to the base model")
    parser.add_argument("--lora_path", type=str, default=lora_path, help="Path to the LoRA adapter")
    args = parser.parse_args()
    main(args)
    
    
