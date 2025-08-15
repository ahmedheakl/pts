from argparse import ArgumentParser 
import sys
from tqdm import tqdm
import re
import math
import json
from datetime import datetime

from datasets import load_dataset
import torch
import torch.multiprocessing as mp

from pts.pipeline.orchestrator import PTSPipeline



QUESTION_PROMPT_TEMPLATE = """Question: {question}\n{choices_text}"""
QUESTION_POSTFIX = """\nAnswer with a single letter (A, B, C, or D) and no explanation. You answer should start with "Answer: " and be followed by the letter of the answer you choose. Do not include any other text in your response."""

DIFFUSION_PLAN_TEMPLATE = """You are an expert in solving multiple-choice questions.
Your task is to generate a detailed plan or reasoning step-by-step of how to tackle the question
provided below. The plan should be comprehensive and cover all necessary steps to arrive at the correct answer.
Do not provide the final answer, just the reasoning steps.
{question}"""

LLM_TEMPLATE = """You are an expert in solving multiple-choice questions.
Given the following plan or reasoning, please solve the question. 
Plan:
{plan}
{question}"""


def prepare_arc_sample(item):
    question = item["question"]
    choices = item["choices"]
    answer_key = item["answerKey"]
    choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices["text"])])
    input_text = QUESTION_PROMPT_TEMPLATE.format(question=question, choices_text=choices_text)
    labels = choices["label"]
    correct_idx = labels.index(answer_key)
    answer_key = chr(65 + correct_idx)  # Convert index to letter (A, B, C, D)
    return {"input": input_text, "correct": answer_key}


def compare_answers(predicted, correct):
    pred_answer = re.match(r"^(?:Answer:\s*)?([A-Da-d])\.?$", predicted.strip())
    if not pred_answer: 
        return 0.0
    matched_group = pred_answer.group(1) or pred_answer.group(2)
    response = matched_group.strip()[0]
    return float(correct.lower().strip()[0] == response.lower())

def worker_evaluate(rank: int, samples, return_dict, configs):
    torch.cuda.set_device(rank)
    pipeline = PTSPipeline.from_yaml(configs)
    local_results = {}
    plans = []
    for i, sample in enumerate(tqdm(samples, desc=f"[GPU {rank}] Diff")):
        user_prompt = sample['input']
        diffusion_input = DIFFUSION_PLAN_TEMPLATE.format(question=user_prompt)
        plan = pipeline.generate_plan(diffusion_input)
        plan['question'] = user_prompt
        plan['correct'] = sample['correct']
        plans.append(plan)
        print(plan['text'])
        
    local_results = []
    for plan in tqdm(plans, desc="Running LLM refinement"):
        question = plan['question']
        llm_input = LLM_TEMPLATE.format(question=question, plan=plan['text']) + QUESTION_POSTFIX
        out = pipeline.generate_answer(llm_input)
        print(out['text'])
        local_results.append({
            "is_correct": compare_answers(out['text'], plan['correct']),
            "question": question,
            "predicted_plan": plan['text'],
            "predicted_answer": out['text'],
            "ground_truth": plan['correct'],
        })
    return_dict[rank] = local_results

def parse_args():
    parser = ArgumentParser(description="Run evaluation on a dataset using the PTS pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default="allenai/ai2_arc")
    parser.add_argument("--subset", default="ARC-Challenge")
    parser.add_argument("--split", default="test")
    parser.add_argument("--llm_refine", action="store_true")
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(f"Loading dataset {args.dataset} subset {args.subset} split {args.split}")
    dataset = load_dataset(args.dataset, args.subset, split=args.split)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(args.num_samples)) if args.num_samples > 0 else dataset
    all_samples = [prepare_arc_sample(sample) for sample in dataset]
    
    world_size = torch.cuda.device_count()
    chunk_size = math.ceil(len(all_samples) / world_size)
    chunks = [all_samples[i:i+chunk_size] for i in range(0, len(all_samples), chunk_size)]

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for rank in range(world_size):
        p = mp.Process(target=worker_evaluate, args=(rank, chunks[rank], return_dict, args.config))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        
    all_results = []
    for rank in range(world_size):
        all_results.extend(return_dict[rank])
    acc = [result['is_correct'] for result in all_results]
    accuracy = sum(acc) * 100 / len(acc)
    print(f"Accuracy: {accuracy:.2f}")
    
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"outputs/arc_evaluation_{time_stamp}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    return 0
    
    
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())