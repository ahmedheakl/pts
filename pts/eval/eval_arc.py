from argparse import ArgumentParser 
import sys
from tqdm import tqdm
import re

from datasets import load_dataset
from mathruler.grader import extract_boxed_content, grade_answer

from pts.pipeline.orchestrator import PTSPipeline



QUESTION_PROMPT_TEMPLATE = """Question: {question}\n{choices_text}"""
QUESTION_POSTFIX = """\nAnswer with a single letter (A, B, C, or D) and no explanation. You answer should start with "Answer: " and be followed by the letter of the answer you choose. Do not include any other text in your response."""

DIFFUSION_PLAN_TEMPLATE = """You are an expert in solving multiple-choice questions. Your task is to generate a detailed plan or reasoning step-by-step of how to tackle the question
provided below. The plan should be comprehensive and cover all necessary steps to arrive at the correct answer.
Do not provide the final answer, just the reasoning steps.
{question}"""

LLM_TEMPLATE = """You are an expert in solving multiple-choice questions. Given the following plan or reasoning, please solve the question. 
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

def compare_answers_dart(predicted, correct):
    pred_answer = extract_boxed_content(predicted.strip())
    return 1.0 if grade_answer(pred_answer, correct) else 0.0, predicted


def parse_args():
    parser = ArgumentParser(description="Run evaluation on a dataset using the PTS pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default="allenai/ai2_arc")
    parser.add_argument("--subset", default="ARC-Challenge")
    parser.add_argument("--split", default="test")
    parser.add_argument("--llm_refine", action="store_true")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--name_architecture", type=str, default="llm")
    args = parser.parse_args()
    return args

def main_llm():
    args = parse_args()
    print(f"Loading dataset {args.dataset} subset {args.subset} split {args.split}")
    dataset = load_dataset(args.dataset, args.subset, split=args.split)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(args.num_samples)) if args.num_samples > 0 else dataset
    pipeline = PTSPipeline.from_yaml(args.config)
    results = []
    for sample in tqdm(dataset, desc="Processing samples"):
        prepared_sample = prepare_arc_sample(sample)
        user_prompt = prepared_sample['input']
        llm_input = user_prompt + QUESTION_POSTFIX
        out = pipeline.generate_answer(llm_input)
        results.append({
            "is_correct": compare_answers(out['text'], prepared_sample['correct']),
            "question": prepared_sample['input'],
            "predicted_answer": out['text'],
            "ground_truth": prepared_sample['correct']
            
        })
    accuracy = sum(result['is_correct'] for result in results) / len(results)
    results = {
        "accuracy": accuracy,
        "num_samples": len(results),
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "config": args.config,
        "results": results,
    }
    from datetime import datetime
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"outputs/arc_results_{time_stamp}.json"
    import json
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Accuracy: {accuracy:.4f}")
    

def main():
    args = parse_args()
    print(f"Loading dataset {args.dataset} subset {args.subset} split {args.split}")
    dataset = load_dataset(args.dataset, args.subset, split=args.split)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(args.num_samples)) if args.num_samples > 0 else dataset
    pipeline = PTSPipeline.from_yaml(args.config)
    plans = []
    for sample in tqdm(dataset, desc="Processing samples"):
        prepared_sample = prepare_arc_sample(sample)
        user_prompt = prepared_sample['input']
        diffusion_input = DIFFUSION_PLAN_TEMPLATE.format(question=user_prompt)
        plan = pipeline.generate_plan(diffusion_input)
        plan['question'] = user_prompt
        plan['correct'] = prepared_sample['correct']
        plans.append(plan)
    
    acc = []
    for plan in tqdm(plans, desc="Running LLM refinement"):
        question = plan['question']
        llm_input = LLM_TEMPLATE.format(question=question, plan=plan['text']) + QUESTION_POSTFIX
        out = pipeline.generate_answer(llm_input)
        acc.append(compare_answers(out['text'], plan['correct']))
    accuracy = sum(acc) / len(acc)
    print(f"Accuracy: {accuracy:.4f}")
        
    return 0
    
    
if __name__ == "__main__":
    # sys.exit(main())
    sys.exit(main_llm())