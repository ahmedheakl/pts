from argparse import ArgumentParser
import sys
from tqdm import tqdm
import re
import math
import json
from datetime import datetime

from datasets import load_dataset, load_from_disk
import torch
import torch.multiprocessing as mp
from mathruler.grader import extract_boxed_content, grade_answer
from pts.pipeline.orchestrator import PTSPipeline
from pts.pipeline.utils import read_yaml
from pts.constants import Pipelines
import os

ARC_QUESTION_PROMPT_TEMPLATE = """Question: {question}\n{choices_text}"""
MCQ_QUESTION_POSTFIX = """\nAnswer with a single letter (A, B, C, or D) and no explanation. Your answer should start with "Answer: " and be followed by the letter of the answer you choose. Do not include any other text in your response."""

DART_QUESTION_PROMPT_TEMPLATE = """Question: {question}"""
DART_QUESTION_PREFIX = (
    """\nThe final answer MUST BE put in \\boxed{{}} and no explanation."""
)

DIFFUSION_PLAN_TEMPLATE = """You are an expert in solving multiple-choice questions.
Your task is to generate a detailed plan or reasoning step-by-step of how to tackle the question
provided below. The plan should be comprehensive and cover all necessary steps to arrive at the correct answer.
Do not provide the final answer, just the reasoning steps.
{question}"""

DIFFUSION_HTNTS_TEMPLATE = """You are a careful problem-solving planner.

Task: Produce ONLY a short list of HINTS that help solve the question. 
Do NOT state or imply the final answer. Do NOT mention any option letter 
(A, B, C, or D). Do NOT quote any option text verbatim. 
If you find yourself about to reveal a specific option or an answer, 
replace it with “[HIDDEN]”.

Format:
- Key facts to recall (2–4 bullets)
- Reasoning steps or elimination rules (2–5 bullets)
- Useful equations or definitions (if relevant)
- Edge cases or common traps (optional)

Be concise (<=120 words). No “Answer:” line. No letters A–D. No option text.

Question (stem only):
{question}
"""

LLM_TEMPLATE = """You are an expert in solving multiple-choice questions.
Given the following plan or reasoning, please solve the question. If the plan contains any explicit answer or option letter, ignore it and solve from the hints + question only.
Plan:
{plan}
{question}"""


def prepare_dart_sample(item):
    question = item["query"]
    answer_key = item["gt_ans"]
    input_text = DART_QUESTION_PROMPT_TEMPLATE.format(question=question)
    return {"input": input_text, "correct": answer_key}


def prepare_arc_sample(item):
    question = item["question"]
    choices = item["choices"]
    answer_key = item["answerKey"]
    choices_text = "\n".join(
        [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices["text"])]
    )
    input_text = ARC_QUESTION_PROMPT_TEMPLATE.format(
        question=question, choices_text=choices_text
    )
    labels = choices["label"]
    correct_idx = labels.index(answer_key)
    answer_key = chr(65 + correct_idx)  # Convert index to letter (A, B, C, D)
    return {"input": input_text, "correct": answer_key}


def compare_answers_mcq(predicted, correct):
    pred_answer = re.match(r"^(?:Answer:\s*)?([A-Da-d])\.?$", predicted.strip())
    if not pred_answer:
        return 0.0
    matched_group = pred_answer.group(1) or pred_answer.group(2)
    response = matched_group.strip()[0]
    return float(correct.lower().strip()[0] == response.lower())


def compare_answers_dart(predicted, correct):
    pred_answer = extract_boxed_content(predicted.strip())
    return 1.0 if grade_answer(pred_answer, correct) else 0.0


# ------------------------------------------------------------------------------------


# Template used to prompt the model – GSM8K problems are just questions.
# You can modify this (e.g. add "Answer:" or "Solve step by step") if desired.
GSM8K_PROMPT_TEMPLATE = "{question}\n"


def extract_numeric_answer(text: str) -> str:
    match = re.search(r"####\s*([-+]?[0-9][\d,\.]*)", text)
    if match:
        answer = match.group(1)
    else:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        answer = numbers[-1] if numbers else ""
    return answer.replace(",", "").strip()


def compare_answers_gsm8k(predicted: str, correct: str) -> float:
    pred_answer = extract_numeric_answer(predicted)
    gold_answer = extract_numeric_answer(correct)
    return 1.0 if pred_answer and (pred_answer == gold_answer) else 0.0


def prepare_gsm8k_sample(item: dict) -> dict:
    question = item["question"]
    answer = item["answer"]
    input_text = GSM8K_PROMPT_TEMPLATE.format(question=question)
    return {"input": input_text, "correct": answer}

def prepare_mmlu_sample(item):
    question = item['question']
    choices = item['choices']
    prompt = f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
    answer_id = item['answer']
    answer = chr(ord('A')+answer_id)
    return {"input": prompt, "correct": answer}

# ------------------------------------------------------------------------------------


def worker_evaluate_single(
    rank: int,
    samples,
    return_dict,
    configs,
    name_architecture,
    compare_func=compare_answers_mcq,
    postfix=MCQ_QUESTION_POSTFIX,
):
    torch.cuda.set_device(rank)
    pipeline = PTSPipeline.from_yaml(configs)
    local_results = []
    for i, sample in enumerate(tqdm(samples, desc=f"[GPU {rank}] Answer")):
        user_prompt = sample["input"]
        llm_input = user_prompt + postfix
        out = pipeline.generate_answer(llm_input, name_architecture)
        print(out["text"])
        local_results.append(
            {
                "is_correct": compare_func(out["text"], sample["correct"]),
                "question": user_prompt,
                "predicted_answer": out["text"],
                "ground_truth": sample["correct"],
            }
        )
    return_dict[rank] = local_results


def worker_evaluate(
    rank: int,
    samples,
    return_dict,
    configs,
    name_architecture,
    compare_func=compare_answers_mcq,
    postfix=MCQ_QUESTION_POSTFIX,
):
    torch.cuda.set_device(rank)
    pipeline = PTSPipeline.from_yaml(configs)
    local_results = {}
    plans = []
    for i, sample in enumerate(tqdm(samples, desc=f"[GPU {rank}] Plan")):
        user_prompt = sample["input"]
        diffusion_input = DIFFUSION_HTNTS_TEMPLATE.format(question=user_prompt)
        plan = pipeline.generate_plan(diffusion_input, name_architecture)
        plan["question"] = user_prompt
        plan["correct"] = sample["correct"]
        plans.append(plan)
        print(plan["text"])

    local_results = []
    for plan in tqdm(plans, desc=f"[GPU {rank}] Answer"):
        question = plan["question"]
        llm_input = LLM_TEMPLATE.format(question=question, plan=plan["text"]) + postfix
        out = pipeline.generate_answer(llm_input, name_architecture)
        print(out["text"])
        local_results.append(
            {
                "is_correct": compare_func(out["text"], plan["correct"]),
                "question": question,
                "predicted_plan": plan["text"],
                "predicted_answer": out["text"],
                "ground_truth": plan["correct"],
            }
        )
    return_dict[rank] = local_results


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default="arc_easy")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--ds_cache", type=str, default="cached_datasets")
    parser.add_argument("--name_architecture", choices=Pipelines.all_architectures())
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(f"Loading dataset {args.dataset}")
    cache_path = os.path.join(args.ds_cache, args.dataset)
    dataset = None
    if os.path.exists(cache_path):
        dataset = load_from_disk(cache_path)
    if args.dataset == "arc_easy":
        if not dataset:
            dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
            dataset.save_to_disk(cache_path)
        process_func = prepare_arc_sample
        compare_func = compare_answers_mcq
        prefix = MCQ_QUESTION_POSTFIX
    elif args.dataset == "arc_challenge":
        if not dataset:
            dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
            dataset.save_to_disk(cache_path)
        process_func = prepare_arc_sample
        compare_func = compare_answers_mcq
        prefix = MCQ_QUESTION_POSTFIX
    elif "dart" in args.dataset:
        if not dataset:
            dataset = load_dataset("hkust-nlp/dart-math-pool-math", split="train")
            dataset.save_to_disk(cache_path)
        level = int(args.dataset.split("-")[-1])
        dataset = dataset.filter(
            lambda x: x["query_metadata"]["level"] == level, num_proc=32
        )
        process_func = prepare_dart_sample
        compare_func = compare_answers_dart
        prefix = DART_QUESTION_PREFIX
    elif "gsm8k" in args.dataset:
        if not dataset:
            dataset = load_dataset("gsm8k", "main", split="test")
            dataset.save_to_disk(cache_path)
        process_func = prepare_gsm8k_sample
        compare_func = compare_answers_gsm8k
        prefix = ""  # GSM8K does not need a postfix
    elif "mmlu" in args.dataset:
        if not dataset:
            dataset = load_dataset("cais/mmlu", "all", split="test")
            dataset.save_to_disk(cache_path)
        process_func = prepare_mmlu_sample
        compare_func = compare_answers_mcq
        prefix = MCQ_QUESTION_POSTFIX
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    dataset = dataset.shuffle(seed=42)
    dataset = (
        dataset.select(range(args.num_samples)) if args.num_samples > 0 else dataset
    )
    all_samples = [process_func(sample) for sample in dataset]

    world_size = torch.cuda.device_count()
    chunk_size = math.ceil(len(all_samples) / world_size)
    chunks = [
        all_samples[i : i + chunk_size] for i in range(0, len(all_samples), chunk_size)
    ]

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    worker_func = (
        worker_evaluate_single if "only" in args.name_architecture else worker_evaluate
    )

    for rank in range(world_size):
        p = mp.Process(
            target=worker_func,
            args=(
                rank,
                chunks[rank],
                return_dict,
                args.config,
                args.name_architecture,
                compare_func,
                prefix,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    all_results = []
    for rank in range(world_size):
        all_results.extend(return_dict[rank])

    acc = [result["is_correct"] for result in all_results]
    accuracy = sum(acc) * 100 / len(acc)
    yaml_config = read_yaml(args.config)
    all_results = {
        "diffusion_model": yaml_config["diffusion"]["model_id"],
        "llm": yaml_config["llm"]["model_id"],
        "name_architecture": args.name_architecture,
        "dataset": args.dataset,
        "num_samples": len(all_samples),
        "accuracy": accuracy,
        "plan_template": DIFFUSION_HTNTS_TEMPLATE,
        "answer_template": LLM_TEMPLATE,
        "diffusion_max_new_tokens": yaml_config["diffusion"]["max_new_tokens"],
        "llm_max_new_tokens": yaml_config["llm"]["max_new_tokens"],
        "results": all_results,
    }
    print(f"Accuracy: {accuracy:.2f}")

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{yaml_config['runtime']['output_dir']}/{args.name_architecture}/{args.dataset}_evaluation_{time_stamp}_{args.name_architecture}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
