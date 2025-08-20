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
from mathruler.grader import extract_boxed_content, grade_answer
from pts.pipeline.orchestrator import PTSPipeline
from pts.pipeline.utils import read_yaml


ARC_QUESTION_PROMPT_TEMPLATE = """Question: {question}\n{choices_text}"""
ARC_QUESTION_POSTFIX = """\nAnswer with a single letter (A, B, C, or D) and no explanation. Your answer should start with "Answer: " and be followed by the letter of the answer you choose. Do not include any other text in your response."""

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


def compare_answers_arc(predicted, correct):
    pred_answer = re.match(r"^(?:Answer:\s*)?([A-Da-d])\.?$", predicted.strip())
    if not pred_answer:
        return 0.0
    matched_group = pred_answer.group(1) or pred_answer.group(2)
    response = matched_group.strip()[0]
    return float(correct.lower().strip()[0] == response.lower())


def compare_answers_dart(predicted, correct):
    pred_answer = extract_boxed_content(predicted.strip())
    return 1.0 if grade_answer(pred_answer, correct) else 0.0





#------------------------------------------------------------------------------------




# Template used to prompt the model – GSM8K problems are just questions.
# You can modify this (e.g. add "Answer:" or "Solve step by step") if desired.
GSM8K_PROMPT_TEMPLATE = "{question}\n"

def extract_numeric_answer(text: str) -> str:
    """
    Extract the final numeric answer from a GSM8K answer string or a model’s output.
    The GSM8K ground‑truth puts the answer after a `####` token:contentReference[oaicite:2]{index=2},
    so we first look for that.  If it’s missing (e.g. in a model’s answer), we fall back
    to picking the last number in the text.  Commas are removed so that '70,000' == '70000'.
    """
    # Look for a number following '####'
    match = re.search(r"####\s*([-+]?[0-9][\d,\.]*)", text)
    if match:
        answer = match.group(1)
    else:
        # Otherwise take the last standalone number in the string
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        answer = numbers[-1] if numbers else ""
    return answer.replace(",", "").strip()

def compare_answers_gsm8k(predicted: str, correct: str) -> float:
    """
    Compare a model’s answer with the ground‑truth solution from GSM8K.
    Both arguments are strings; the ground‑truth string contains the full explanation
    and final answer:contentReference[oaicite:3]{index=3}.  Returns 1.0 if the extracted numeric
    answers match, 0.0 otherwise.
    """
    pred_answer = extract_numeric_answer(predicted)
    gold_answer = extract_numeric_answer(correct)
    return 1.0 if pred_answer and (pred_answer == gold_answer) else 0.0

def prepare_gsm8k_sample(item: dict) -> dict:
    """
    Prepare a GSM8K sample for evaluation or model prompting.
    `item` is a dict with keys 'question' and 'answer'.
    Returns a dict with the formatted prompt in `input` and the full ground‑truth
    answer string in `correct`.
    """
    question = item["question"]
    answer   = item["answer"]
    input_text = GSM8K_PROMPT_TEMPLATE.format(question=question)
    return {"input": input_text, "correct": answer}







#------------------------------------------------------------------------------------


def worker_evaluate_llm(
    rank: int,
    samples,
    return_dict,
    configs,
    architecture_name,
    compare_func=compare_answers_arc,
    postfix=ARC_QUESTION_POSTFIX,
):
    torch.cuda.set_device(rank)
    pipeline = PTSPipeline.from_yaml(configs)
    local_results = []
    for i, sample in enumerate(tqdm(samples, desc=f"[GPU {rank}] LLM")):
        user_prompt = sample["input"]
        llm_input = user_prompt + postfix
        out = pipeline.generate_answer(llm_input, architecture_name)
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
    architecture_name,
    compare_func=compare_answers_arc,
    postfix=ARC_QUESTION_POSTFIX, 
):
    torch.cuda.set_device(rank)
    pipeline = PTSPipeline.from_yaml(configs)
    local_results = {}
    plans = []
    for i, sample in enumerate(tqdm(samples, desc=f"[GPU {rank}] Diff")):
        user_prompt = sample["input"]
        diffusion_input = DIFFUSION_HTNTS_TEMPLATE.format(question=user_prompt)
        plan = pipeline.generate_plan(diffusion_input, architecture_name)
        plan["question"] = user_prompt
        plan["correct"] = sample["correct"]
        plans.append(plan)
        print(plan["text"])

    local_results = []
    for plan in tqdm(plans, desc="Running LLM refinement"):
        question = plan["question"]
        llm_input = LLM_TEMPLATE.format(question=question, plan=plan["text"]) + postfix
        out = pipeline.generate_answer(llm_input, architecture_name)
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
    parser = ArgumentParser(
        description="Run evaluation on a dataset using the PTS pipeline"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default="allenai/ai2_arc")
    parser.add_argument("--llm_refine", action="store_true")
    parser.add_argument("--llm_eval", action="store_true")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--architecture_name", type=str, default="llm-only")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(f"Loading dataset {args.dataset}")
    if args.dataset == "arc_easy":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        process_func = prepare_arc_sample
        compare_func = compare_answers_arc
        prefix = ARC_QUESTION_POSTFIX
    elif args.dataset == "arc_challenge":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        process_func = prepare_arc_sample
        compare_func = compare_answers_arc
        prefix = ARC_QUESTION_POSTFIX
    elif "dart" in args.dataset:
        level = int(args.dataset.split("-")[-1])
        dataset = load_dataset("hkust-nlp/dart-math-pool-math", split="train")
        dataset = dataset.filter(
            lambda x: x["query_metadata"]["level"] == level, num_proc=32
        )
        process_func = prepare_dart_sample
        compare_func = compare_answers_dart
        prefix = DART_QUESTION_PREFIX
    elif "gsm8k" in args.dataset:
        dataset = load_dataset("gsm8k", "main", split="test")
        process_func = prepare_gsm8k_sample
        compare_func = compare_answers_gsm8k
        prefix = ""  # GSM8K does not need a postfix
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
    worker_func = worker_evaluate_llm if args.llm_eval else worker_evaluate
    print(f"worker_func = {worker_func}")

    for rank in range(world_size):
        p = mp.Process(
            target=worker_func,
            args=(rank, chunks[rank], return_dict, args.config, compare_func, prefix, args.architecture_name),
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
        "architecture_name" : args.architecture_name,
        "dataset": args.dataset,
        "num_samples": len(all_samples),
        "accuracy": accuracy,
        "is_llm": args.llm_eval,
        "diffusion_template": DIFFUSION_HTNTS_TEMPLATE,
        "llm_template": LLM_TEMPLATE,
        "diffusion_max_new_tokens": yaml_config["diffusion"]["max_new_tokens"],
        "llm_max_new_tokens": yaml_config["llm"]["max_new_tokens"],
        "results": all_results,
    }
    print(f"Accuracy: {accuracy:.2f}")

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "dart" in args.dataset:
        dataset_name = "dart"
    elif "gsm8k" in args.dataset:
        dataset_name = "gsm8k"
    else:
        dataset_name = "arc"
    output_file = f"{yaml_config['runtime']['output_dir']}/{args.architecture_name}/{dataset_name}_evaluation_{time_stamp}_{args.architecture_name}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)
    return 0

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
