from argparse import ArgumentParser
import sys
from tqdm import tqdm
import re
import math
import json
from datetime import datetime
from functools import partial

from datasets import load_dataset, load_from_disk
import torch
import torch.multiprocessing as mp
from mathruler.grader import extract_boxed_content, grade_answer
from pts.pipeline.orchestrator import PTSPipeline
from pts.pipeline.utils import read_yaml
from pts.constants import Pipelines
import os



from pts.eval.compare_prepare.humaneval import run_humaneval, prepare_humaneval_sample

PLAN_PROMPT = """You are a senior software engineer preparing to implement a Python function. 
You will be given the beginning of a Python function and a docstring describing its desired behavior.

Your ONLY task at this stage is to produce a detailed, structured implementation plan.
This plan is crucial — it will serve as the blueprint for writing the actual code later. The quality of the implementation depends entirely on the clarity and completeness of your plan.
Do not write any actual code. Instead:

1. Carefully analyze the function signature and the docstring.

2. Decompose the problem into logical, sequential steps.

3. Identify edge cases, input constraints, validation logic, special conditions, or any hidden complexity.

4. If applicable, clarify any assumptions or ambiguities.


Here is the function you need to plan for: \n "{prompt}"\n
Be clear Do NOT write any actual code. Be concise (<=120 words).
"""


CODER_PROMPT_SINGLE = """You are an expert Python developer. 
You will be given the beginning of a Python function, a descriptive docstring.
Your task is to complete the function according to the specifications in the docstring.
Be concise, correct, and handle edge cases where appropriate. Use best practices in Python code.
Here is the Python function : {prompt}"""

CODER_PROMPT_AFTERPLAN = """You are an expert Python developer. 
You will be given the beginning of a Python function, a descriptive docstring, and detailed implementation indications.
Your task is to complete the function according to the specifications in the docstring with the help of the indications.
Be concise, correct, and handle edge cases where appropriate. Use best practices in Python code.
Here is the Python function : {prompt}\n
Here are the implementation indications : {plan}"""




def worker_evaluate_single(
    rank: int,
    samples,
    return_dict,
    configs,
    name_architecture,
    run_test_func,
    postfix,
):
    torch.cuda.set_device(rank)
    pipeline = PTSPipeline.from_yaml(configs)
    local_results = []
    for i, sample in enumerate(tqdm(samples, desc=f"[GPU {rank}] Answer")):
        user_prompt = CODER_PROMPT_SINGLE.format(prompt=sample["input"])
        llm_input = user_prompt + postfix
        out = pipeline.generate_answer(llm_input, name_architecture)
        print(out["text"])
        local_results.append(
            {
                "is_correct": run_test_func(out["text"], sample["setup"]), #runs the test of the method in out["text"] on the setup
                "question": user_prompt,
                "predicted_answer": out["text"],
                "ground_truth": sample["setup"],
            }
        )
    return_dict[rank] = local_results


def worker_evaluate(
    rank: int,
    samples, # list of dicts with "input" and "setup" keys (from processed function)
    return_dict,
    configs,
    name_architecture,
    run_test_func,
    postfix,
):
    torch.cuda.set_device(rank)
    pipeline = PTSPipeline.from_yaml(configs)
    local_results = {}
    plans = []
    for i, sample in enumerate(tqdm(samples, desc=f"[GPU {rank}] Plan")):
        user_prompt = sample["input"]
        first_model_input = PLAN_PROMPT.format(prompt=user_prompt)
        plan = pipeline.generate_plan(first_model_input, name_architecture)
        #add these to the dict
        plan["question"] = user_prompt
        plan["setup"] = sample["setup"]
        plans.append(plan)
        print(plan["text"])

    local_results = []
    for plan in tqdm(plans, desc=f"[GPU {rank}] Answer"):
        question = plan["question"]
        second_model_input = CODER_PROMPT_AFTERPLAN.format(prompt=question, plan=plan["text"]) + postfix
        out = pipeline.generate_answer(second_model_input, name_architecture)
        print(out["text"])
        local_results.append(
            {
                "is_correct": run_test_func(out["text"], plan["setup"]), #runs the test of the method in out["text"] on the setup
                "question": question,
                "predicted_plan": plan["text"],
                "predicted_answer": out["text"],
                "Test Setup": plan["setup"],
            }
        )
    return_dict[rank] = local_results


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default="arc_easy")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--ds_cache", type=str, default="/l/users/abdulrahman.mahmoud/heakl/cached_datasets")
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
    if args.dataset == "humaneval":
        if not dataset:
            dataset = load_dataset("evalplus/humanevalplus", split="test")
            dataset.save_to_disk(cache_path)
        process_func = prepare_humaneval_sample
        run_test_func = run_humaneval
        prefix = ""

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")




    dataset = dataset.shuffle(seed=42)
    num_samples = min(len(dataset), args.num_samples) if args.num_samples > 0 else len(dataset)
    dataset = (
        dataset.select(range(num_samples))
    )
    # return a list of dicts with "input" and "setup" keys
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
                chunks[rank], #list of dicts with "input" and "setup" keys
                return_dict,
                args.config,
                args.name_architecture,
                run_test_func,
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
        "plan_template": PLAN_PROMPT,
        "answer_template": CODER_PROMPT_AFTERPLAN if "only" not in args.name_architecture else CODER_PROMPT_SINGLE,
        "diffusion_max_new_tokens": yaml_config["diffusion"]["max_new_tokens"],
        "llm_max_new_tokens": yaml_config["llm"]["max_new_tokens"],
        "results": all_results,
    }
    print(f"Accuracy: {accuracy:.2f}")

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{yaml_config['runtime']['output_dir']}/code/{args.name_architecture}/{args.dataset}_evaluation_{time_stamp}_{args.name_architecture}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
