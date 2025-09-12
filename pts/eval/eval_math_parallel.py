from argparse import ArgumentParser
import sys
from tqdm import tqdm
import re
import math
import json
from datetime import datetime
from functools import partial
import os

from datasets import load_dataset, load_from_disk
import torch
import torch.multiprocessing as mp
from mathruler.grader import extract_boxed_content, grade_answer
from sympy.parsing.latex import parse_latex

from pts.pipeline.orchestrator import PTSPipeline
from pts.pipeline.utils import read_yaml
from pts.constants import Pipelines
#from pts.eval.compare_prepare.aime import compare_answers_aime, prepare_aime_sample
from pts.eval.compare_prepare.truthfulQA import compare_answers_truthqa, prepare_truthfulqa_sample, speaker_only_template_tqa, plan_template_tqa, speaker_after_plan_template_tqa




def worker_evaluate_single(
    rank: int,
    samples,
    return_dict,
    configs,
    name_architecture,
    compare_func,
    speaker_only_template,
    plan_template = None,
    speaker_after_plan_template = None,
):
    torch.cuda.set_device(rank)
    pipeline = PTSPipeline.from_yaml(configs)
    local_results = []
    for i, sample in enumerate(tqdm(samples, desc=f"[GPU {rank}] Answer")):
        user_prompt = sample["input"]
        llm_input = speaker_only_template.format(question=user_prompt)
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
    compare_func,
    plan_template,
    speaker_after_plan_template,
    speaker_only_template =None,
):
    torch.cuda.set_device(rank)
    pipeline = PTSPipeline.from_yaml(configs)
    local_results = {}
    plans = []
    for i, sample in enumerate(tqdm(samples, desc=f"[GPU {rank}] Plan")):
        user_prompt = sample["input"]
        diffusion_input = plan_template.format(question=user_prompt)
        plan = pipeline.generate_plan(diffusion_input, name_architecture)
        plan["question"] = user_prompt
        plan["correct"] = sample["correct"]
        plans.append(plan)
        print(plan["text"])

    local_results = []
    for plan in tqdm(plans, desc=f"[GPU {rank}] Answer"):
        question = plan["question"]
        llm_input = speaker_after_plan_template.format(question=question, plan=plan["text"]) 
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
    if "truthfulqa" in args.dataset:
        if not dataset:
            dataset = load_dataset("domenicrosati/TruthfulQA", split="train")
            dataset.save_to_disk(cache_path)
        process_func = prepare_truthfulqa_sample
        compare_func = compare_answers_truthqa
        speaker_only_template = speaker_only_template_tqa
        plan_template = plan_template_tqa   
        speaker_after_plan_template = speaker_after_plan_template_tqa
        
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
                speaker_only_template,
                plan_template,
                speaker_after_plan_template,
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
        "plan_template": plan_template ,
        "answer_template": speaker_only_template,
        "diffusion_max_new_tokens": yaml_config["diffusion"]["max_new_tokens"],
        "llm_max_new_tokens": yaml_config["llm"]["max_new_tokens"],
        "results": all_results,
    }
    print(f"Accuracy: {accuracy:.2f}")

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{yaml_config['runtime']['output_dir']}/math/{args.name_architecture}/{args.dataset}_evaluation_{time_stamp}_{args.name_architecture}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
