from argparse import ArgumentParser
import sys
from tqdm import tqdm
import math
import json
from datetime import datetime

from datasets import load_dataset, Dataset
import torch
import torch.multiprocessing as mp
from pts.pipeline.orchestrator import PTSPipeline
from pts.pipeline.utils import read_yaml
from pts.eval.metrics.essay_metrics import (
    get_scores,
    D3,
    R4,
    LR_n,
)


PLAN_PROMPT = """Create a concise plan for an essay on the following topic:  
"{topic}"

The plan should be brief and structured, suitable for an essay of 200–250 words.  
Include:  
- Introduction (with clear thesis statement)  
- 2–3 body paragraphs (each with one main argument and short explanation)  
- Conclusion (summarizing position)  

Write the plan in bullet points, no long sentences."""

ESSAY_PROMPT = """
Write a well-structured essay of about 200-250 words on the following topic:
"{topic}"

Follow this exact plan when writing the essay:  
{plan}  


Requirements:  
- Formal academic tone  
- Clear introduction, body, and conclusion  
- Logical flow with transitions between paragraphs 
- Stay within 200–250 words
"""

def prepare_ielts_sample(sample):
    return {"topic": sample["prompt"]}

def evaluate_essay(essay: str) -> dict:
    scores = get_scores(essay)
    scores["D3"] = D3(essay)
    scores["R4"] = R4(essay)
    scores["LR2"] = LR_n(essay, n=2)
    scores["LR3"] = LR_n(essay, n=3)
    return scores

def worker_evaluate(
    rank: int,
    samples,
    return_dict,
    configs,
    name_architecture,
    eval_func,
):
    torch.cuda.set_device(rank)
    pipeline = PTSPipeline.from_yaml(configs)
    local_results = {}
    plans = []
    for i, sample in enumerate(tqdm(samples, desc=f"[GPU {rank}] Diff")):
        topic = sample["topic"]
        plan_input = PLAN_PROMPT.format(topic=topic)
        plan = pipeline.generate_plan(plan_input, name_architecture=name_architecture)
        plan["topic"] = topic
        plan["plan"] = plan['text']
        plans.append(plan)
        print(plan["text"])

    local_results = []
    for plan in tqdm(plans, desc="Running LLM refinement"):
        topic = plan["topic"]
        answer_input = ESSAY_PROMPT.format(topic=topic, plan=plan["text"])
        out = pipeline.generate_answer(answer_input, name_architecture=name_architecture)
        print(out["text"])
        metrics = eval_func(out["text"])
        local_results.append(
            {
                **metrics,
                "topic": topic,
                "predicted_plan": plan["text"],
                "predicted_answer": out["text"],
            }
        )
    return_dict[rank] = local_results


def parse_args():
    parser = ArgumentParser(
        description="Run evaluation on a dataset using the PTS pipeline"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default="ielts-essays")
    parser.add_argument("--llm_eval", action="store_true")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--name_architecture", type=str, default="llm-only")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(f"Loading dataset {args.dataset}")
    if args.dataset == "ielts-essays":
        dataset = load_dataset("chillies/IELTS-writing-task-2-evaluation", split="test")
        # make the col "prompt" unique, there are many duplicates
        dataset = Dataset.from_dict({"prompt": dataset.unique("prompt")})
        process_func = prepare_ielts_sample
        eval_func = evaluate_essay
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
    # TODO: add llm/diffusion only evaluation
    worker_func = worker_evaluate

    for rank in range(world_size):
        p = mp.Process(
            target=worker_func,
            args=(rank, chunks[rank], return_dict, args.config, args.name_architecture, eval_func),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    all_results = []
    for rank in range(world_size):
        all_results.extend(return_dict[rank])

    yaml_config = read_yaml(args.config)
    metric_keys = ["Task Achievement", "Coherence and Cohesion", "Vocabulary", "Grammar", "Overall", "D3", "R4", "LR2", "LR3"]
    def acc(key):
        return sum(r[key] for r in all_results) / len(all_results)
    r = {key: acc(key) for key in metric_keys}
    
    all_results = {
        "diffusion_model": yaml_config["diffusion"]["model_id"],
        "llm": yaml_config["llm"]["model_id"],
        "name_architecture" : args.name_architecture,
        "dataset": args.dataset,
        **r,
        "num_samples": len(all_samples),
        "is_llm": args.llm_eval,
        "plan_template": PLAN_PROMPT,
        "answer_template": ESSAY_PROMPT,
        "diffusion_max_new_tokens": yaml_config["diffusion"]["max_new_tokens"],
        "llm_max_new_tokens": yaml_config["llm"]["max_new_tokens"],
        "results": all_results,
    }
    for k, v in r.items():
        print(f"{k}: {v}")
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{yaml_config['runtime']['output_dir']}/{args.name_architecture}/{args.dataset}_evaluation_{time_stamp}_{args.name_architecture}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)
    return 0

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
