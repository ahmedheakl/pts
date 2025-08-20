import argparse
import sys
from .pipeline.orchestrator import PTSPipeline


def main():
    ap = argparse.ArgumentParser(description="LLada → Llama pipeline")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--no-refine", action="store_true", help="Skip the stage-1 Llama refinement")
    ap.add_argument("--number_samples", type=int, default=10, help="Number of samples to generate")
    ap.add_argument("--name_architechture", type=str, default="diffusion-llm", help="Name of the architecture to use")
    args = ap.parse_args()

    pipe = PTSPipeline.from_yaml(args.config)
    from datasets import load_dataset
    ds = load_dataset("chillies/IELTS-writing-task-2-evaluation")
    unique_prompts = list(set([d["prompt"] for d in ds["train"]]))

    import os
    import json
    from datetime import datetime

    results = []
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs_essay/{args.name_architechture}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"essay_evaluation_{time_stamp}.json")

    for topic in unique_prompts[:args.number_samples]:
        plan_prompt = f"Write a plan for writing an essay about {topic} "
        essay_prompt = f"Write an essay about {topic} following this plan"

        plan = pipe.generate_plan(user_prompt=plan_prompt)
        answer = pipe.generate_answer(user_prompt=essay_prompt + '\n' + plan['text'])
        result = {
            "topic": topic,
            "plan": plan["text"],
            "name architecture": args.name_architechture,
            "plan_metadata": plan.get("metadata"),
            "essay": answer["text"],
            "essay_metadata": answer.get("metadata"),
        }
        results.append(result)
        print(f"Topic: {topic}\nPlan: {plan['text']}\nEssay: {answer['text']}\n{'-'*40}")

        # Save results after each topic
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    sys.exit(main())
