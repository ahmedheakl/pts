import argparse
import sys
from .pipeline.orchestrator import Pipeline


def main():
    ap = argparse.ArgumentParser(description="LLada → Llama pipeline")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--prompt", required=True, help="User prompt to start with")
    ap.add_argument("--extra", nargs="*", default=[], help="Extra context strings to concatenate")
    ap.add_argument("--no-refine", action="store_true", help="Skip the stage-1 Llama refinement")
    args = ap.parse_args()

    pipe = Pipeline.from_yaml(args.config)
    out = pipe.run(user_prompt=args.prompt, extra_text=args.extra, refine_with_llm=not args.no_refine)
    print(f"Saved: {out['saved_path']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
