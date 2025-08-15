User Prompt
   │
   ▼
[Diffusion Model]  -- plan/structure/thoughts -->  plan_text
   │
   ▼
[LLM] (stage 1, optional)  -- refined_text -->
   │
   ├─ concatenate(plan_text, refined_text, extra_context)
   ▼
[LLM] (stage 2, final)  -- final_output -->
   │
   ▼
Save JSON {prompt, plan_text, refined_text, combined_input, final_output, meta}



# Diffusion + LLM Pipeline

This repository implements a simple but extensible pipeline where:

1. A **diffusion text model** generates a structured "plan" from a user prompt.
2. An **LLM** optionally refines that plan.
3. The plan, refined version, and optional extra context are concatenated.
4. The concatenated text is fed back into the LLM for the final answer.
5. The results (including metadata) are saved in a **JSON** file.

---


# Configuration to edit
Edit configs/default.yaml to set:

* diffusion : model ID, device, max tokens

* llm*: model ID, generation parameters

* Runtime: output folder, JSON prefix, random seed

* Prompting: headers and system messages

Example:

```yaml
diffusion:
  impl: "local"
  model_id: "LLada-Text-Diffuser-1.0"
  device: "cuda:0"
  max_new_tokens: 256

llm:
  impl: "hf"
  model_id: "meta-llama/Llama-3.1-8B-Instruct"
  device: "cuda:0"
  max_new_tokens: 512
  temperature: 0.2
  do_sample: false

runtime:
  output_dir: "outputs"
  json_prefix: "run"
  seed: 42

prompting:
  final_system_msg: "You are a precise assistant. Produce a clear, concise answer."
  concat_header_plan: "## PLAN"
  concat_header_refined: "## REFINED"
  concat_header_extra: "## CONTEXT" 
```

# Usage

Run from the command line:
```yaml
python -m src.cli \
  --config configs/default.yaml \
  --prompt "Write a plan for writing an essay about the pros and cons of artificial intelligence in education. " \
  --extra "Write an essay following this plan" \
  --no-refine
```
--no-refine means that you do not want an extra step of LLM that would rewrite the plan that outputed the diffusion. Remove this if you want to allow this extra step.

# Setup 1/ Setup2
Modify concatenate.py to change how intermediate outputs are stitched.



## 📂 Repository Structure

diffusion-llm-pipeline/
├── README.md
├── requirements.txt
├── configs/
│ └── default.yaml
└── src/
├── cli.py # Command-line entry point
├── pipeline/
│ ├── init.py
│ ├── config.py # Pydantic-based config schema
│ ├── concatenate.py # Combines plan + refined + context
│ ├── io_utils.py # Saving/loading helpers
│ ├── orchestrator.py # Pipeline coordinator
│ └── models/
│ ├── diffusion.py # Diffusion model adapter
│ └── llm_.py # LLM adapter
└── tests/
└── test_smoke.py

