from dataclasses import dataclass
from typing import List, Dict, Optional
import yaml
import random
import torch

from .config import AppCfg
from .models.llm import LLM
from .models.diffusion import Diffusion
from .concatenate import stitch
from .utils import timestamp

@dataclass
class PTSPipeline:
    cfg: AppCfg
    llm: LLM
    diffusion: Diffusion

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        cfg = AppCfg(**raw)
        # seeds
        random.seed(cfg.runtime.seed)
        torch.manual_seed(cfg.runtime.seed)
        # models
        diffusion = Diffusion(**cfg.diffusion.model_dump())
        llm = LLM(**cfg.llm.model_dump())
        return cls(cfg=cfg, diffusion=diffusion, llm=llm)
    
    def generate_plan(self, user_prompt: str) -> Dict:
        return self.diffusion.generate(user_prompt)
    
    def generate_answer(self, user_prompt: str):
        return self.llm.generate(
            system_prompt=self.cfg.prompting.final_system_msg,
            user_prompt=user_prompt
        )

    def run(self, user_prompt: str, extra_text: Optional[List[str]] = None,
            refine_with_llm: bool = False) -> Dict:
        # 1) diffusion produces plan/structure
        plan = self.diffusion.generate(user_prompt)
        plan_text = plan["text"]

        # 2) (Optional) First-pass llm to refine the plan
        refined_text = ""
        if refine_with_llm:
            refined = self.llm.generate(
                system_prompt="You rewrite plans into crisp bullet points.",
                user_prompt=f"Rewrite the following plan more clearly:\n\n{plan_text}"
            )
            refined_text = refined["text"]

        # 3) Concatenate inputs for final llm call
        combined = stitch(
            plan=plan_text,
            refined=refined_text,
            extras=extra_text or [],
            headers=self.cfg.prompting
        )

        final = self.llm.generate(
            system_prompt=self.cfg.prompting.final_system_msg,
            user_prompt=f"{combined}"
        )

        # 4) JSON result
        result = {
            "timestamp": timestamp(),
            "input": {
                "prompt": user_prompt,
                "extra_text": extra_text or []
            },
            "diffusion_output": {
                "text": plan_text,
                "metadata": plan["metadata"]
            },
            "llm_stage1_output": {
                "text": refined_text,
                "metadata": None if not refine_with_llm else {
                    "model_id": self.llm.model_id,
                    "max_new_tokens": self.llm.max_new_tokens
                }
            },
            "combined_input": combined,
            "final_output": {
                "text": final["text"],
                "metadata": final["metadata"]
            },
            "models": {
                "diffusion": plan["metadata"],
                "llm": final["metadata"]
            }
        }
        # path = save_json(result, self.cfg.runtime.output_dir, self.cfg.runtime.json_prefix)
        # result["saved_path"] = path
        return result
