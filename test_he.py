import json
from tqdm import tqdm
import evaluate as hf_evaluate
import os
from glob import glob

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
compute_ = hf_evaluate.load("code_eval", cache_dir="/l/users/abdulrahman.mahmoud")
root = "outputs/code"
files = glob(f"{root}/**/*.json")


def extract_code_between_backticks(s):
    first = s.find("```")
    if first != -1:
        start = first + 3
        if s[start:start+6].lower() == 'python':
            start += 6
        if start < len(s) and s[start] == '\n':
            start += 1
        second = s.find("```", start)
        if second != -1:
            return s[start:second].strip()
        else:
            return s[start:].strip()
    
    # If no code blocks, look for function definitions
    lines = s.split('\n')
    code_lines = []
    in_function = False
    
    for line in lines:
        if line.strip().startswith('def ') or line.strip().startswith('return '):
            in_function = True
        if in_function:
            code_lines.append(line)
        # Stop if we hit explanatory text after code
        if in_function and line.strip() and not line.startswith(' ') and not line.strip().startswith('def') and not line.strip().startswith('return') and not line.strip().startswith('#'):
            break
    
    if code_lines:
        return '\n'.join(code_lines).strip()
    
    return s.strip()
    
    
def run_humaneval(predicted: str, setup: dict) -> float:
    try:
        full_code = extract_code_between_backticks(predicted)
        test_cases = [setup["test"]]  # Ensure it's a list
        predictions = [[full_code]]
        results = compute_.compute(
            references=test_cases,
            predictions=predictions,
            k=[1]
        )
        return float(results[0]["pass@1"] > 0)
        
    except Exception as e:
        # print(f"Error in code evaluation: {e}")
        # import traceback
        # traceback.print_exc()  # Add this for debugging
        return 0.0

for path in files:
    with open(path, "r") as f:
        data = json.load(f)

    acc = 0
    for d in tqdm(data['results']):
        test_setup = d.get("Test Setup", None)
        if not test_setup:
            test_setup = d['ground_truth']
        acc += run_humaneval(d['predicted_answer'], test_setup)

        
    acc = round(acc*100 / len(data['results']), 2)
    print(f"Accuracy for {data['name_architecture']}: {acc}%")