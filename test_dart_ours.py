import json
from tqdm import tqdm
from sympy.parsing.latex import parse_latex
from mathruler.grader import extract_boxed_content, grade_answer
from glob import glob

def compare_answers_dart(pred, gt):
    if '\\boxed{' not in pred:
        pred = f"\\boxed{{{pred}}}"
    if '\\boxed{' not in gt:
        gt = f"\\boxed{{{gt}}}"
    pred_answer = extract_boxed_content(pred.strip())
    gt_answer = extract_boxed_content(gt.strip())
    try:
        x = parse_latex(pred_answer)
        y = parse_latex(gt_answer)
    except:
        return float(grade_answer(pred_answer, gt_answer))
    return float(x.equals(y) or grade_answer(pred_answer, gt_answer))
    
files = glob("outputs/dl_dual/dart-*.json")
files = sorted(files)
for path in files:
    with open(path, "r") as f:
        data = json.load(f) 
    if "llama3b-pixart-4bs-2grad-35klatents" in data['llm']: 
        continue
    acc = 0
    for idx, d in enumerate(tqdm(data['results'])):
        
        score = compare_answers_dart(d['predicted_answer'], d['ground_truth'])
        acc += score
        data['results'][idx]['is_correct'] = score

    # with open(path, "w") as f:
    #     json.dump(data, f, indent=4)

    acc = round(acc*100 / len(data['results']), 2)
    print(data['dataset'], "-->", acc)
