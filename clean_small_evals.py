from glob import glob 
import os
import json
from tqdm import tqdm

files = glob("outputs/**/*.json")
MIN_SAMPLES = 10

for f in tqdm(files):
    try:
        with open(f, "r") as file:
            data = json.load(file)
    except json.JSONDecodeError:
        continue
    if isinstance(data, list): 
        os.remove(f) 
        continue
    num_samples = data.get("num_samples", 100)
    if num_samples < MIN_SAMPLES:
        print(f"Removing {f} with {num_samples} samples")
        os.remove(f)