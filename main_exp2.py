import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer
import yaml
from tqdm import tqdm
from speculative import speculative_sampling
from auto_regressive import auto_regressive
import json


# opening config files
with open("Paper-Reproduction-Speculative-Decoding/config.yaml", "r") as f:
    config = yaml.safe_load(f)


device = "cuda" if torch.cuda.is_available() else "cpu"


# loading dataset
humanEval = load_dataset("openai/openai_humaneval",trust_remote_code=True)["test"]
humanEval=humanEval.select(range(100))


# setting up tokenizer
tokenizer = AutoTokenizer.from_pretrained(config['models']['tokenizer'])



results = []

for sample in tqdm(humanEval):
    

    prompt=sample['prompt']
    
    target_ids,target_sp=auto_regressive(prompt)
    speculative_ids,acc_rate,speculative_sp=speculative_sampling(prompt)



    original_code=tokenizer.decode(target_ids[0], skip_special_tokens=True)
    speculative_code=tokenizer.decode(speculative_ids[0], skip_special_tokens=True)



    results.append({

    "ids":sample['task_id'],
    "speculative_code": speculative_code,
    "original_code": original_code,
    "acceptance_ratio": acc_rate,
    "inference_speed": target_sp,
    "speculative_speed": speculative_sp

    })





with open("Paper-Reproduction-Speculative-Decoding\\Results\\results1.jsonl", "a", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

