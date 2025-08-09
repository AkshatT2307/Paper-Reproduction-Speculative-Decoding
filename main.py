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
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)



# loading dataset
xsum = load_dataset("xsum", split=f"test[:{config['dataset']['size']}]",trust_remote_code=True)



# setting up tokenizer
tokenizer = AutoTokenizer.from_pretrained(config['models']['tokenizer'])


# original_summary=[]
# speculative_summary=[]
# acceptance_rate=0
# target_speed=0
# speculative_speed=0
results = []

for sample in tqdm(xsum):
    

    prompt="Generate a short Summary for this article: "+sample['document']
    n=len(prompt)

    target_ids,target_sp=auto_regressive(prompt)
    speculative_ids,acc_rate,speculative_sp=speculative_sampling(prompt)


    # acceptance_rate+=acc_rate
    # target_speed+=target_sp
    # speculative_speed+=speculative_sp


    original_summary=tokenizer.decode(target_ids[0], skip_special_tokens=True)[n:]
    speculative_summary=tokenizer.decode(speculative_ids[0], skip_special_tokens=True)[n:]

    results.append({
    "prompt": sample,
    "reference": sample['summary'],
    "speculative_summary": speculative_summary,
    "original_summary": original_summary,
    "acceptance_ratio": acc_rate,
    "inference_speed": target_sp,
    "speculative_speed": speculative_sp
    })



# target_speed/=config['dataset']['size']
# speculative_speed/=config['dataset']['size']
# acceptance_rate/=config['dataset']['size']

# print(target_speed,speculative_speed,acceptance_rate)


with open("results.jsonl", "a", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

