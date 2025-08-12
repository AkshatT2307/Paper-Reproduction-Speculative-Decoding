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
xsum = load_dataset("xsum", split=f"test[:{config['dataset']['size']}]",trust_remote_code=True)



# setting up tokenizer
tokenizer = AutoTokenizer.from_pretrained(config['models']['tokenizer'])



results = []

for sample in tqdm(xsum):
    

    prompt="Generate a short Summary for this article: "+sample['document']
    n=len(prompt)

    target_ids,target_sp=auto_regressive(prompt)
    speculative_ids,acc_rate,speculative_sp=speculative_sampling(prompt)



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





with open("Paper-Reproduction-Speculative-Decoding\\Results\\results1.jsonl", "a", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

