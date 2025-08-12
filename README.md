
# Speculative Decoding

Accelerating Large language model inferencing with Speculative Sampling.

This method is based on the fact that some tokens are too obivious to predict and hence doesn't required to be generated from Large Target model, instead we can draw few drafts from a smaller Draft model and give scores to them by Target model in one pass, whether to accept or reject each draft.

Here, to reproduce the results from the original paper from deepmind https://arxiv.org/pdf/2302.01318 .



## Experiment 01 
GPT neo family models fine tuned on xsum, on local RTX3050 6GB

-   Target Model: GPT neo 1.3B

    https://huggingface.co/mia-llm/gpt-neo-1.3B-xsum-roya

-   Draft Model:GPT neo 125M

    https://huggingface.co/mia-llm/gpt-neo-125m-xsum-roya

Results:

Evaluated on 10 xsum samples
| Method | Sampling |Rouge-1 Score| Speedup|Acceptance rate
|----------|-----|-----|----------|---|
| Auto-regressive|Greedy   | 0.160    | 1x|NIL|
| Speculative|Greedy   | 0.159    | 1.84x    |32%|

The draft model being significantly smaller than the target model led to poor acceptance rate but it still boosts inference speeed on the cost of very little rouge1 score drop.



## Experiment 02 
Base Deepseek coder ai, on kaggle T4 15GB 

-   Target Model: Deepseek code 6.7B base

    https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base

-   Draft Model:Deepseek code 1.3B base

    https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base

Results:

Evaluated on 100 open-ai HumanEval samples
| Method | Sampling |pass@1| Speedup|Acceptance rate
|----------|-----|-----|----------|---|
| Auto-regressive|Greedy   | 35%   | 1x|NIL|
| Speculative|Greedy   | 40%    | 2.52x    |75%|

75% of the tokens from the draft model are accepted which is good enough and therefore obsesrving a speedup of 2.52x while also increasing the pass@1 score.



# About the Repository 

```text
Paper-Reproduction-Speculative-Decoding/
├── main_exp1.py
├── main_exp2.py
├── speculative.py
├── autoregressive.py
├── config.yaml
|      
├── Results/
|    ├── result1.jsonl
|    └── rsults2.jsonl
|
└── Benchmarks/
        ├── bench1.ipynb
```     └── bench2.ipynb

Let's say to do Experiment 02, first modify the config.yaml files accordingly and then run main_exp2.py, a jsonl file will be saved, now to get the benchmarks run bench2.ipynb.







