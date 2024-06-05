import json
import numpy as np
import pandas as pd
import vllm
import torch
import string
import re
import collections
from sklearn.metrics import f1_score
import Levenshtein
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaTokenizer
import datetime
from functools import reduce
from huggingface_hub import login

valfile = []
with open("/scratch/cse/btech/cs1200448/MatLlama/ift_cif/val.jsonl", 'r') as f:
    valfile = [json.loads(line) for line in f.readlines()]

valinputs = [f"<|im_start|>system\n{i['system']}<|im_end|>\n"+f"<|im_start|>question\n{i['input']}<|im_end|>\n"+"<|im_start|>answer\n" for i in valfile]

init_seed = 2
seed = 1
num_seeds = 3

kwargs = {
    "model": "/scratch/cse/btech/cs1200448/MatLlama/ft-checkpoints/ift_cif_hf",
    "tokenizer": "/scratch/cse/btech/cs1200448/llama-weights/7b/tokenizer.model",
    "trust_remote_code": True,
    "tensor_parallel_size": 1,
    "seed":seed,
    # "gpu_memory_utilization":args.mem_util,
}

client = vllm.LLM(**kwargs)


# for _ in range(1, num_seeds+1):
seed = seed * init_seed

response = client.generate(valinputs, sampling_params=vllm.SamplingParams(
        skip_special_tokens=True,
        best_of=1,
        presence_penalty=0.0,
        frequency_penalty=1.0,
        top_k=50,
        top_p=1.0,
        temperature=0.75,
        stop=["<|im_start|>", "<|im_end|>"],
        use_beam_search=False,
        max_tokens=300,
        logprobs=2
    ))
predictions = [i.outputs[0].text for i in response]

def most_similar_answer(a,answer_set):
    a = a.strip().replace(' ', '')
    if(a in answer_set):
        return a
    dis = [Levenshtein.distance(a,x) for x in answer_set]
    idx = np.argmin(dis)
    return answer_set[idx]

for _, sample in enumerate(valfile):
    task = sample['task']
    system = sample['system']
    output = predictions[_]
    print(sample['input'])
    print(sample['output'])
    print("--------")
    print(output)
    break