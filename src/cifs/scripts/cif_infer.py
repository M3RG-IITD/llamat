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
from tqdm import tqdm

valfile = []
with open("/scratch/cse/btech/cs1200448/MatLlama/ift_cif_large/val.jsonl", 'r') as f:
    valfile = [json.loads(line) for line in f.readlines()]

valinputs = [f"system- {i['system']} "+f"question- {i['input']} "+"output- " for i in valfile]

init_seed = 2
seed = 1
num_seeds = 3

kwargs = {
    "model": "/scratch/civil/phd/cez188393/zaki_epcc/checkpoints_llamat_cit/checkpoint_17000_to_hf",
    "tokenizer": "/scratch/civil/phd/cez188393/zaki_epcc/checkpoints_llamat_cit/checkpoint_17000_to_hf",
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

for _, pred in tqdm(enumerate(predictions)):
    with open(f'/home/cse/btech/cs1200448/MatLlama/cif_infer_outputs/{_}.txt', 'w') as f:
        f.write(pred)