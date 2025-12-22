
#NOTE: USAGE = 
# python3 discomat_run.py <CUDA_GPU_NUMBER> <MODEL_PATH> <SAVE_NAME_PREFIX>
# Output will be stored in the same folder as <SAVE_NAME_PREFIX>_doping_test.pkl
# %%
import os
import json
import pandas as pd
import random
import sys
from warnings import filterwarnings
from tqdm import tqdm, trange
import torch
import math

if(len(sys.argv) < 4):
    print("Usage: python doping_llamat3.py <gpu_id> <model_name> <prefix_savename>")
    sys.exit(1)

os.environ['CUDA_VISIBLE_DEVICES']=str(sys.argv[1]); 
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
path = sys.argv[2];
model_dir = path
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# %%
model.to("cuda")

# %%
import random
import json
def load_jsonl(path):
    with open(path, 'r') as f:
        a = f.readlines()
        g = [json.loads(i) for i in a]
    return g

# %%
doping_train = load_jsonl('doping_train.jsonl')
doping_test = load_jsonl('doping_test.jsonl')
type(doping_train[0]), doping_train[0].keys()
len(doping_train), len(doping_test)
doping_test[0]

from tqdm import tqdm

# %%
out_tokens = []
for sample in tqdm(doping_test):
    eval_out = str(sample['answer'])
    model_output = tokenizer(eval_out, return_tensors="pt")#.to("cuda")
    out_tokens.append(model_output['input_ids'].shape[1])

idx = 224

sample = doping_test[idx]

# %%

eval_prompt = f"{sample['system']} "+f"question-{sample['question']}" + "output-"
eval_prompt = ''
for role in ['system','question']:
    eval_prompt += f"<|im_start|>{role}\n{sample[role]}<|im_end|>\n"

eval_prompt += f"<|im_start|>answer\n"

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

# %%

generation_config = {
  "bos_token_id": 128010,
  "do_sample": True,
  "eos_token_id": 128011,
}

import math

def get_gen_len(n):
    if n%50==0: n+=1
    else:pass
    return min(math.ceil(n / 100) * 100, 2048)
outpad = [71,71,60,70,70, 70, 50, 70,   70, 60,  60,  60, 60,  60, 60]
model.eval()
from tqdm import tqdm
from warnings import  filterwarnings
filterwarnings("ignore")

outputcs = []

start = 0

original_schema='''The answer should be in the following schema:
{
"basemats": {
 "h0": <host 0>,
 "h1": <host 1>
},
"dopants": {
 "d0": <dopant 0>
},
"dopants2basemats": {
 "<dopant key>": [
  "<basemat key>"
 ], 
}
}.'''
replaced_schema = 'The answer should be in the following schema:\n{\n"basemats": {\n "b0": <basemat 0>,\n "b1": <basemat 1>\n},\n"dopants": {\n "d0": <dopant 0>\n},\n"dopants2basemats": {\n "<dopant key>": [\n  "<basemat key>"\n ], \n}\n}.'
def prepare_prompt(sample):
    if '/7b-chat' in path:
        return prepare_prompt_llama2(sample)
    if '/8b-chat' in path:
        return prepare_prompt_llama3(sample)
    eval_prompt = ''
    sample['question'] = sample['question'].replace(original_schema, replaced_schema) #to get the schema output as b0, b1 (basemats) instead of host
    for role in ['system','question']:
        eval_prompt += f"<|im_start|>{role}\n{sample[role]}<|im_end|>\n"
    eval_prompt += f"<|im_start|>answer\n"
    return eval_prompt; 
  
def prepare_prompt_llama2(sample):
    eval_prompt = '<s>[INST]<<SYS>>\n' + sample['system'] + '<</SYS>>\n' + sample['question'] + '\n[/INST]' 
    return eval_prompt
def prepare_prompt_llama3(sample):
    eval_prompt = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n' + sample['system'] + '<|eot_id|><|start_header_id|>user<|end_header_id|>\n' + sample['question'] + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n'
    return eval_prompt

for idx, sample in tqdm(enumerate(doping_test[:])):
    eval_prompt = prepare_prompt(sample)
    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    if model_input['input_ids'].shape[1]>=2048:
        print('skipping because input is greater than 2048')
        outputcs.append(2048)
    
    else:
        output = model.generate(**model_input, temperature = 0.05, top_p=0.5, max_length= min(int(model_input['input_ids'].shape[1]+ out_tokens[idx]+350),2048),do_sample=False) # for atom count and atom name
        generated_tokens = output[0, model_input['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        outputcs.append(generated_text)
        
import pickle
if len(sys.argv) > 3:
    savepath = sys.argv[3] + '_doping_test.pkl'
    print("*"*20, "\nsavepath is ", savepath)
else:
    savepath = sys.argv[2].split('/')[-3] + '_doping_test.pkl';
    if('cs12' in savepath or len(savepath) <= 15):
        x = sys.argv[2].split('/')
        savepath = 'llama_basic_' + x[-1] + '_doping_test.pkl';
    print("******************Savepath name updated to", savepath);
with open(savepath,'wb') as f:
    pickle.dump(outputcs, f)
f.close()
print("saved to ", savepath);
