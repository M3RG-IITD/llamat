

#NOTE: USAGE = 
# python3 discomat_run.py <CUDA_GPU_NUMBER> <MODEL_PATH> <SAVE_NAME_PREFIX>
# Output will be stored in the same folder as <SAVE_NAME_PREFIX>_mof2_test.pkl

import os
import json
import pandas as pd
import random
import sys;
os.environ['CUDA_VISIBLE_DEVICES']=str(sys.argv[1]);

# %%
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

mof1_train = load_jsonl('mof2_train.jsonl')
mof1_test = load_jsonl('mof2_test.jsonl')

# %%
type(mof1_train[0]), mof1_train[0].keys()

# %%
len(mof1_train), len(mof1_test)

# %%
# %%
from tqdm import tqdm

# %%
out_tokens = []
for sample in tqdm(mof1_test):
    eval_out = str(sample['answer'])
    model_output = tokenizer(eval_out, return_tensors="pt")#.to("cuda")
    out_tokens.append(model_output['input_ids'].shape[1])

# %%


# %%
# out_tokens#[11]

# %%
out_tokens.index(max(out_tokens))

# %%
idx = 1
sample = mof1_test[idx]

# %%

eval_prompt = f"{sample['system']} "+f"question-{sample['question']}" + "output-"
eval_prompt = ''
for role in ['system','question']:
    eval_prompt += f"<|im_start|>{role}\n{sample[role]}<|im_end|>\n"

eval_prompt += f"<|im_start|>answer\n"

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")


# %%
from tqdm import tqdm, trange

# %%
# out_tokens

# %%
import torch

# %%
# dfch[~mask].head(20)#.value_counts()

# %%
# help(model.generate)

# %%
generation_config = {
  "bos_token_id": 128010,
  "do_sample": True,
  "eos_token_id": 128011,
  # "max_length": 2048,
  # "temperature": 0.6,
  # "top_p": 0.9,
  # "transformers_version": "4.41.0"
}


# %%
# [sample['answer'] for sample in valfile]

# %%
import math

def get_gen_len(n):
    if n%50==0: n+=1
    else:pass
    return min(math.ceil(n / 100) * 100, 2048)

# %%
min(int(model_input['input_ids'].shape[1]+ get_gen_len(len(sample['answer']))),2048)

# %%
sample['answer']

# %%
model.eval()
with torch.no_grad():

    # for min(int(model_input['input_ids'].shape[1]+ get_gen_len(len(sample['answer']))),2048)
    print(tokenizer.decode(model.generate(**model_input, top_p=0.95, max_length= min(int(model_input['input_ids'].shape[1] + out_tokens[idx]+400),2048),do_sample=True)[0], skip_special_tokens=True))

# %%
import ast
(ast.literal_eval(sample['answer']))

# %%


# %%
from tqdm import tqdm

# %%
from warnings import  filterwarnings
filterwarnings("ignore")

# %%
outputcs = []
def prepare_prompt(sample):
    if '/7b-chat' in path:
        return prepare_prompt_llama2(sample)
    if '/8b-chat' in path:
        return prepare_prompt_llama3(sample)
    eval_prompt = ''
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
# start = 0
# finish = 1100 + 165 +165
for sample in tqdm(mof1_test[:]):
    
    # eval_prompt = f"{sample['system']} "+f"input-{sample['input']}" + "output-"
    eval_prompt = prepare_prompt(sample)

    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    if model_input['input_ids'].shape[1]>=2048:
        print('skipping because input is greater than 2048')
        outputcs.append(2048)
    
    else:
        output = model.generate(**model_input, temperature = 0.05, top_p=0.95, max_length= min(int(model_input['input_ids'].shape[1]+ out_tokens[idx]+400),2048),do_sample=False) # for atom count and atom name
    
        generated_tokens = output[0, model_input['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
        outputcs.append(generated_text)

# %%

# %%
import ast
import pickle


if len(sys.argv) > 3:
    savepath = sys.argv[3] + '_mof2_test.pkl'
    print("*"*20, "\nsavepath is ", savepath)
else:
    savepath = sys.argv[2].split('/')[-3] + '_mof2_test.pkl';
    if('cs12' in savepath or len(savepath) <= 15):
        x = sys.argv[2].split('/')
        savepath = 'llama_basic_' + x[-1] + '_mof2_test.pkl';
    print("******************Savepath name updated to", savepath);

#savepath = path.split('/')[-3] + "_mof2_test.pkl";
with open(savepath,'wb') as f:
    pickle.dump(outputcs, f)
f.close()
print("saved output to ", savepath);

# %%
def check(gold_entry, test_entry):
    gold_entry = sorted(gold_entry, key=lambda x: x.get('formula', ''))
    test_entry = sorted(test_entry, key=lambda x: x.get('formula', ''))

    # print(gold_entry)
    ### order each dictionary by keys
    gold_entry = [dict(sorted(d.items())) for d in gold_entry]
    test_entry = [dict(sorted(d.items())) for d in test_entry]
    print(gold_entry)
    ### compare strings
    return str(gold_entry) == str(test_entry)

# %%
# jws = jellyfish.jaro_winkler_similarity(str(gold), str(prediction), long_tolerance=True)
# jaro_winkler_similarities.append(jws)
