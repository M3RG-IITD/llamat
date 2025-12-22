
#NOTE: USAGE = 
# python3 discomat_run.py <CUDA_GPU_NUMBER> <MODEL_PATH> <SAVE_NAME_PREFIX>
# Output will be stored in the same folder as <SAVE_NAME_PREFIX>_discomat_test.pkl


import os
import pickle
import sys
import json
import pandas as pd
import random

# %%
os.environ['CUDA_VISIBLE_DEVICES']=str(sys.argv[1]);
# %%
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
path = sys.argv[2];
model_dir = sys.argv[2];
give_examples = 0;

if('discomat_ex' in path[-3]):
    give_examples = 1; #will give examples in this case.
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

model.to("cuda")
#discomat = json.load(open('../../datasets/ift/raw_ift/ift_discomat.json', 'r'))

def convert_dsm_to_text(inp):
    output = f"Caption: {inp['caption']}\n\nTable: {inp['act_table']}\n\nFooter: {inp['footer']}"
    return output

with open("discomat_train.jsonl", 'r') as f:
    discomat_train = [json.loads(line) for line in f.readlines()]
with open("discomat_test.jsonl", 'r') as f:
    discomat_test = [json.loads(line) for line in f.readlines()]
with open("discomat_test_valfile.jsonl", 'r') as f:
    discomat_val = [json.loads(line) for line in f.readlines()]

train_system = [data['system'] for data in discomat_train]
train_question = [data['question'] for data in discomat_train]
train_answer = [data['answer'] for data in discomat_train]

df_train = pd.DataFrame(train_system, columns=['system'])
df_train['question'] = train_question
df_train['answer'] = train_answer

test_system = [data['system'] for data in discomat_test]
test_question = [data['question'] for data in discomat_test]
test_answer = [data['answer'] for data in discomat_test]

df_test = pd.DataFrame(test_system, columns=['system'])
df_test['question'] = test_question
df_test['answer'] = test_answer

val_system = [data['system'] for data in discomat_val]
val_question = [data['question'] for data in discomat_val]
val_answer = [data['answer'] for data in discomat_val]

df_val = pd.DataFrame(val_system, columns=['system'])
df_val['question'] = val_question
df_val['answer'] = val_answer

valfile = []
with open("discomat_test_valfile.jsonl", 'r') as f:
    valfile = [json.loads(line) for line in f.readlines()]

from tqdm import tqdm

out_tokens = []
in_tokens = []
for sample in tqdm(valfile):
    eval_out = str(sample['answer'])
    model_output = tokenizer(eval_out, return_tensors="pt")#.to("cuda")
    out_tokens.append(model_output['input_ids'].shape[1])
    eval_in = str(sample['question'])
    model_output = tokenizer(eval_in, return_tensors="pt")#.to("cuda")
    in_tokens.append(model_output['input_ids'].shape[1])

out_tokens.index(max(out_tokens)), in_tokens.index(max(in_tokens))


eval_prompt = ''
for role in ['system','question']:
    eval_prompt += f"<|im_start|>{role}\n{sample[role]}<|im_end|>\n"

eval_prompt += f"<|im_start|>answer\n"

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")


# %%
from tqdm import tqdm, trange


import torch

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

import math

def get_gen_len(n):
    if n%50==0: n+=1
    else:pass
    return min(math.ceil(n / 100) * 100, 2048)

# %%
min(int(model_input['input_ids'].shape[1]+ get_gen_len(len(sample['answer']))),2048)

# %%
import matplotlib.pyplot as plt
# %%
model.eval()
from tqdm import tqdm

from warnings import  filterwarnings
filterwarnings("ignore")

examples = [{},{},{}]; 
examples[0]['question'] = """Caption: Composition of the glasses, their glass transition temperature T
g, and sample preparation temperature T

Table: [['', 'mol%', 'mol%', '', 'mol%', 'mol%', 'mol%', '', ''], ['Sample No.', 'AgI', 'Ag2MoO4', '', 'AgI', 'Ag2O', 'MoO3', 'T g (degC)', 'T p (degC)'], ['series I', 'series I', 'series I', 'series I', 'series I', 'series I', 'series I', 'series I', 'series I'], ['1', '75', '25', '', '(60', '20', '20)', '54', '500'], ['2', '70', '30', '', '(53.8', '23.1', '23.1)', '60', '500'], ['3', '66.7', '33.3', '', '(50', '25', '25)', '65', '500'], ['4', '60', '40', '', '(42.8', '28.6', '28.6)', '79', '500'], ['series II', 'series II', 'series II', 'series II', 'series II', 'series II', 'series II', 'series II', 'series II'], ['3', '(66.7', '33.3)', '', '50', '25', '25', '65', '500'], ['5', '', '', '', '50', '24', '26', '75', '500'], ['6', '', '', '', '50', '23', '27', '84', '530'], ['7', '', '', '', '50', '22', '28', '94', '530'], ['8', '', '', '', '50', '21', '29', '101', '530'], ['9', '', '', '', '50', '20', '30', '111', '530']]

Footer: {}"""
examples[0]['answer'] = "{'comp_table': [1], 'composition_row_index': [3, 4, 5, 6, 8, 9, 10, 11, 12, 13], 'chemical_col_index': [4, 5, 6], 'gid_col_index': [0], 'regex_table': [0]}"
examples[1]['question'] = """Caption: Stopping powers and ion ranges of the ions used in this work, calculated with SRIM [30].

Table: [['Ion', 'E (MeV)', 'Se (keV/mm)', 'Sn (keV/mm)', 'ST (keV/mm)', 'Ion range (mm)'], ['H+', '2.0', '27.46', '0.02', '27.48', '46'], ['He+', '4.0', '173.0', '0.1', '173.1', '16.7'], ['C+', '4.0', '1291', '3', '1294', '4.2'], ['Si6+', '28.6', '3417', '6', '3423', '10.2'], ['Br4+', '18.0', '4959', '104', '5063', '6.3'], ['Br6+', '28.0', '6066', '73', '6139', '8.1']]

Footer: {}"""
examples[1]['answer'] = "{'comp_table': [0], 'regex_table': [0]}"; 
examples[2]['question'] = """Caption: Composition of prepared bulk and thin films together with PLD conditions and calculated average deposition rates

Table: [['Sample', 'Preliminary pressurex10-4 (Pa)', 'Laser fluency(Jcm-2)', 'Deposition time(min)', 'Average deposition rate(nmpulse-1)', 'Composition (at.%)', 'Composition (at.%)', 'Composition (at.%)'], ['Sample', 'Preliminary pressurex10-4 (Pa)', 'Laser fluency(Jcm-2)', 'Deposition time(min)', 'Average deposition rate(nmpulse-1)', 'Ag', 'Sb', 'S'], ['Bulk 1', '-', '-', '-', '-', '26.5', '24.5', '49.0'], ['PLD 1', '8.2', '1', '10', '0.065', '27.8', '26.9', '45.3'], ['PLD 2', '5.8', '1', '20', '0.032', '26.4', '25.6', '48.0'], ['PLD 3', '7.4', '2', '16', '0.029', '27.7', '27.3', '45.0']]

Footer: {}""";
examples[2]['answer'] = "{'comp_table': [1], 'composition_row_index': [3, 4, 5], 'chemical_col_index': [5, 6, 7], 'gid_col_index': [0], 'regex_table': [0]}"
examples = examples[1:]; #due to lack of context length
    
def prepare_prompt_fewshot(sample):
    eval_prompt = ''
    for role in ['system']:
        eval_prompt += f"<|im_start|>{role}\n{sample[role]}<|im_end|>\n"

    eval_prompt += "the following are some examples so you can see the output format\n"
    for example in examples:
        for role in ['question', 'answer']:
            eval_prompt += f"<|im_start|>{role}\n{example[role]}<|im_end|>\n"
    # eval_prompt += "now answer the final question in the same format as the example answers above\n";
    #then  we ask it the real question and leave the answer hanging
    eval_prompt += f"<|im_start|>question\n{sample['question']}<|im_end|>\n";
    eval_prompt += f"<|im_start|>answer\n"
    return eval_prompt; 

def prepare_prompt(sample):
    if '/7b-chat' in path: # for base llama2
        return prepare_prompt_llama2(sample)
    if '/8b-chat' in path: # for base llama3
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
# %%
outputcs = []
give_examples = 0
for idx, sample in tqdm(enumerate(valfile[:])):
    # eval_prompt = f"{sample['system']} "+f"input-{sample['input']}" + "output-"
    eval_prompt = prepare_prompt_fewshot(sample) if give_examples != 0 else prepare_prompt(sample); ## Original prompt preparation.
    # eval_prompt = prepare_prompt_fewshot(sample) 
    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    print("len is :", model_input['input_ids'].shape)
    if model_input['input_ids'].shape[1]>=2048:
        print('skipping because input is greater than 2048')
        outputcs.append(2048)
    else:
        output = model.generate(**model_input, temperature = 0.05, top_p=0.5, max_length= min(int(model_input['input_ids'].shape[1]+ out_tokens[idx]+100),2048),do_sample=False) # for atom count and atom name
        generated_tokens = output[0, model_input['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        outputcs.append(generated_text)

import pickle
if len(sys.argv) > 3:
    savepath = sys.argv[3] + "_discomat_test.pkl"
else:
    savepath = path.split('/')[-3] + "_discomat_test.pkl";

with open(savepath,'wb') as f:
    pickle.dump(outputcs, f)
f.close()

print("saved to ", savepath);

exit(); #we exit after generating results.

