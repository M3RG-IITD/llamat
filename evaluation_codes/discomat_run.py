
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

discomat = json.load(open('../../datasets/ift/raw_ift/ift_discomat.json', 'r'))
discomat_train = discomat['train']
discomat_val = discomat['val']
discomat_test = discomat['test']

len(discomat_train['input']), len(discomat_train['output'])

# %%
sp_discomat=['''You are an expert in materials science and extracting data from tables. You have the fill the following dictionary for the given table. Each key is defined as follows:
'comp_table'- If the input table has material compositions then return [1], else [0];
'regex_table'- If the input table has material compositions and they can be extracted using a regular expression parser, then return [1], else [0].
'composition_row_index'-The list containing the index of rows which have complete information about material composition.
'chemical_col_index'-The list containing the index of columns which report values of constituent chemicals of the material.
'composition_col_index'-The list containing the index of columns which have complete information about material composition.
'chemical_row_index'-The list containing the index of rows which report values of constituent chemicals of the material.
'gid_row_index'-The index of row having material identifier.
'gid_col_index'-The index of column having material identifier.
 
dictionary =  
{'comp_table': [],
'regex_table': [],
'composition_row_index': [],
'composition_col_index': [],
'chemical_row_index': [],
'chemical_col_index': [],
'gid_row_index': [],
'gid_col_index': []}
NOTE:The output will be the dictionary with keys having non-empty lists ONLY.
''',
'''As a materials science expert skilled in extracting information from tables, your objective is to complete the following dictionary based on the table provided. Define each key as follows:
'comp_table'- Assign [1] if the table includes data on material compositions, otherwise [0];
'regex_table'- Assign [1] if material compositions are present and extractable via regex, otherwise [0];
'composition_row_index'- Indices of rows with full material composition details.
'chemical_col_index'- Indices of columns showing the constituent chemicals' values.
'composition_col_index'- Indices of columns with full material composition details.
'chemical_row_index'- Indices of rows showing the constituent chemicals' values.
'gid_row_index'- Index of the row with the material identifier.
'gid_col_index'- Index of the column with the material identifier.

dictionary =
{'comp_table': [],
 'regex_table': [],
 'composition_row_index': [],
 'composition_col_index': [],
 'chemical_row_index': [],
 'chemical_col_index': [],
 'gid_row_index': [],
 'gid_col_index': []}
NOTE: Only keys with non-empty lists will be included in the output.''',
'''You are an expert in materials science with skills in table data extraction. Please fill in the following dictionary according to the specified table. Each key should be defined as:
'comp_table'- [1] if material compositions are included in the table, otherwise [0];
'regex_table'- [1] if regex can be used to extract material compositions, otherwise [0];
'composition_row_index'- List of row indices with complete composition information.
'chemical_col_index'- List of column indices with values of constituent chemicals.
'composition_col_index'- List of column indices with complete composition data.
'chemical_row_index'- List of row indices with chemical values.
'gid_row_index'- Row index for material identifier.
'gid_col_index'- Column index for material identifier.

dictionary =
{'comp_table': [],
 'regex_table': [],
 'composition_row_index': [],
 'composition_col_index': [],
 'chemical_row_index': [],
 'chemical_col_index': [],
 'gid_row_index': [],
 'gid_col_index': []}
Remember, only populate keys that contain data.
''',
'''Utilize your expertise in materials science and data extraction to populate the following dictionary from the data in the provided table. Each key should be populated as follows:
'comp_table'- Include [1] if there are material compositions in the table, else [0];
'regex_table'- Include [1] if material compositions can be parsed using regex, else [0];
'composition_row_index'- Row indices with complete material composition details.
'chemical_col_index'- Column indices that display constituent chemicals.
'composition_col_index'- Column indices with detailed material compositions.
'chemical_row_index'- Row indices that display constituent chemicals.
'gid_row_index'- Specific row index with the material identifier.
'gid_col_index'- Specific column index with the material identifier.

dictionary =
{'comp_table': [],
 'regex_table': [],
 'composition_row_index': [],
 'composition_col_index': [],
 'chemical_row_index': [],
 'chemical_col_index': [],
 'gid_row_index': [],
 'gid_col_index': []}
NOTE: The final output will include only keys that have entries.
''',
'''You, as a material scientist skilled in data extraction from structured formats, are to fill out the below dictionary from the provided table data. Each dictionary key corresponds to:
'comp_table'- Enter [1] if the table contains material compositions, otherwise [0];
'regex_table'- Enter [1] if regex tools can extract material compositions, otherwise [0];
'composition_row_index'- Index list of rows with full material composition info.
'chemical_col_index'- Index list of columns reporting chemical constituents.
'composition_col_index'- Index list of columns with detailed material compositions.
'chemical_row_index'- Index list of rows reporting chemical constituents.
'gid_row_index'- Index of the row containing the material identifier.
'gid_col_index'- Index of the column containing the material identifier.

dictionary =
{'comp_table': [],
 'regex_table': [],
 'composition_row_index': [],
 'composition_col_index': [],
 'chemical_row_index': [],
 'chemical_col_index': [],
 'gid_row_index': [],
 'gid_col_index': []}
Output will only show keys with data-filled lists.
''',
'''Leverage your expertise in materials science to fill the provided dictionary based on the table characteristics. Define each dictionary key as follows:
'comp_table'- [1] if the table shows material compositions, [0] otherwise;
'regex_table'- [1] if material compositions are extractable via regex, [0] otherwise;
'composition_row_index'- Rows that fully detail material compositions.
'chemical_col_index'- Columns reporting the chemical constituents.
'composition_col_index'- Columns that detail material compositions fully.
'chemical_row_index'- Rows reporting the chemical constituents.
'gid_row_index'- The row with the material identifier.
'gid_col_index'- The column with the material identifier.

dictionary =
{'comp_table': [],
 'regex_table': [],
 'composition_row_index': [],
 'composition_col_index': [],
 'chemical_row_index': [],
 'chemical_col_index': [],
 'gid_row_index': [],
 'gid_col_index': []}
Note: Only include keys in the output that have filled lists.
'''
]

# %%

def convert_dsm_to_text(inp):
    output = f"Caption: {inp['caption']}\n\nTable: {inp['act_table']}\n\nFooter: {inp['footer']}"
    return output

random.seed(0)
discomat_train = [{'system':random.sample(sp_discomat, 1)[0], 'question':convert_dsm_to_text(discomat_train['input'][i]), 'answer':str(discomat_train['output'][i])} for i in range(len((discomat_train['input'])))]
discomat_val = [{'system':random.sample(sp_discomat, 1)[0], 'question':convert_dsm_to_text(discomat_val['input'][i]), 'answer':str(discomat_val['output'][i])} for i in range(len((discomat_val['input'])))]
discomat_test = [{'system':random.sample(sp_discomat, 1)[0], 'question':convert_dsm_to_text(discomat_test['input'][i]), 'answer':discomat_test['output'][i]} for i in range(len((discomat_test['input'])))]
# discomat_train += discomat_val
random.shuffle(discomat_train)
random.shuffle(discomat_test)
# with open('processed_ift/discomat_train.jsonl', 'w') as f:
#     for i in discomat_train:
#         f.write(json.dumps(i)+'\n')
# with open('processed_ift/discomat_test.jsonl', 'w') as f:
#     for i in discomat_test:
#         f.write(json.dumps(i)+'\n')

# %%
len(discomat_test)

# %%
train_system = [data['system'] for data in discomat_train]
train_question = [data['question'] for data in discomat_train]
train_answer = [data['answer'] for data in discomat_train]

df_train = pd.DataFrame(train_system, columns=['system'])
df_train['question'] = train_question
df_train['answer'] = train_answer
# df_train.to_csv('/home/scai/phd/aiz217586/llama-recipes/recipes/finetuning/huggingface_trainer/train.csv')

# %%
train_answer[0]

# %%
# print(df_train['system'].iloc[0] + '\n\ninput: ' + df_train['question'].iloc[0])

# %%
test_system = [data['system'] for data in discomat_test]
test_question = [data['question'] for data in discomat_test]
test_answer = [data['answer'] for data in discomat_test]

df_test = pd.DataFrame(test_system, columns=['system'])
df_test['question'] = test_question
df_test['answer'] = test_answer
# df_test.to_csv('/home/scai/phd/aiz217586/llama-recipes/recipes/finetuning/huggingface_trainer/test.csv')

# %%
val_system = [data['system'] for data in discomat_val]
val_question = [data['question'] for data in discomat_val]
val_answer = [data['answer'] for data in discomat_val]

df_val = pd.DataFrame(val_system, columns=['system'])
df_val['question'] = val_question
df_val['answer'] = val_answer
# df_val.to_csv('/home/cse/btech/cs1200389/llama-recipes/src/llama_recipes/datasets/discomat_dataset/val.csv')

# %%
# @dataclass
# class train_config:
#     model_name: str="/scratch/cse/btech/cs1200448/hf-to-meditron-weights/cerebras_4k_llama38b_chat_stage_1_hb_math3x_msqa_3x_cement_matscinlp_mascqa/320/hf"
#     tokenizer_name: str=None
#     enable_fsdp: bool=False
#     low_cpu_fsdp: bool=False
#     run_validation: bool=True
#     batch_size_training: int=4
#     batching_strategy: str="packing" #alternative: padding
#     context_length: int=2048
#     gradient_accumulation_steps: int=1
#     gradient_clipping: bool = False
#     gradient_clipping_threshold: float = 1.0
#     num_epochs: int=3
#     max_train_step: int=0
#     max_eval_step: int=0
#     num_workers_dataloader: int=1
#     lr: float=2e-5
#     weight_decay: float=0.0
#     gamma: float= 0.85
#     seed: int=42
#     use_fp16: bool=False
#     mixed_precision: bool=True
#     val_batch_size: int=1
#     dataset = "discomat_dataset"
#     peft_method: str = "lora" # None, llama_adapter (Caution: llama_adapter is currently not supported with FSDP)
#     use_peft: bool=False
#     from_peft_checkpoint: str="" # if not empty and use_peft=True, will load the peft checkpoint and resume the fine-tuning on that checkpoint
#     output_dir: str = "/scratch/cse/btech/cs1200448/hf-to-meditron-weights/cerebras_4k_llama38b_chat_stage_1_hb_math3x_msqa_3x_cement_matscinlp_mascqa_discomat_chat"
#     freeze_layers: bool = False
#     num_freeze_layers: int = 1
#     quantization: bool = False
#     one_gpu: bool = False
#     save_model: bool = True
#     dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
#     dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
#     save_optimizer: bool=False # will be used if using FSDP
#     use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
#     use_wandb: bool = False # Enable wandb for experient tracking
#     save_metrics: bool = True # saves training metrics to a json file for later plotting
#     flop_counter: bool = True # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time.
#     flop_counter_start: int = 3 # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
#     use_profiler: bool = False # Enable pytorch profiler, can not be used with flop counter at the same time.
#     profiler_dir: str = "./results" # will be used if using profiler

# %%
valfile = []
with open("../../datasets/ift/final_ift/discomat_dense_test.jsonl", 'r') as f:
    valfile = [json.loads(line) for line in f.readlines()]

# %%
for i in range(737):
    if(discomat_test[i]['question'][:40] == valfile[4]['question'][:40]):
        print(i)

# %%
discomat_test[264].keys()

# %%
# discomat_test
# import pickle
# with open("shuffled_discotest_to_valfile.pkl", "wb") as f:
#     pickle.dump(shuffled, f);

# %%
shuffled = [0]*len(discomat_test);
for i in range(len(valfile)):
    for j in range(len(discomat_test)):
        if(valfile[i]['question'] == discomat_test[j]['question']):
            shuffled[j] = i;
            break;

# %%
from tqdm import tqdm

# %%
out_tokens = []
in_tokens = []
for sample in tqdm(valfile):
    eval_out = str(sample['answer'])
    model_output = tokenizer(eval_out, return_tensors="pt")#.to("cuda")
    out_tokens.append(model_output['input_ids'].shape[1])
    eval_in = str(sample['question'])
    model_output = tokenizer(eval_in, return_tensors="pt")#.to("cuda")
    in_tokens.append(model_output['input_ids'].shape[1])

# %%
out_tokens.index(max(out_tokens)), in_tokens.index(max(in_tokens))

# %%
# in_tokens

# %%

# eval_prompt = f"{sample['system']} "+f"question-{sample['question']}" + "output-"
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
import matplotlib.pyplot as plt

# %%
plt.hist(out_tokens)

# %%
idx=4
out_tokens[idx]

# %%
model.eval()
with torch.no_grad():
    # for min(int(model_input['input_ids'].shape[1]+ get_gen_len(len(sample['answer']))),2048)
    print(tokenizer.decode(model.generate(**model_input, top_p=0.95, temperature=0.01, max_length= min(int(model_input['input_ids'].shape[1]+ out_tokens[idx]+50),2048),do_sample=False)[0], skip_special_tokens=True))

# %%
from tqdm import tqdm

# %%
from warnings import  filterwarnings
filterwarnings("ignore")

# %%
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
# def prepare_prompt(sample):
#     eval_prompt = ''
#     for role in ['system','question']:
#         eval_prompt += f"<|im_start|>{role}\n{sample[role]}<|im_end|>\n"
#     eval_prompt += f"<|im_start|>answer\n"
#     return eval_prompt; 
    
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

# %%
import pickle
## name should be simply.
if len(sys.argv) > 3:
    savepath = sys.argv[3] + "_discomat_test.pkl"
else:
    savepath = path.split('/')[-3] + "_discomat_test.pkl";

with open(savepath,'wb') as f:
    pickle.dump(outputcs, f)
f.close()

print("saved to ", savepath);

exit(); #we exit after doing all this.
# %%

# old_output = outputcs
# with open("old_output.pkl", "wb") as f:
#     pickle.dump(outputcs, f)

# %%

# %%
# outputcs[5] is "{'comp_table': [1], 'composition_col_index': [1, 2, 3, 4], 'chemical_row_index': [1, 2], 'gid_row_index': [0],'regex_table': [0]}<|im_end|>\n<|im_start|>answer\n{'comp_table': [1], 'composition_col_index': [1, 2, 3, 4], 'chemical_row_index': ["

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
testidx = 16;
sample = valfile[testidx]
eval_prompt = ''
for role in ['system','question']:
    eval_prompt += f"<|im_start|>{role}\n{sample[role]}<|im_end|>\n"

eval_prompt += f"<|im_start|>answer\n"
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
if model_input['input_ids'].shape[1]>=2048:
    print('skipping because input is greater than 2048')
    outputcs.append(2048)
else:
    output = model.generate(**model_input, bos_token_id=1380101, eos_token_id=128011,top_p=0.95, max_length= min(int(model_input['input_ids'].shape[1]+ out_tokens[idx]+50),2048),do_sample=False) # for atom count and atom name
    generated_tokens = output[0, model_input['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    # outputcs.append(generated_text)
print(generated_text)

# %%
valfile[testidx]["answer"]

# %%


# %%
len(valfile)

# %%
import pickle



# %%
end_tok = '<|im_end|>'
# end_tok = '\n an'
import ast
ast.literal_eval(outputcs[-1].split(end_tok)[0])

# %%
import json

# %%
valfile = []
with open("../../datasets/ift/final_ift/discomat_dense_test.jsonl", 'r') as f:
    valfile = [json.loads(line) for line in f.readlines()]

# %%
len(valfile), len(outputcs)

# %%
actual = []
predicted = []
done = []
missed = []
count = 0
for i, o in enumerate(outputcs):
    try:
        actual.append(valfile[i]['answer'])
        predicted.append(ast.literal_eval(o.split(end_tok)[0]))
        done.append(i)
        if valfile[i]['answer'] == ast.literal_eval(o.split(end_tok)[0]):
            count += 1
            # if len(valfile[i]['answer'])>2: #pass
            # # else: print(valfile[i]['answer'])
            #     print(valfile[i]['answer'])
        else:
            print(i, actual[i], predicted[i])
            print(valfile[i]['answer'])
    except Exception as e:
        # print(e)
        missed.append(i)
    
import pickle
import os
import pandas as pd
import sys

# %%
# res_discomat = pickle.load(open('res_discomat.pkl', 'rb'))
res_discomat = pickle.load(open('../../../DiSCoMaT_Kausik_new_ultimate_final_2/code/inference/res_discomat.pkl', 'rb'))

# %%
res_discomat.keys()

# %%
for k in ['identifier', 'pred_table_type_labels', 'pred_row_col_labels', 'pred_edges', 'pred_tuples']:
    print(k, len(res_discomat[k]))

# %%
# print(len(res_discomat['pred_tuples']))
# print((res_discomat['pred_row_col_labels']))

table_dir = '../../data'

train_val_test_split = pickle.load(open(os.path.join(table_dir, 'train_val_test_split.pkl'), 'rb'))
#comp_data_dict = {(c['pii'], c['t_idx']): c for c in comp_data}

datta = pickle.load(open('../../data/val_test_data.pkl','rb'))


pii_tid_dict = dict()
for table in datta:
    pii = table['pii']
    t_idx = table['t_idx']
    pii_tid = (pii, t_idx)
    pii_tid_dict[pii_tid] = table

len(valfile)

discopred = dict()
for i, iden in enumerate(res_discomat['identifier']):
    if iden in train_val_test_split['test']:
        pred_table_type_labels = res_discomat['pred_table_type_labels'][i]
        pred_row_col_labels = res_discomat['pred_row_col_labels'][i]
        pred_edges = res_discomat['pred_edges'][i]
        pred_tuples = res_discomat['pred_tuples'][i]
        discopred[iden] = {'pred_table_type_labels':pred_table_type_labels,
                           'pred_row_col_labels': pred_row_col_labels,
                           'pred_edges': pred_edges,
                           'pred_tuples': pred_tuples,
                           }

# %%
identifiers = []

for i, iden in enumerate(res_discomat['identifier']):
    if iden in train_val_test_split['test']:
        # pred_test_list.append(res_discomat['pred_row_col_labels'][i])
        identifiers.append(iden)

# %%
len(identifiers)

# %%
import json

# %%
valfile = []
with open("/home/cse/btech/cs1200389/MatLlama/MatLLaMA/datasets/ift/final_ift/discomat_test_dense.jsonl", 'r') as f:
    valfile = [json.loads(line) for line in f.readlines()]

# %%
# dataset = []
msg2piitid = dict()

for pii_tid in identifiers:
    pii = pii_tid[0]
    tid = pii_tid[1]
    # print(pii,tid)
    table = pii_tid_dict[(pii,tid)]
    msg1 = f"Caption: {table['caption']}\n\nTable: {table['act_table']}\n\nFooter: {table['footer']}"

    msg2piitid[msg1] = (pii,tid)
    # # break
    # if msg1 == msg:
    #     print(pii_tid)

# %%
pred_llm = pickle.load(open('/home/cse/btech/cs1200389/MatLlama/MatLLaMA/src/inference_newtasks/cerebras_4k_llama38b_chat_stage_1_discomat_dense_test.pkl','rb'))

# %%
for idx, val in enumerate(valfile):
    pii_tid = msg2piitid[val['question']]
    val['pii_tid'] = {'pii':pii_tid[0], 'tid':pii_tid[1], 'gold':pii_tid_dict[(pii,tid)], 'llamat3':pred_llm[idx], 'discomat':discopred[pii_tid]}

# valfile[0]

with open('/home/cse/btech/cs1200389/MatLlama/MatLLaMA/src/inference_newtasks/all_data_cerebras_4k_llama38b_chat_stage_1_discomat_dense_test.pkl','wb') as f:
    pickle.dump(valfile,f)
f.close()

val['pii_tid']['discomat']

import pickle

# with open('test_discomat_dense_indexing

# msg2piitid

print(len(pred_test_list))

pred_test_list[9]

all_pred_tuples = []
for tup in res_discomat['pred_tuples']:
    if any(tup):
        all_pred_tuples += tup

material_pred = {}
for tup in all_pred_tuples:
    key = tup[0]
    if key not in material_pred:
        material_pred[key] = []
    material_pred[key].append(tup)

results_ = {};

import pickle
import json
tests = ['llamat3_13812_additional_discomat_test.pkl', 'llamat3_chat4k_discomat_test.pkl',  'llamat3_chat8k_discomat_test.pkl', 'llamat3_chat13k_discomat_test.pkl' , 'llamat3_chat_13812_discomat_test.pkl','discomat_dense_test_llama2.pkl']
idx = -1;
names = ['13812_additional', '4k', '8k', '13k', '13812', 'llama2']
curname = names[idx];
curtest = tests[idx];
# curtest = 'cerebras_13812_llama38b_mof12_doping_discomat_downstream_updated_discomat_test.pkl';
curtest = 'cerebras_13812_llama38b_orca_576k_mof12_doping_discomat_downstream_updated_test3_discomat_test_updated.pkl';
curname = '13812_updated';
with open(curtest,'rb') as f:
    outputcs = pickle.load(f)
f.close()
results_[curname] = {}; 
results_[curname]['ds'] = {}

if 'llama2' in curtest: split_token = '\n an'
else: split_token = '<|im_e'

import pickle
import json
import ast
valfilepath = 'discomat_test_valfile.jsonl'
with open(valfilepath,'r') as f:
    valfile = [json.loads(x) for x in f.readlines()]
f.close()


# %%
valfile[0]

# %%
len(valfile), len(outputcs)
valfile[0]['answer']

# %%
ast.literal_eval(outputcs[0].split(split_token)[0])

# %%
import ast

# %%
# valfile[5]['pii_tid']['gold']
                      

# %%
def format_discomat_preds(idx):
    disco_pred_answer = dict()
    rcd = valfile[idx]['pii_tid']['discomat']['pred_row_col_labels']
    
    # for tables having chemical names in rows, and one complete comp is in 1 column
    if 2 in rcd['row']:
        disco_pred_answer['chemical_row_index'] = list(np.where(np.array(rcd['row']) == 2)[0])
        disco_pred_answer['composition_col_index'] = list(np.where(np.array(rcd['col']) == 1)[0])
        
        if 3 in rcd['row']:
            disco_pred_answer['gid_row_index'] = [rcd['row'].index(3)]
    
    if 2 in rcd['col']:
        disco_pred_answer['chemical_col_index'] = list(np.where(np.array(rcd['col']) == 2)[0])
        disco_pred_answer['composition_row_index'] = list(np.where(np.array(rcd['row']) == 1)[0])
        
        if 3 in rcd['col']:
            disco_pred_answer['gid_col_index'] = [rcd['col'].index(3)]
    
    if valfile[idx]['pii_tid']['discomat']['pred_table_type_labels'] == 0:
        disco_pred_answer['comp_table'] = [1]
        disco_pred_answer['regex_table'] = [1]
        if 3 in rcd['row']:
            disco_pred_answer['gid_row_index'] = [rcd['row'].index(3)]
        if 3 in rcd['col']:
            disco_pred_answer['gid_col_index'] = [rcd['col'].index(3)]
        
    if 0 < valfile[idx]['pii_tid']['discomat']['pred_table_type_labels'] < 3:
        disco_pred_answer['comp_table'] = [1]
        disco_pred_answer['regex_table'] = [0]
    
    if valfile[idx]['pii_tid']['discomat']['pred_table_type_labels'] == 3:
        disco_pred_answer['comp_table'] = [0]
        disco_pred_answer['regex_table'] = [0]
    
    return disco_pred_answer

# %%
import numpy as np

# %%
# format_discomat_preds(5)

# %%
exact_match = {'discomat':0, 'llamat2':0}


# %%
missed = []

for i in range(len(valfile)):

    actual = valfile[i]['answer']
    # pred_d = format_discomat_preds(i) # valfile[i]['pii_tid']['discomat']
    if outputcs[i] != 2048:
        try:
            # pred_2 = outputcs[i]
            pred_d = ast.literal_eval(pred_2.split(split_token)[0])
            if pred_d == actual:
                exact_match['discomat'] += 1
            # elif pred == actual:
            #     exact_match['llamat2'] += 1
        except:
            missed.append(i)
    
    # outputcs[

# %%
exact_match

# %%
missed = []

for i in range(len(valfile)):

    actual = valfile[i]['answer']
    # pred_d = format_discomat_preds(i) # valfile[i]['pii_tid']['discomat']
    if outputcs[i] != 2048:
        try:
            pred_2 = outputcs[i]
            pred = ast.literal_eval(pred_2.split(split_token)[0])
            # if pred_d == actual:
            #     exact_match['discomat'] += 1
            if pred == actual:
                exact_match['llamat2'] += 1
        except:
            missed.append(i)
    
    # outputcs[

# %%
len(missed)

# %%
exact_match

# %%
longlen = 0
for t in outputcs:
    if t == 2048:longlen+=1
longlen

# %%
print(f'Tables non parsable = {len(missed)}/{len(valfile)-longlen}')
for k,v in exact_match.items():
    print(k)
    print(f'Exact match for {k}:', v)
    
    if k=='discomat':
        print(f'Fraction Exact match for {k}:', v/(len(valfile)))
#         results_['discomat']['exact_match'] =  [v,(len(valfile))]
        results_[curname]['ds']['exact_match'] = [v,(len(valfile))]; 
    else:
        print(f'Fraction Exact match for {k}:', v/(len(valfile)-len(missed)-longlen))
        results_[curname]['exact_match'] = [v,(len(valfile)-len(missed)-longlen)];
    # print(f'Exact match for {k}:', v)
    
    

# %%
# help(precision_score)
valfile[5]['answer']

# %%
act_comptable = []
act_regextable = []

pred_comptable_discomat = []
pred_comptable_llamat = []

pred_regex_discomat = []
pred_regex_llamat = []


# for discomat
for i in range(len(valfile)):
    if i not in missed:
        if outputcs[i] != 2048:
            actual = valfile[i]['answer']
            act_comptable.append(actual['comp_table'][0])
            act_regextable.append(actual['regex_table'][0])
            # pred_d = format_discomat_preds(i) # valfile[i]['pii_tid']['discomat']
            # pred_comptable_discomat.append(pred_d['comp_table'][0])
            # pred_regex_discomat.append(pred_d['regex_table'][0])
        
            pred_2 = outputcs[i]
            pred = ast.literal_eval(pred_2.split(split_token)[0])
            try:
                pred_comptable_llamat.append(pred['comp_table'][0])
            except:
                pred_comptable_llamat.append(0)
            try:
                pred_regex_llamat.append(pred['regex_table'][0])
            except:
                pred_regex_llamat.append(0)

# %%
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

target = ['NON-C','COMP']
print('llamat')
print('Report', classification_report(act_comptable, pred_comptable_llamat, target_names=target))

# %%
# 737-144-4
results_[curname]['comptable'] = [(f1_score(act_comptable, pred_comptable_llamat)), np.array(act_comptable).sum()]; #stores the sum as well as the support.
results_[curname]['ds']['comptable'] = ''; #(f1_score(act_comptable, pred_comptable_discomat)); 

# %%
np.array(act_comptable).sum()

# %%
from sklearn.metrics import classification_report

target = ['no-regex','regex']
# print('discomat')
# print('Report', classification_report(act_regextable, pred_regex_discomat, target_names=target))

print('llamat')
print('Report', classification_report(act_regextable, pred_regex_llamat, target_names=target))

# %%
results_[curname]['regex'] = [(f1_score(act_regextable, pred_regex_llamat)), np.array(act_regextable).sum()]; #stores the sum as well as the support.
results_[curname]['ds']['regex'] = ' '; #(f1_score(act_regextable, pred_regex_discomat)); 

# %%
a = {'a':'hello'}
(type(a))

# %%
gid_actual = []
gid_pred = []

gid_discomat_actual = []
gid_discomat_pred = []


for i in range(len(valfile)):
    if i not in missed:
        if outputcs[i] != 2048:
            
            actual = valfile[i]['answer']
            
            
            if 'gid_row_index' in actual.keys(): #or 'gid_col_index' in actual.keys():
                
                # gid_actual.append(1)
                
                # pred_d = format_discomat_preds(i)
                
                actual_label = actual['gid_row_index']
                
                
                pred_2 = outputcs[i]
                
                pred = ast.literal_eval(pred_2.split(split_token)[0])
                
                if type(pred) is dict:
                    
                    # gid_discomat_actual.append(1)
                    gid_actual.append(1)
                    
                    if 'gid_row_index' not in pred.keys():
                        pred['gid_row_index'] = []


                    # if 'gid_row_index' not in pred_d.keys():
                        # pred_d['gid_row_index'] = []

                # print(actual_label, pred['gid_row_index'])

                    if pred['gid_row_index'] == actual_label:
                        gid_pred.append(1)
                    else:
                        gid_pred.append(0)


                    # if pred_d['gid_row_index'] == actual_label:
                        # gid_discomat_pred.append(1)
                    # else:
                        # gid_discomat_pred.append(0)
                # else:
                #     gid_pred.append(0)
                    
            if 'gid_col_index' in actual.keys(): #or 'gid_col_index' in actual.keys():
                
                # gid_actual.append(1)
                
                # pred_d = format_discomat_preds(i)
                
                actual_label = actual['gid_col_index']
                pred_2 = outputcs[i]
                
                pred = ast.literal_eval(pred_2.split(split_token)[0])
                
                if type(pred) is dict:
                    gid_actual.append(1)
                    
                    # gid_discomat_actual.append(1)
                    if 'gid_col_index' not in pred.keys():
                        pred['gid_col_index'] = []

                    # if 'gid_col_index' not in pred_d.keys():
                        # pred_d['gid_col_index'] = []
                    # print(actual_label, pred['gid_col_index'])

                    if pred['gid_col_index'] == actual_label:
                        gid_pred.append(1)
                    else:
                        gid_pred.append(0)

                    # if pred_d['gid_col_index'] == actual_label:
                        # gid_discomat_pred.append(1)
                    # else:
                        # gid_discomat_pred.append(0)
                # else:
                #     gid_pred.append(0)
                

# %%
from sklearn.metrics import f1_score, recall_score, precision_score

# %%
# Glass id scores
from sklearn.metrics import classification_report

# print('discomat')
# for func in [precision_score, recall_score, f1_score]:
#     print(round(func(gid_actual, gid_discomat_pred),3))
    
print('llamat')
for func in [precision_score, recall_score, f1_score]:
    print(round(func(gid_actual, gid_pred),3))

# %%
# 737-144-4
results_[curname]['gid'] = [(f1_score(gid_actual, gid_pred)), np.array(gid_actual).sum()]; #stores the sum as well as the support.
results_[curname]['ds']['gid'] = ""; #(f1_score(gid_actual, gid_discomat_pred)); 

# %%
results_[curname].keys()

# %%
composition_actual = []
composition_pred = []

composition_discomat_pred = []


for i in range(len(valfile)):
    if i not in missed:
        if outputcs[i] != 2048:
            
            actual = valfile[i]['answer']
            
            
            if 'composition_row_index' in actual.keys(): #or 'gid_col_index' in actual.keys():
                
                # composition_actual.append(1)
                
                # pred_d = format_discomat_preds(i)
                
                actual_label = actual['composition_row_index']
                pred_2 = outputcs[i]
                
                pred = ast.literal_eval(pred_2.split(split_token)[0])
                
                if type(pred) is dict:
                    
                    composition_actual.append(1)
                    
                    if 'composition_row_index' not in pred.keys():
                        pred['composition_row_index'] = []


                    # if 'composition_row_index' not in pred_d.keys():
                    #     pred_d['composition_row_index'] = []

                # print(actual_label, pred['gid_row_index'])

                    if sorted(pred['composition_row_index']) == sorted(actual_label):
                        composition_pred.append(1)
                    else:
                        composition_pred.append(0)


                    # if sorted(pred_d['composition_row_index']) == sorted(actual_label):
                    #     composition_discomat_pred.append(1)
                    # else:
                    #     composition_discomat_pred.append(0)
                    
            if 'composition_col_index' in actual.keys(): #or 'gid_col_index' in actual.keys():
                
                # composition_actual.append(1)
                
                # pred_d = format_discomat_preds(i)
                
                actual_label = actual['composition_col_index']
                pred_2 = outputcs[i]
                
                pred = ast.literal_eval(pred_2.split(split_token)[0])
                
                if type(pred) is dict:
                    
                    composition_actual.append(1)
                    
                    if 'composition_col_index' not in pred.keys():
                        pred['composition_col_index'] = []


                    # if 'composition_col_index' not in pred_d.keys():
                    #     pred_d['composition_col_index'] = []

                # print(actual_label, pred['gid_row_index'])

                    if sorted(pred['composition_col_index']) == sorted(actual_label):
                        composition_pred.append(1)
                    else:
                        composition_pred.append(0)


                    # if sorted(pred_d['composition_col_index']) == sorted(actual_label):
                    #     composition_discomat_pred.append(1)
                    # else:
                    #     composition_discomat_pred.append(0)
                

# %%
# composition row-col scores
from sklearn.metrics import classification_report

# print('discomat')
# for func in [precision_score, recall_score, f1_score]:
#     print(round(func(composition_actual, composition_discomat_pred),3))
    
print('llamat')
for func in [precision_score, recall_score, f1_score]:
    print(round(func(composition_actual, composition_pred),3))

# %%
# 737-144-4
results_[curname]['composition'] = [(f1_score(composition_actual, composition_pred)), np.array(composition_actual).sum()]; #stores the sum as well as the support.
results_[curname]['ds']['composition'] = "" ; #(f1_score(composition_actual, composition_discomat_pred)); 


# %%
composition_actual = []
composition_pred = []

composition_discomat_pred = []


for i in range(len(valfile)):
    if i not in missed:
        if outputcs[i] != 2048:
            
            actual = valfile[i]['answer']
            
            
            if 'chemical_row_index' in actual.keys(): #or 'gid_col_index' in actual.keys():
                
                # composition_actual.append(1)
                
                # pred_d = format_discomat_preds(i)
                
                actual_label = actual['chemical_row_index']
                pred_2 = outputcs[i]
                
                pred = ast.literal_eval(pred_2.split(split_token)[0])
                
                if type(pred) is dict:
                    
                    composition_actual.append(1)
                    if 'chemical_row_index' not in pred.keys():
                        pred['chemical_row_index'] = []


                    # if 'chemical_row_index' not in pred_d.keys():
                    #     pred_d['chemical_row_index'] = []

                # print(actual_label, pred['gid_row_index'])

                    if sorted(pred['chemical_row_index']) == sorted(actual_label):
                        composition_pred.append(1)
                    else:
                        composition_pred.append(0)


                    # if sorted(pred_d['chemical_row_index']) == sorted(actual_label):
                    #     composition_discomat_pred.append(1)
                    # else:
                    #     composition_discomat_pred.append(0)
                    
            if 'chemical_col_index' in actual.keys(): #or 'gid_col_index' in actual.keys():
                
                # composition_actual.append(1)
                
                # pred_d = format_discomat_preds(i)
                
                actual_label = actual['chemical_col_index']
                pred_2 = outputcs[i]
                
                pred = ast.literal_eval(pred_2.split(split_token)[0])
                
                if type(pred) is dict:
                    
                    composition_actual.append(1)
                    if 'chemical_col_index' not in pred.keys():
                        pred['chemical_col_index'] = []


                    # if 'chemical_col_index' not in pred_d.keys():
                    #     pred_d['chemical_col_index'] = []

                # print(actual_label, pred['gid_row_index'])

                    if sorted(pred['chemical_col_index']) == sorted(actual_label):
                        composition_pred.append(1)
                    else:
                        composition_pred.append(0)


                    # if sorted(pred_d['chemical_col_index']) == sorted(actual_label):
                    #     composition_discomat_pred.append(1)
                    # else:
                    #     composition_discomat_pred.append(0)
                

# %%
# chemical row-col scores
from sklearn.metrics import classification_report

# print('discomat')
# for func in [precision_score, recall_score, f1_score]:
#     print(round(func(composition_actual, composition_discomat_pred),3))
    
print('llamat')
for func in [precision_score, recall_score, f1_score]:
    print(round(func(composition_actual, composition_pred),3))

# %%
# 737-144-4
results_[curname]['chemical'] = [(f1_score(composition_actual, composition_pred)), np.array(composition_actual).sum()]; #stores the sum as well as the support.
results_[curname]['ds']['chemical'] = (f1_score(composition_actual, composition_discomat_pred)); 

# %%


# %%
print('discomat')
print('Precision', precision_score(act_comptable, pred_comptable_discomat,labels=[1,0]))

print('llamat')
print('Precision', precision_score(act_comptable, pred_comptable_llamat))

# %%
print('discomat')
print('Reccall', recall_score(act_comptable, pred_comptable_discomat))

print('llamat')
print('Reccall', recall_score(act_comptable, pred_comptable_llamat))

# %%
print('discomat')
print('F1-score', f1_score(act_comptable, pred_comptable_discomat))

print('llamat')
print('F1-score', f1_score(act_comptable, pred_comptable_llamat))

# %%
results_.keys()

# %%
def printR(*args, end = "\n"):
    for x in args:
        if(type(x) == float or type(x) == np.float64 or type(x) == np.float32):
            print(round(x,3), end = "");
        else:
            print(x, end = "");
    print(end, end = "");
    
cur = 0;
for model in results_.keys():
    
    if(cur == 0):
        print("model, ", end = "");
        for task in results_[model].keys():
            if(task == 'ds'): 
                continue; 
            if(task != 'exact_match'):
                print(task, ",ds", ",support", end = ",");
            else:
                print(task, ",ds", end = ",");
        print();
    cur += 1;
    printR(model, end = ",")
    for task in results_[model].keys():
        if(task == 'ds'): 
            continue; 
        if(task != 'exact_match'):
            printR(results_[model][task][0], end = ",") #, ",", results_[model]['ds'][task], ",", results_[model][task][1], end = ",");
            pass
        else:
            printR(results_[model][task][0], end = ",") #,'/',results_[model][task][1], ",", results_[model]['ds'][task][0],"/", results_[model]['ds'][task][1], end = ",");

    print()

# %%
results_

# %%
printR(results_[model][task][0])




