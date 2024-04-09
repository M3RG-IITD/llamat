import torch
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from accelerate import notebook_launcher
import os
import sys
import json
from tqdm import tqdm
import datasets
from sklearn.metrics import f1_score
import Levenshtein
import collections
import re
import string

np.random.seed(6041)
torch.manual_seed(6041)
torch.cuda.is_available()

model_id = sys.argv[1]
val_file = sys.argv[2]
val_dataset = load_dataset("json", data_files=val_file)['train']

tokenizer = LlamaTokenizer.from_pretrained(model_id, use_fast = True, padding_side = 'left')
model = LlamaForCausalLM.from_pretrained(model_id, device_map='auto', load_in_8bit=True, torch_dtype=torch.bfloat16)

tokenizer.add_special_tokens({'pad_token': '<unk>'})
model.config.pad_token_id = tokenizer.convert_tokens_to_ids('<unk>')

max_len = 2048
batch_size = 128
max_new_tokens = 50
predictions = []
with torch.no_grad():
    for i in tqdm(range(0,len(val_dataset), batch_size)):
        input_to_model = tokenizer(val_dataset['input'][i:i+batch_size], truncation = True, max_length = max_len, padding = True, return_tensors = "pt")['input_ids'].to("cuda")
        output = model.generate(input_to_model, max_new_tokens=max_new_tokens, pad_token_id=0, do_sample = True, top_p = 0.7, temperature = 0.5)
        predictions.append(output.cpu().numpy())
        if i % (10 * batch_size) == 0:
            del input_to_model
            del output
            torch.cuda.empty_cache()

predictions_reformed = [np.concatenate((np.zeros((i.shape[0], 2048 - i.shape[1])), i), axis = 1).astype(int) for i in predictions]
predictions_reformed = np.concatenate(predictions_reformed).astype(int)
sentences = tokenizer.batch_decode(predictions_reformed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_raw_scores(gt_pred_pair):
    exact_scores = {}
    f1_scores = {}
    gold_answers = [i[0] for i in gt_pred_pair]
    for i in tqdm(gt_pred_pair):
        qid = i[2]
        a_pred = i[1]
        exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
    return exact_scores, f1_scores

def make_eval_dict(exact_scores, f1_scores):
    total = len(exact_scores)
    assert len(exact_scores) == len(f1_scores)
    return collections.OrderedDict([
        ('exact', 1.0 * sum(exact_scores[k] for k in exact_scores) / total),
        ('f1', 1.0 * sum(f1_scores[k] for k in f1_scores) / total),
        ('total', total),
    ])

def most_similar_answer(a,answer_set):
    a = a.strip().replace(' ', '')
    if(a in answer_set):
        return a
    dis = [Levenshtein.distance(a,x) for x in answer_set]
    idx = np.argmin(dis)
    return answer_set[idx]

tasks = ["ner","pc","ee","re","sar","sc","qna","completion"]
out_dict = {i : [] for i in tasks}
out_dict["ner"] = {i : [] for i in ["sc_comics"]}
out_dict["pc"] = {i : [] for i in ["glass_non_glass"]}
out_dict["ee"] = {i : [] for i in ["sc_comics"]}
out_dict["re"] = {i : [] for i in ["structured_re", "sc_comics"]}
out_dict["sar"] = {i : [] for i in ["synthesis_actions"]}
out_dict["sc"] = {i : [] for i in ["sofc_sent"]}
out_dict["qna"] = {i : [] for i in ["squadv2"]}
out_dict["completion"] = {i : [] for i in ["hellaswag"]}

for i in range(len(sentences)):
    dp = val_dataset[i]
    raw = sentences[i].removeprefix(dp['input']).lower()
    if dp['task'] in ['sc', 'pc']:
        if 'yes' in raw and 'no' not in raw:
            ans = 'yes'
        elif 'no' in raw and 'yes' not in raw:
            ans = 'no'
        elif 'yes' not in raw and 'no' not in raw:
            continue
        else:
            ans = 'yes' if raw.find('yes') > raw.find('no') else 'no'
        out_dict[dp['task']][dp['dataset']].append([dp['output'], ans])
    if dp['task'] in ['qna']:
        ans = raw.split('\n')[0]
        out_dict[dp['task']][dp['dataset']].append([dp['output'], ans, dp['qid']])

scores = {i : [] for i in tasks if i in ['pc', 'sc', 'qna']}
# scores["ner"] = {i : [] for i in ["matscholar", "sofc_token", "sc_comics"]}
scores["pc"] = {i : [] for i in ["glass_non_glass"]}
# scores["sf"] = {i : [] for i in ["sofc_token"]}
# scores["ee"] = {i : [] for i in ["sc_comics"]}
# scores["re"] = {i : [] for i in ["structured_re", "sc_comics"]}
# scores["sar"] = {i : [] for i in ["synthesis_actions"]}
scores["sc"] = {i : [] for i in ["sofc_sent"]}
scores["qna"] = {i : [] for i in ["squadv2"]}

def evaluate_squad():
    exact_raw, f1_raw = get_raw_scores(out_dict['qna']['squadv2'])
    out_eval = make_eval_dict(exact_raw, f1_raw)
    scores['qna']['squadv2'] = out_eval['exact'], out_eval['f1']
    return out_eval['exact'], out_eval['f1']

def evaluate_pc(dataset):
    micro_f1 = f1_score(*list(zip(*out_dict['pc'][dataset])), average='micro', labels = list(['yes','no']))
    macro_f1 = f1_score(*list(zip(*out_dict['pc'][dataset])), average='macro', labels = list(['yes','no']))
    scores['pc'][dataset] = micro_f1, macro_f1
    return micro_f1, macro_f1

def evaluate_sc(dataset):
    micro_f1 = f1_score(*list(zip(*out_dict['sc'][dataset])), average='micro', labels = ['yes','no'])
    macro_f1 = f1_score(*list(zip(*out_dict['sc'][dataset])), average='macro', labels = ['yes','no'])
    scores['sc'][dataset] = micro_f1, macro_f1
    return micro_f1, macro_f1

evaluate_pc('glass_non_glass')
evaluate_sc('sofc_sent')
evaluate_squad()

df = pd.DataFrame.from_dict({(i,j): scores[i][j] for i in scores.keys() for j in scores[i].keys()},orient='index', columns = ['micro-f1/em', 'macro-f1/f1'])
df = df.apply(lambda x: 100 * round(x, 5))

print()
print(df.to_string())
print()