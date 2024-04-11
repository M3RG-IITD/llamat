import json
import logging
import argparse
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
# login('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')

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, help="Path to the checkpoint to run inference on")
parser.add_argument('--valfile', type=str)
parser.add_argument('--mem_util', type=float, default=0.9, help="how much gpu mem")
parser.add_argument('--num_gpu', type=int, default=-1, help="how many gpus")
parser.add_argument('--num_seeds', type=int, default=3, help="seed")
args = parser.parse_args()

if args.num_gpu == -1:
    args.num_gpu = torch.cuda.device_count()
    
valfile = json.load(open(args.valfile, 'r'))
valfile = [i for i in valfile if i['task'] != 'ner']

valinputs = [i['input'] for i in valfile if i['task'] != 'completion']

init_seed = 2
args.seed = 1

# see all variations
logging.info(f'Loaded tokenizer \n\tfrom checkpoint: {args.checkpoint}')
kwargs = {
    "model": args.checkpoint,
    "tokenizer": args.checkpoint,
    "trust_remote_code": True,
    "tensor_parallel_size": args.num_gpu,
    "seed":args.seed,
    "gpu_memory_utilization":args.mem_util,
}
client = vllm.LLM(**kwargs)

daddy_df = []
for seed in range(1,args.num_seeds+1):
    args.seed = args.seed * init_seed
    scores = {}
    for i in valfile:
        scores[i['task'], i['dataset']] = []
    
    response = client.generate(valinputs, sampling_params=vllm.SamplingParams(
            best_of=1,
            presence_penalty=0.0,
            frequency_penalty=1.0,
            top_k=50,
            top_p=1.0,
            temperature=0.75,
            # stop=stop_seq,
            use_beam_search=False,
            max_tokens=30,
            logprobs=2
        ))
    predictions = [i.outputs[0].text for i in response]
    
    samples = {}
    for _,i in enumerate(valfile):
        # remove the datapoints that have >=2048 tokens.
        if len(response[_].prompt_token_ids + response[_].outputs[0].token_ids) > 2048:
            continue
        i['prediction'] = predictions[_].split('\n')[0].strip().lower()
        if (i['task'], i['dataset']) not in samples:
            samples[(i['task'], i['dataset'])] = [i]
        else:
            samples[(i['task'], i['dataset'])].append(i)
    
    #====inference done, now evaluation====
    
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
    
    def get_raw_scores(dataset):
        exact_scores = {}
        f1_scores = {}
        for article in dataset:
            qid = article['qid']
            gold_answers = [a for a in article['valid_answers'] if normalize_answer(a)]
            if not gold_answers:
                print("sad")
                gold_answers = ['']
            a_pred = article['prediction']
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
    
    def evaluate(key, examples):
        if key == ('re', 'sc_comics'):
            answer_set = ['condition', 'equivalent', 'target'] 
        elif key == ('re', 'structured_re'):
            answer_set = ['coulombic efficiency', 'capacity', 'conductivity', 'voltage', 'energy'] 
        elif key == ('sc', 'sofc_sent'):
            answer_set = ['yes', 'no']
        elif key == ('pc', 'glass_non_glass'):
            answer_set = ['yes', 'no']
        elif key == ('sar', 'synthesis_actions'):
            answer_set = ['cooling', 'heating', 'mixing', 'non-altering', 'purification', 'reaction', 'shaping', 'starting']
        elif key == ('ee', 'sc_comics'):
            answer_set = ['site', 'dopant']
        else:
            assert 1 == 0
        
        gt = []
        pred = []
        invalid = 0
        for example in examples:
            cp = example['prediction']
            cg = example['output']
            ans_found = 0
            for ans in answer_set:
                if ans in cp and ans_found == 0:
                    ans_found = 1
                    pred.append(ans)
                    gt.append(cg)
                elif ans in cp and ans_found == 1:
                    ans_found = 0
                    pred.pop()
                    gt.pop()
                    break
            if ans_found == 0:
                invalid += 1
                
        micro_f1 = f1_score(gt, pred, average='micro', labels = answer_set)
        macro_f1 = f1_score(gt, pred, average='macro', labels = answer_set)
        scores[key] = {'micro_f1': micro_f1, 'macro_f1': macro_f1, 'ans_not_found': invalid / len(examples)}
    
    # update this
    def evaluate_ner(key, examples):
        if key == ('ner', 'sc_comics'):
            answer_set = None 
        else:
            assert 1 == 0
        
        gt = []
        pred = []
        invalid = 0
        for example in examples:
            cp = example['prediction']
            cg = example['output']
            ans_found = 0
            for ans in answer_set:
                if ans in cp and ans_found == 0:
                    ans_found = 1
                    pred.append(ans)
                    gt.append(cg)
                elif ans in cp and ans_found == 1:
                    ans_found = 0
                    pred.pop()
                    gt.pop()
                    break
            if ans_found == 0:
                invalid += 1
                
        micro_f1 = f1_score(gt, pred, average='micro', labels = answer_set)
        macro_f1 = f1_score(gt, pred, average='macro', labels = answer_set)
        scores[key] = {'micro_f1': micro_f1, 'macro_f1': macro_f1, 'ans_not_found': invalid / len(examples)}
    
    for key,examples in samples.items():
        if key == ('qna', 'squadv2'):
            exact_raw, f1_raw = get_raw_scores(examples)
            out_eval = make_eval_dict(exact_raw, f1_raw)
            scores[('qna','squadv2')] = {'em': out_eval['exact'], 'f1':out_eval['f1']}
        elif key[0] not in ['ner']:
            evaluate(key, examples)
        elif key[0] in ['ner']:
            pass
            # evaluate_ner(key, examples)
        else:
            assert 1 == 0
    
    # scores.pop(('ner', 'sc_comics'))
    df = pd.DataFrame(scores).apply(lambda x : 100 * round(x,5))
    # df = df.fillna('-')
    print(df.to_string())
    daddy_df.append(df)

df_sum = reduce(lambda x, y: x.add(y, fill_value=0), daddy_df)
df_mean = df_sum / len(daddy_df)
df_concat = pd.concat(daddy_df)
df_std = df_concat.groupby(df_concat.index).std()

df_mean.columns = pd.MultiIndex.from_tuples([(i[0], f"{i[1]}_mean") for i in df_mean.columns])
df_std.columns = pd.MultiIndex.from_tuples([(i[0], f"{i[1]}_std") for i in df_std.columns])

combined_df = pd.concat([df_mean, df_std], axis=1)
combined_df = combined_df.sort_index(axis=1).apply(lambda x : round(x,5))


print(f"===eval over average of {args.num_seeds} runs===")
print(combined_df.to_string())

prefix = args.checkpoint.split('/')[-1]

now = datetime.datetime.now()
datetimestring = now.strftime("%y-%m-%d-%H-%M")

combined_df.to_csv(f'csvs/{prefix}-{datetimestring}.csv')