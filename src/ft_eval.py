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
    
valfile = []
with open(args.valfile, 'r') as f:
    valfile = [json.loads(line) for line in f.readlines()]
    
# valfile = json.load(open(args.valfile, 'r'))
# valfile = [i for i in valfile if i['task'] != 'ner']

valfile = valfile

valinputs = [f"<|im_start|>system\n{i['system']}<|im_end|>\n"+f"<|im_start|>question\n{i['question']}<|im_end|>\n"+"<|im_start|>answer\n" for i in valfile]

# print(valinputs[0])

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

    tasks = ["ner","pc","sf","ee","re","sar","sc"]
    kw = {i : [] for i in tasks}
    kw["ner"] = {i : [] for i in ["matscholar", "sofc_token", "sc_comics"]}
    kw["pc"] = {i : [] for i in ["glass_non_glass"]}
    kw["sf"] = {i : [] for i in ["sofc_token"]}
    kw["ee"] = {i : [] for i in ["sc_comics"]}
    kw["re"] = {i : [] for i in ["structured_re", "sc_comics"]}
    kw["sar"] = {i : [] for i in ["synthesis_actions"]}
    kw["sc"] = {i : [] for i in ["sofc_sent"]}
    kw["mcq"] = {i : [] for i in ["story_cloze", "hellaswag", "boolqa"]}
    
    kw['ner']['matscholar'] = ['b-mat','i-mat','b-pro','i-pro','b-dsc','i-dsc','b-spl','i-spl','b-apl','i-apl','b-smt','i-smt','b-cmt','i-cmt']
    kw['ner']['sofc_token'] = ['b-material', 'i-material', 'b-device', 'i-device', 'b-experiment', 'i-experiment', 'b-value', 'i-value']
    kw['ner']['sc_comics'] = ['material', 'doping', 'sc', 'value', 'process', 'characterization', 'element', 'property', 'main']
    kw['pc']['glass_non_glass'] = ['yes','no']
    kw['sf']['sofc_token'] = ['i-device', 'b-voltage', 'b-anode_material', 'b-cathode_material', 'b-time_of_operation', 'i-working_temperature', 'b-conductivity', 'i-fuel_used', 'i-interlayer_material', 'i-time_of_operation', 'i-anode_material', 'i-current_density', 'b-degradation_rate', 'i-resistance', 'i-conductivity', 'b-current_density', 'b-working_temperature', 'i-thickness', 'i-experiment_evoking_word', 'b-open_circuit_voltage', 'i-degradation_rate', 'b-electrolyte_material', 'i-open_circuit_voltage', 'i-electrolyte_material', 'b-fuel_used', 'b-power_density', 'i-power_density', 'b-interlayer_material', 'b-thickness', 'b-device', 'b-experiment_evoking_word', 'i-cathode_material', 'b-resistance', 'i-support_material', 'i-voltage', 'b-support_material']
    kw['ee']['sc_comics'] = ['site', 'dopant']
    kw['re']['structured_re'] = ['capacity','voltage','coulombic efficiency','conductivity','energy']
    kw['re']['sc_comics'] = ['target', 'condition', 'equivalent']
    kw['sar']['synthesis_actions'] = ['cooling', 'heating', 'mixing', 'non-altering', 'purification', 'reaction', 'shaping', 'starting']
    kw['sc']['sofc_sent'] = ['yes','no']
    
    kw['mcq']['hellaswag'] = ['a','b','c','d']
    kw['mcq']['boolqa'] = ['yes','no']
    kw['mcq']['story_cloze'] = ['a','b']

    def most_similar_answer(a,answer_set):
        a = a.strip().replace(' ', '')
        if(a in answer_set):
            return a
        dis = [Levenshtein.distance(a,x) for x in answer_set]
        idx = np.argmin(dis)
        return answer_set[idx]

    out_dict = {i : [] for i in tasks}
    out_dict["ner"] = {i : [] for i in ["matscholar", "sofc_token", "sc_comics"]}
    out_dict["pc"] = {i : [] for i in ["glass_non_glass"]}
    out_dict["sf"] = {i : [] for i in ["sofc_token"]}
    out_dict["ee"] = {i : [] for i in ["sc_comics"]}
    out_dict["re"] = {i : [] for i in ["structured_re", "sc_comics"]}
    out_dict["sar"] = {i : [] for i in ["synthesis_actions"]}
    out_dict["sc"] = {i : [] for i in ["sofc_sent"]}
    out_dict["qna"] = {i : [] for i in ["squad"]}
    out_dict["mcq"] = {i : [] for i in ["hellaswag", "boolqa", "story_cloze"]}

    for _,sent in enumerate(valfile):
        task = None
        dataset = None
        # task = sent['task']
        # dataset = sent['dataset']
        system = sent['system']
        output = predictions[_]
        if len(output) == 0: # cases with prompt length > ctxlen
            continue
        answer = sent['answer']
        
        if 'SOFC' in system:
            task = 'sc'
            dataset = 'sofc_sent'
        elif 'named entity' in system:
            task = 'ner'
            if 'b-dsc' in system:
                dataset = 'matscholar'
            elif 'b-device' in system:
                dataset = 'sofc_token'
            elif 'doping' in system:
                dataset = 'sc_comics'
        elif 'synthesis action' in system:
            task = 'sar'
            dataset = 'synthesis_actions'
        elif 'slots' in system:
            task = 'sf'
            dataset = 'sofc_token'
        elif 'event, identify the roles of the arguments' in system:
            task = 'ee'
            dataset = 'sc_comics'
        elif 'extract relation' in system:
            task = 're'
            for ds in kw[task]:
                if kw[task][ds][0] in system:
                    dataset = ds
                    break   
        elif 'inorganic glass' in system:
            task = 'pc'
            dataset = 'glass_non_glass'
        elif 'option most likely to be correct' in system:
            task = 'mcq'
            dataset = 'hellaswag'
        elif 'You are a linguist with reasoning abilities. Read the passage given below and using its information determine whether the answer to the question that follows it is Yes or No. You should output exactly one of Yes or No and nothing else.' in system:
            task = 'mcq'
            dataset = 'boolqa'
        elif 'story' in system:
            task = 'mcq'
            dataset = 'story_cloze'
        else:
            task = 'qna'
            dataset = 'squad'
        
        assert task != None
        assert dataset != None
        
        if task in ["ner", "sf", "ee", "sar"]:
            answer = [i.split(" : ") for i in answer.lower().split('\n')]
            try:
                temp = [[j.strip().lower() for j in i.split(":")] for i in output.split('\n')]
                temp = [i for i in temp if len(i) == 2]
                output = temp
            except:
                print(f"for tc {_}, output pattern mismatched. {output}")
        else:
            answer = answer.strip().lower()
            output = output.strip().lower()
    
        out_dict[task][dataset].append((answer, output))

    scores = {i : [] for i in tasks}
    scores["ner"] = {i : (0,0) for i in ["matscholar", "sofc_token", "sc_comics"]}
    scores["pc"] = {i : (0,0) for i in ["glass_non_glass"]}
    scores["sf"] = {i : (0,0) for i in ["sofc_token"]}
    scores["ee"] = {i : (0,0) for i in ["sc_comics"]}
    scores["re"] = {i : (0,0) for i in ["structured_re", "sc_comics"]}
    scores["sar"] = {i : (0,0) for i in ["synthesis_actions"]}
    scores["sc"] = {i : (0,0) for i in ["sofc_sent"]}
    scores["qna"] = {i : (0,0) for i in ["squad"]}
    scores["mcq"] = {i : (0,0) for i in ["hellaswag", "boolqa", "story_cloze"]}

    def evaluate(task, dataset):
        all_gt = []
        all_pred = []
        for gt, pred in out_dict[task][dataset]:
            dict = {}
            for word, entity in gt:
                dict[word] = [entity, 'O']
                    
            for word, entity in pred:
                to_put = most_similar_answer(entity, kw[task][dataset])
                if word in dict:
                    dict[word][1] = to_put
                else:
                    dict[word] = ['O', to_put]
                    
            for i in dict.values():
                all_gt.append(i[0])
                all_pred.append(i[1])
    
        micro_f1 = f1_score(all_gt, all_pred, average='micro', labels = list(kw[task][dataset]))
        macro_f1 = f1_score(all_gt, all_pred, average='macro', labels = list(kw[task][dataset]))
        scores[task][dataset] = micro_f1, macro_f1
        return micro_f1, macro_f1

    def evaluate_pc(dataset):
        out_dict['pc'][dataset] = [(most_similar_answer(i[0], kw['pc'][dataset]), most_similar_answer(i[1], kw['pc'][dataset])) for i in out_dict['pc'][dataset]]
    
        micro_f1 = f1_score(*list(zip(*out_dict['pc'][dataset])), average='micro', labels = list(kw['pc'][dataset]))
        macro_f1 = f1_score(*list(zip(*out_dict['pc'][dataset])), average='macro', labels = list(kw['pc'][dataset]))
        scores['pc'][dataset] = micro_f1, macro_f1
        return micro_f1, macro_f1
    
    def evaluate_sc(dataset):
        out_dict['sc'][dataset] = [(most_similar_answer(i[0], kw['sc'][dataset]), most_similar_answer(i[1], kw['sc'][dataset])) for i in out_dict['sc'][dataset]]
        
        micro_f1 = f1_score(*list(zip(*out_dict['sc'][dataset])), average='micro', labels = list(kw['sc'][dataset]))
        macro_f1 = f1_score(*list(zip(*out_dict['sc'][dataset])), average='macro', labels = list(kw['sc'][dataset]))
        scores['sc'][dataset] = micro_f1, macro_f1
        return micro_f1, macro_f1
    
    def evaluate_re(dataset):
        out_dict['re'][dataset] = [(most_similar_answer(i[0], kw['re'][dataset]), most_similar_answer(i[1], kw['re'][dataset])) for i in out_dict['re'][dataset]]
        
        micro_f1 = f1_score(*list(zip(*out_dict['re'][dataset])), average='micro', labels = list(kw['re'][dataset]))
        macro_f1 = f1_score(*list(zip(*out_dict['re'][dataset])), average='macro', labels = list(kw['re'][dataset]))
        scores['re'][dataset] = micro_f1, macro_f1
        return micro_f1, macro_f1

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
        exact_scores = []
        f1_scores = []
        for article in dataset:
            gold_answers = [article[0] if normalize_answer(article[0]) else None]
            if not gold_answers[0]:
                print("sad")
                gold_answers = ['']
            a_pred = article[1]
            exact_scores.append(max(compute_exact(a, a_pred) for a in gold_answers))
            f1_scores.append(max(compute_f1(a, a_pred) for a in gold_answers))
        return exact_scores, f1_scores
    
    def make_eval_dict(exact_scores, f1_scores):
        total = len(exact_scores)
        assert len(exact_scores) == len(f1_scores)
        return collections.OrderedDict([
            ('exact', 1.0 * sum(k for k in exact_scores) / total),
            ('f1', 1.0 * sum(k for k in f1_scores) / total),
            ('total', total),
        ])
    
    def most_similar_answer(a,answer_set):
        a = a.strip().replace(' ', '')
        if(a in answer_set):
            return a
        dis = [Levenshtein.distance(a,x) for x in answer_set]
        idx = np.argmin(dis)
        return answer_set[idx]
    
    def evaluate_qna(dataset):
        exact_raw, f1_raw = get_raw_scores(out_dict['qna'][dataset])
        out_eval = make_eval_dict(exact_raw, f1_raw)
        scores['qna'][dataset] = out_eval['f1'], out_eval['exact']
        
        return out_eval

    def evaluate_mcq(dataset):
        out_dict['mcq'][dataset] = [(most_similar_answer(i[0], kw['mcq'][dataset]), most_similar_answer(i[1], kw['mcq'][dataset])) for i in out_dict['mcq'][dataset]]
        micro_f1 = f1_score(*list(zip(*out_dict['mcq'][dataset])), average='micro', labels = list(kw['mcq'][dataset]))
        macro_f1 = f1_score(*list(zip(*out_dict['mcq'][dataset])), average='macro', labels = list(kw['mcq'][dataset]))
        scores['mcq'][dataset] = micro_f1, macro_f1
        return micro_f1, macro_f1

    if len(out_dict['qna']['squad']) > 0:
        evaluate_qna('squad')
    if len(out_dict['ner']['matscholar']) > 0:
        evaluate('ner', 'matscholar')
    if len(out_dict['ner']['sofc_token']) > 0:
        evaluate('ner', 'sofc_token')
    if len(out_dict['ner']['sc_comics']) > 0:
        evaluate('ner', 'sc_comics')
    if len(out_dict['sar']['synthesis_actions']) > 0:
        evaluate('sar', 'synthesis_actions')
    if len(out_dict['ee']['sc_comics']) > 0:
        evaluate('ee', 'sc_comics')
    if len(out_dict['sf']['sofc_token']) > 0:
        evaluate('sf', 'sofc_token')
    if len(out_dict['sc']['sofc_sent']) > 0:
        evaluate_sc('sofc_sent')
    if len(out_dict['pc']['glass_non_glass']) > 0:
        evaluate_pc('glass_non_glass')
    if len(out_dict['re']['structured_re']) > 0:
        evaluate_re('structured_re')
    if len(out_dict['re']['sc_comics']) > 0:
        evaluate_re('sc_comics')
    if len(out_dict['mcq']['hellaswag']) > 0:
        evaluate_mcq('hellaswag')
    if len(out_dict['mcq']['boolqa']) > 0:
        evaluate_mcq('boolqa')
    if len(out_dict['mcq']['story_cloze']) > 0:
        evaluate_mcq('story_cloze')

    df = pd.DataFrame.from_dict({(i,j): scores[i][j] for i in scores.keys() for j in scores[i].keys()},orient='index', columns = ['micro-f1/f1', 'macro-f1/em'])
    df = df.apply(lambda x: 100 * round(x, 5))

    # scores.pop(('ner', 'sc_comics'))
    # df = pd.DataFrame(scores).apply(lambda x : 100 * round(x,5))
    # df = df.fillna('-')
    print(df.to_string())
    daddy_df.append(df)

df_sum = reduce(lambda x, y: x.add(y, fill_value=0), daddy_df)
df_mean = df_sum / len(daddy_df)
df_concat = pd.concat(daddy_df)
df_std = df_concat.groupby(df_concat.index).std()

df_mean.columns = pd.MultiIndex.from_tuples([(i, f"mean") for i in df_mean.columns])
df_std.columns = pd.MultiIndex.from_tuples([(i, f"std") for i in df_std.columns])

combined_df = pd.concat([df_mean, df_std], axis=1)
combined_df = combined_df.sort_index(axis=1).apply(lambda x : round(x,2))


print(f"===eval over average of {args.num_seeds} runs===")
print(combined_df.to_string())

prefix = args.checkpoint.split('/')[-3] + '-' + args.checkpoint.split('/')[-2]

now = datetime.datetime.now()
datetimestring = now.strftime("%y-%m-%d-%H-%M")

combined_df.to_csv(f'/home/cse/btech/cs1200448/MatLlama/scripts/csvs/{prefix}-{datetimestring}.csv')

# df_sum = reduce(lambda x, y: x.add(y, fill_value=0), daddy_df)
# df_mean = df_sum / len(daddy_df)
# df_concat = pd.concat(daddy_df)
# df_std = df_concat.groupby(df_concat.index).std()

# df_mean.columns = pd.MultiIndex.from_tuples([(i[0], f"{i[1]}_mean") for i in df_mean.columns])
# df_std.columns = pd.MultiIndex.from_tuples([(i[0], f"{i[1]}_std") for i in df_std.columns])

# combined_df = pd.concat([df_mean, df_std], axis=1)
# combined_df = combined_df.sort_index(axis=1).apply(lambda x : round(x,5))


# print(f"===eval over average of {args.num_seeds} runs===")
# print(combined_df.to_string())

# prefix = args.checkpoint.split('/')[-1]

# now = datetime.datetime.now()
# datetimestring = now.strftime("%y-%m-%d-%H-%M")

# combined_df.to_csv(f'csvs/{prefix}-{datetimestring}.csv')
