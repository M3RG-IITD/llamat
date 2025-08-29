"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import random
import argparse
import pandas as pd
import numpy as np

from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from llama_finetune import (
    get_crystal_string,   
    MAX_LENGTH
)
from cif_eval_util import make_swap_table
        
def unconditional_sample(args):
    model, tokenizer = prepare_model_and_tokenizer(args)

    prompts = []
    for _ in range(args.num_samples):
        prompt = "Below is a description of a bulk material. "
        prompt += (
            "Generate a description of the lengths and angles of the lattice vectors "
            "and then the element type and coordinates for each atom within the lattice:\n"
        )
        prompts.append(prompt)
 
    outputs = []
    while len(outputs) < args.num_samples:
        batch_prompts = prompts[len(outputs):len(outputs)+args.batch_size]

        batch = tokenizer(
            list(batch_prompts), 
            return_tensors="pt",
        )
        batch = {k: v.cuda() for k, v in batch.items()}

        generate_ids = model.generate(
            **batch,
            do_sample=True,
            max_new_tokens=500,
            temperature=args.temperature, 
            top_p=args.top_p, 
        )

        gen_strs = tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        for gen_str, prompt in zip(gen_strs, batch_prompts):
            material_str = gen_str.replace(prompt, "")

            try:
                cif_str = parse_fn(material_str)
                _ = Structure.from_str(cif_str, fmt="cif")
            except Exception as e:
                print(e)
                continue

            outputs.append({
                "gen_str": gen_str,
                "cif": cif_str,
                "model_name": args.model_name,
            })

    df = pd.DataFrame(outputs)
    df.to_csv(out_path, index=False)

condition_templates = {
    "pretty_formula": "The chemical formula is {pretty_formula}. ",
    "e_above_hull": "The energy above the convex hull is {e_above_hull}. ",
    "spacegroup.number": "The spacegroup number is {spacegroup.number}. ",
}

def conditional_sample(args):
    model, tokenizer = prepare_model_and_tokenizer(args)

    conditions_data = pd.read_csv(args.conditions_file)[
        ["e_above_hull", "pretty_formula", "spacegroup.number"]
    ].drop_duplicates()
    conditions_data = conditions_data.sample(args.num_samples, replace=False).to_dict(orient="records")

    conditions = args.conditions.split(",")

    prompts = []
    for d in conditions_data:
        prompt = "Below is a description of a bulk material. "
        for c in conditions:
            prompt += condition_templates[c].format(**d)

        prompt += (
            "Generate a description of the lengths and angles of the lattice vectors "
            "and then the element type and coordinates for each atom within the lattice:\n"
        )
        prompts.append(prompt)
 
    outputs = []
    while len(outputs) < args.num_samples:
        batch_prompts = prompts[len(outputs):len(outputs)+args.batch_size]
        batch_conditions = conditions[len(outputs):len(outputs)+args.batch_size]

        batch = tokenizer(
            list(batch_prompts), 
            return_tensors="pt",
        )
        batch = {k: v.cuda() for k, v in batch.items()}

        generate_ids = model.generate(
            **batch,
            do_sample=True,
            max_new_tokens=500,
            temperature=args.temperature, 
            top_p=args.top_p, 
        )

        gen_strs = tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        for gen_str, prompt, _conditions in zip(gen_strs, batch_prompts, batch_conditions):
            material_str = gen_str.replace(prompt, "")

            try:
                cif_str = parse_fn(material_str)
                _ = Structure.from_str(cif_str, fmt="cif") #double check valid cif string
            except Exception as e:
                print(e)
                continue

            sample = {
                "gen_str": gen_str,
                "cif": cif_str,
                "model_name": args.model_name,
            }
            sample.update(_conditions)
            outputs.append(sample)

    df = pd.DataFrame(outputs)
    df.to_csv(out_path, index=False)

def infill_sample(args, start_crystal_cif=None):
    swap_table = make_swap_table(0.1)

    outputs = []
    for i in range(0, args.num_samples, args.batch_size):
        batch_prompts = prompts[i:i+args.batch_size]
        species_to_remove_batch = species_to_remove_list[i:i+args.batch_size]
        masked_crystals = masked_crystal_strs[i:i+args.batch_size]

        batch = tokenizer(
            list(batch_prompts), 
            return_tensors="pt",
        )
        batch = {k: v.cuda() for k, v in batch.items()}

        possible_elems = [str(s) for s in swap_table[species_to_remove_batch[0]]]

        kwargs = {
            "model": "/scratch/cse/btech/cs1200448/MatLlama/ft-checkpoints/llamat_cifpt_iftcif_new/2559/hf",
            "tokenizer": "/scratch/cse/btech/cs1200448/MatLlama/ft-checkpoints/llamat_cifpt_iftcif_new/2559/hf",
            "trust_remote_code": True,
            "tensor_parallel_size": 1,
            "do_sample": True,
            "max_new_tokens": 10,
            "temperature": args.temperature,
            "top_p": args.top_p
        }

        if args.infill_do_constraint:
            kwargs["bad_words_ids"] = [tokenizer.encode(s) for s in possible_elems]

        generate_ids = model.generate(
            **batch,
            **kwargs,
        )

        gen_strs = tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        for gen_str, prompt, species_to_remove, masked_crystal in zip(
            gen_strs, batch_prompts, species_to_remove_batch, masked_crystals
        ):
            new_element = gen_str.replace(prompt, "").split("\n")[0]
            
            print(f"Swap {species_to_remove} with {new_element}")

            orig_crys_str = masked_crystal.replace("[MASK]", species_to_remove)
            new_crys_str = masked_crystal.replace("[MASK]", new_element)

            try:
                new_cif = parse_fn(new_crys_str)
                _ = Structure.from_str(new_cif, fmt="cif") #double check valid cif string
                original_cif = parse_fn(orig_crys_str)
            except Exception as e:
                print(e)
                continue

            sample = {
                "original_element": species_to_remove,
                "new_element": new_element,
                "original_crystal": original_cif,
                "new_crystal": new_cif,
                "model_name": args.model_name,
            }
            outputs.append(sample)

    df = pd.DataFrame(outputs)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out_path", type=str, default="llm_samples.csv")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--format_instruction_prompt", type=int, default=0)
    parser.add_argument("--format_response_format", type=int, default=0)
    parser.add_argument("--conditions", type=str, default="pretty_formula")
    parser.add_argument("--conditions_file", type=str, default="") #"data/with_tags/test.csv"
    parser.add_argument("--infill_file", type=str, default="") #"data/with_tags/test.csv"
    parser.add_argument("--infill_do_constraint", type=int, default=0)
    parser.add_argument("--infill_constraint_tolerance", type=float, default=0.1)
    args = parser.parse_args()

    if ".csv" in args.out_path:
        out_path = args.out_path
    else:
        i = os.environ.get("SLURM_ARRAY_TASK_ID", 0)
        out_path = os.path.join(args.out_path, f"samples_{i}.csv") 
        args.out_path = out_path

    if args.conditions_file:
        conditional_sample(args)
    elif args.infill_file:
        infill_sample(args)
    else:
        unconditional_sample(args)
