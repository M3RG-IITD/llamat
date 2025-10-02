import time

# for llamat-2-cif:
model_name = '/scratch/civil/phd/cez188393/zaki_epcc/checkpoints_llamat_cit/checkpoint_17000_to_hf'
model_path = "/scratch/cse/btech/cs1210556.deleted/crystal-text-llm/exp/llamat-cif-7b-dhruv/checkpoint-33920"
model_name = '/scratch/civil/phd/cez188393/zaki_epcc/checkpoints_llamat3_cif/checkpoint_15000_to_hf'
model_path = '/home/scai/phd/aiz218326/github/crystal-text-llm/exp/8b-test-run/checkpoint-33920'
outname = "llamat3_seed_1_1000_congen"

batch_size = 64

out_path = "_cif_out_llamat2_seed_1_1000_congen.csv"
temperature = 0.9
top_p = 0.9
format_instruction_prompt = 0

format_response_format = 0

conditions = "pretty_formula"

conditions_file = "data/with_tags/test.csv"
infill_file = "" #, type=str, default="") #"data/with_tags/test.csv"
infill_do_constraint = 0
infill_constraint_tolerance = 0.1

from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice


import os
import random
import argparse
import pandas as pd
import numpy as np

from transformers import (
    LlamaForCausalLM, LlamaTokenizer
)
# from peft import PeftModel


from llama_finetune import (
    get_crystal_string,   
    MAX_LENGTH
)
# from templating import make_swap_table

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def parse_fn(gen_str):
    lines = [x for x in gen_str.split("\n") if len(x) > 0]
    lengths = [float(x) for x in lines[0].split(" ")]
    angles = [float(x) for x in lines[1].split(" ")]
    species = [x for x in lines[2::2]]
    coords = [[float(y) for y in x.split(" ")] for x in lines[3::2]]
    
    structure = Structure(
        lattice=Lattice.from_parameters(
            *(lengths + angles)),
        species=species,
        coords=coords, 
        coords_are_cartesian=False,
    )
    
    return structure.to(fmt="cif")

from peft import PeftModel


def make_swap_table(tolerance):
    # Placeholder to avoid linter undefined warnings; replace with real import if needed
    return {}


# Removed top-level model print/initialization to avoid side effects


def prepare_model_and_tokenizer(model_name,model_path):
    # llama_options = model_name.split("-")
    # is_chat = len(llama_options) == 2
    # model_size = llama_options[0]

    # def llama2_model_string(model_size, chat):
    #     chat = "chat-" if chat else ""
    #     return f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"

    # model_string = llama2_model_string(model_size, is_chat)
    model_string = model_name
    model = LlamaForCausalLM.from_pretrained(
        model_string,
        load_in_8bit=True,
        device_map="auto",
    )

    print('model loaded from', model_string)
    
    tokenizer = LlamaTokenizer.from_pretrained(
        model_string,
        model_max_length=MAX_LENGTH,
        padding_side="right",
        use_fast=False,
    )

    print('tokenizer loaded from', model_string)
    
    model.eval()

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        llama_tokenizer=tokenizer,
        model=model,
    )

    model = PeftModel.from_pretrained(model, model_path, device_map="auto")
    
    return model, tokenizer


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict, 
    llama_tokenizer, 
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = llama_tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(llama_tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def unconditional_sample(args):
    base_model = getattr(args, "model_name", None) or model_name
    lora_path = getattr(args, "model_path", None) or model_path
    model, tokenizer = prepare_model_and_tokenizer(base_model, lora_path)

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
    out_csv = getattr(args, "out_csv", "unconditional_outputs.csv")
    df.to_csv(out_csv, index=False)


condition_templates = {
    "pretty_formula": "The chemical formula is {pretty_formula}. ",
    "e_above_hull": "The energy above the convex hull is {e_above_hull}. ",
    "spacegroup_number": "The spacegroup number is {spacegroup_number}. ",
}
# Prompts will be built at runtime from the input CSV in conditional_sample

def conditional_sample(args):
    # Prepare model using args (fall back to module-level defaults if not set)
    base_model = getattr(args, "model_name", None) or model_name
    lora_path = getattr(args, "model_path", None) or model_path
    model, tokenizer = prepare_model_and_tokenizer(base_model, lora_path)

    # Load and normalize condition data
    df = pd.read_csv(args.conditions_file)
    if "spacegroup.number" in df.columns and "spacegroup_number" not in df.columns:
        df = df.rename(columns={"spacegroup.number": "spacegroup_number"})

    conditions = [c.strip() for c in args.conditions.split(",") if len(c.strip()) > 0]
    missing = [c for c in conditions if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns in conditions_file: {missing}")

    df = df[conditions].drop_duplicates()
    all_records = df.to_dict(orient="records")

    # Bound num_samples to available records
    total = len(all_records)
    num_samples = min(args.num_samples, total)
    records = all_records[:num_samples]

    # Build prompts
    prompts = []
    for d in records:
        prompt = "Below is a description of a bulk material. "
        for c in conditions:
            prompt += condition_templates[c].format(**d)
        prompt += (
            "Generate a description of the lengths and angles of the lattice vectors "
            "and then the element type and coordinates for each atom within the lattice:\n"
        )
        prompts.append(prompt)

    outputs = []
    # Save frequency for periodic checkpointing
    save_freq = max(1, getattr(args, "save_freq", 50))
    try:
        import pickle as _pickle  # ensure available
    except Exception:
        _pickle = None

    while len(outputs) < num_samples:
        batch_prompts = prompts[len(outputs):len(outputs)+args.batch_size]
        batch_conditions = records[len(outputs):len(outputs)+args.batch_size]

        batch = tokenizer(
            list(batch_prompts), 
            return_tensors="pt", padding=True, truncation=True
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

        for gen_str, prompt, _cond in zip(gen_strs, batch_prompts, batch_conditions):
            material_str = gen_str.replace(prompt, "")
            sample = {
                "gen_str": material_str,
                "prompt": prompt,
                "model_name": model_name,
            }
            sample.update(_cond)
            outputs.append(sample)

        # Periodic saving
        if len(outputs) % save_freq == 0 or len(outputs) == num_samples:
            df_out = pd.DataFrame(outputs)
            out_csv = getattr(args, "out_csv", "conditional_outputs.csv")
            out_pkl = getattr(args, "out_pkl", "conditional_outputs.pkl")
            df_out.to_csv(out_csv, index=False)
            if _pickle is not None:
                with open(out_pkl, "wb") as f:
                    _pickle.dump(outputs, f)


def infill_sample(args, start_crystal_cif=None):
    base_model = getattr(args, "model_name", None) or model_name
    lora_path = getattr(args, "model_path", None) or model_path
    model, tokenizer = prepare_model_and_tokenizer(base_model, lora_path)

    if start_crystal_cif is None:
        df = pd.read_csv(args.infill_file)
        idx = np.random.randint(len(df))
        start_crystal_cif = df['cif_str'][idx]

    print("Start crystal cif:")
    print(start_crystal_cif)

    prompts = []
    species_to_remove_list = []
    masked_crystal_strs = []
    for _ in range(args.num_samples):

        prompt = (
            'Below is a partial description of a bulk material where one '
            'element has been replaced with the string "[MASK]":\n'
        )

        structure = Structure.from_str(start_crystal_cif, fmt="cif")
        species = [str(s) for s in structure.species]
        species_to_remove = random.choice(species)
        species_to_remove_list.append(species_to_remove)

        crystal_string = get_crystal_string(start_crystal_cif)

        partial_crystal_str = crystal_string.replace(
            species_to_remove, "[MASK]"
        )
        masked_crystal_strs.append(partial_crystal_str)

        prompt = prompt + partial_crystal_str + "\n"

        prompt += (
            "Generate an element that could replace [MASK] in the bulk material:\n"
        )

        prompts.append(prompt)
 
    assert args.batch_size == 1, "Batch size must be 1 for infill sampling"

    swap_table = make_swap_table(args.infill_constraint_tolerance)

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
            "do_sample": True,
            "max_new_tokens": 10,
            "temperature": args.temperature,
            "top_p": args.top_p,
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
    out_csv = getattr(args, "out_csv", "infill_outputs.csv")
    df.to_csv(out_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="Generate crystal structure text with optional conditions")
    # Model selection and overrides
    parser.add_argument("--model_version", type=str, choices=["llamat2", "llamat3"], default="llamat3")
    parser.add_argument("--model_name", type=str, default="", help="HF model path; overrides preset")
    parser.add_argument("--model_path", type=str, default="", help="PEFT/LoRA checkpoint; overrides preset")

    # Data and prompts
    parser.add_argument("--conditions_file", default="data/with_tags/test.csv", type=str, required=True, help="CSV file with prompt conditions")
    parser.add_argument("--conditions", type=str, default="pretty_formula,e_above_hull,spacegroup_number",
                        help="Comma-separated condition columns to include")

    # Generation parameters
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--out_prefix", type=str, default="llamat_gen")
    parser.add_argument("--example", action="store_true", help="Run a tiny example (8 samples)")

    args = parser.parse_args()

    # Presets if not provided
    if args.model_version == "llamat2":
        preset_model_name = '/scratch/civil/phd/cez188393/zaki_epcc/checkpoints_llamat_cit/checkpoint_17000_to_hf'
        preset_model_path = "/scratch/cse/btech/cs1210556.deleted/crystal-text-llm/exp/llamat-cif-7b-dhruv/checkpoint-33920"
    else:
        preset_model_name = '/scratch/civil/phd/cez188393/zaki_epcc/checkpoints_llamat3_cif/checkpoint_15000_to_hf'
        preset_model_path = '/home/scai/phd/aiz218326/github/crystal-text-llm/exp/8b-test-run/checkpoint-33920'

    args.model_name = args.model_name or preset_model_name
    args.model_path = args.model_path or preset_model_path

    # Outputs
    ts = int(time.time())
    args.out_csv = f"{args.out_prefix}_{args.model_version}_{ts}.csv"
    args.out_pkl = f"{args.out_prefix}_{args.model_version}_{ts}.pkl"

    # Example mode reduces samples
    if args.example:
        args.num_samples = min(8, args.num_samples)

    # Seed
    try:
        import transformers
        transformers.set_seed(1)
    except Exception:
        pass
    random.seed(1)
    np.random.seed(1)

    # Run conditional generation
    conditional_sample(args)
    print(f"Saved outputs to: {args.out_csv} and {args.out_pkl}")


if __name__ == "__main__":
    main()