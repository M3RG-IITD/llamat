import re
import os
import time
import math

from pathlib import Path
from typing import Optional
from subprocess import Popen, PIPE
from argparse import ArgumentParser, Namespace

DEFAULT_EPOCHS = 3
DEFAULT_SEQ = 2048

def execute(cmd: list[str]):
    with Popen(cmd) as proc:
        assert proc.wait() == 0

def get_parallel_levels():
    return 1, 4

def infer_ndocs(cmd: list[str], args) -> int:
    return args.numdocs
    # with open(args.data, 'r') as f:
    #     return (len(f.readlines()))
    
    # return 19942 #downstream train
    # return 170594 #downstream val
    # return 43841 #ift train
    # return 2307 #ift val
    # return 10581 #downstream old

def finetune(args: Namespace):
    load_from = args.checkpoint
    latest_txt = os.path.join(load_from, "latest_checkpointed_iteration.txt")

    tp, pp = get_parallel_levels()
    model_name = "llama2"

    cmd = ["bash", "./ft_sft.sh", model_name, "--micro-batch",
           args.micro_batch, "--global-batch", "64", "--tp", tp, "--pp", pp, "--seq-len",
           args.seq, "--checkpoint", load_from,
           "--out", args.save_checkpoint_dir, "--loss-mask", 0.0, "--save-interval", args.save_interval]

    cmd = list(map(str, cmd))
    n_docs = infer_ndocs(cmd, args)
    n_iters = args.epochs*n_docs/64
    print(n_docs, n_iters)
    n_iters = 10*int(math.ceil(n_iters/10))  # to make it a multiple of 10 xd
    if args.load_iters != 'release':
        cmd += ["--it", args.load_iters]
    cmd += ["--iters", n_iters]
    cmd += ["--data", args.traindata]
    cmd += ["--val-path", args.valdata]

    if args.nodes > 1:
        cmd += ["--nodes", args.nodes, "--rank", args.rank, "--addr", args.addr]

    # execute command
    print("Finetuning")
    cmd = list(map(str, cmd))
    execute(cmd)
    return

def main(args: Namespace):
    finetune(args)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--traindata")
    parser.add_argument("--valdata")
    parser.add_argument("--numdocs", type=int)
    parser.add_argument("--checkpoint", help="Name of the model to finetune")
    parser.add_argument("--load_iters", help="Which iteration to finetune")
    parser.add_argument("--save_checkpoint_dir", type=str, help="Directory to save the trained checkpoint")
    parser.add_argument("--size", default=7, choices=[7, 13, 70], type=int, help="Size of the model to finetune")
    parser.add_argument("--micro_batch", type=int, default=2, help="Micro batch size")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Epochs to train for")
    parser.add_argument("--seq", type=int, default=DEFAULT_SEQ, help="Sequence length")
    parser.add_argument("--rank", type=int, default=0, help="Rank")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--addr", help="Master addr")
    parser.add_argument("--save_interval", type=int, default=200)
    args = parser.parse_args()
    main(args)
