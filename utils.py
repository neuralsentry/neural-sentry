import os
import glob


import tqdm
import pandas as pd
from git import Repo, Commit
from transformers import AutoTokenizer, PreTrainedTokenizer

from constants import CLS_TOKEN, MASK_TOKEN, PAD_TOKEN, SEPARATOR_TOKEN, CODE_SEPARATOR


def tokenize_dataset_example(tokenizer: PreTrainedTokenizer, example: dict):
    formatted = example["formatted"]

    if not formatted:
        raise ValueError(
            'You must format the example before tokenizing it. (example["formatted"]])'
        )

    return tokenizer(text=formatted, truncation=True)


def prepare_starncoder_tokenizer(tokenizer_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_auth_token=True)

    tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    tokenizer.add_special_tokens({"sep_token": SEPARATOR_TOKEN})
    tokenizer.add_special_tokens({"cls_token": CLS_TOKEN})
    tokenizer.add_special_tokens({"mask_token": MASK_TOKEN})
    return tokenizer


def get_latest_checkpoint(checkpoint_dir: str):
    checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    checkpoint_dirs = [dir for dir in checkpoint_dirs if os.path.isdir(dir)]
    checkpoint_dirs = sorted(checkpoint_dirs)

    if len(checkpoint_dirs) == 0:
        return None
    return checkpoint_dirs[-1]


def concat_tokens_to_chunks(chunk_size: int):
    def apply(examples: dict):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # Drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size
        # Split by chunks of max_len
        result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column for masking
        result["labels"] = result["input_ids"].copy()
        return result

    return apply
