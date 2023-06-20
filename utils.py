import os
import glob


import tqdm
import pandas as pd
from git import Repo, Commit
from transformers import AutoTokenizer, PreTrainedTokenizer

from constants import CLS_TOKEN, MASK_TOKEN, PAD_TOKEN, SEPARATOR_TOKEN, CODE_SEPARATOR


def clone_or_pull_repo(remote_url: str):
    repo_name = remote_url.split("/")[-1]
    owner_name = remote_url.split("/")[-2]
    destination = f"data/repos/{owner_name}/{repo_name}"

    repo = None
    if os.path.exists(destination):
        repo = Repo(destination)
        print(f"Pulling {repo_name}")
        repo.remotes.origin.pull()
        print(f"[DONE] Pulling {repo_name}")
    else:
        print(f"Cloning {repo_name}")
        repo = Repo.clone_from(remote_url, destination)
        print(f"[DONE] Cloning {repo_name}")

    return repo


def get_file_paths_of_commit(commit: Commit):
    parent = commit.parents[0] if commit.parents else None

    if parent:
        diffs = commit.diff(parent)

        return {
            "new": [diff.b_path for diff in diffs if diff.change_type == "A"],
            "modified": [
                (diff.a_path, diff.b_path) for diff in diffs if diff.change_type == "M"
            ],
            "deleted": [diff.a_path for diff in diffs if diff.change_type == "D"],
            "renamed": [
                (diff.a_path, diff.b_path) for diff in diffs if diff.change_type == "R"
            ],
        }
    else:
        # This is the first commit in the repo so all files are considered new
        return {
            "new": [
                item.path for item in commit.tree.traverse() if item.type == "blob"
            ],
            "modified": [],
            "deleted": [],
            "renamed": [],
        }


def get_file_contents_at_commit(commit: Commit, path: str):
    return commit.tree[path].data_stream.read().decode("utf-8", errors="ignore")


def format_code(code: str, filename: str = None):
    formatted = ""
    if filename:
        formatted += f"<filename>{filename}"
    formatted += f"\n{code}"
    return formatted


def export_repo_to_csv(repo, destination: str):
    write_header = not os.path.isfile(destination) or os.path.getsize(destination) == 0

    for commit in repo.iter_commits():
        add_commit_to_csv(commit, destination, write_header)
        write_header = False


def add_commit_to_csv(
    commit: Commit,
    destination: str,
    write_header: bool = False,
    include_files: bool = True,
):
    file_paths = get_file_paths_of_commit(commit)
    modified_file_paths = file_paths["modified"]

    parent = commit.parents[0] if commit.parents else None

    if not parent:
        return

    output = {
        "commit_msg": commit.message,
        "sha": commit.hexsha,
        "remote_url": commit.repo.remotes.origin.url,
        "date": commit.authored_datetime,
        "labels": -1,
    }

    if include_files:
        commit_before = ""
        commit_after = ""
        for i, path in enumerate(modified_file_paths):
            commit_before += format_code(
                get_file_contents_at_commit(commit.parents[0], path[0]), path[0]
            )
            commit_after += format_code(
                get_file_contents_at_commit(commit, path[1]), path[1]
            )

            if i < len(modified_file_paths) - 1:
                commit_before += CODE_SEPARATOR
                commit_after += CODE_SEPARATOR
        output["commit_before"] = commit_before
        output["commit_after"] = commit_after

    df = pd.DataFrame([output])
    df.to_csv(destination, mode="a", index=False, header=write_header)


def add_commits_to_csv(
    commits: list[Commit],
    destination: str,
    write_header: bool = False,
    include_files: bool = True,
    batch_size: int = 20,
    pbar: tqdm = None,
):
    for i in range(0, len(commits), batch_size):
        batch = commits[i : i + batch_size]
        outputs = []
        for commit in batch:
            file_paths = get_file_paths_of_commit(commit)
            modified_file_paths = file_paths["modified"]

            parent = commit.parents[0] if commit.parents else None

            if not parent:
                continue

            output = {
                "commit_msg": commit.message,
                "sha": commit.hexsha,
                "remote_url": commit.repo.remotes.origin.url,
                "date": commit.authored_datetime,
                "labels": -1,
            }

            if include_files:
                commit_before = ""
                commit_after = ""
                for i, path in enumerate(modified_file_paths):
                    commit_before += format_code(
                        get_file_contents_at_commit(commit.parents[0], path[0]), path[0]
                    )
                    commit_after += format_code(
                        get_file_contents_at_commit(commit, path[1]), path[1]
                    )

                    if i < len(modified_file_paths) - 1:
                        commit_before += CODE_SEPARATOR
                        commit_after += CODE_SEPARATOR
                output["commit_before"] = commit_before
                output["commit_after"] = commit_after
            outputs.append(output)
        df = pd.DataFrame(outputs)
        df.to_csv(destination, mode="a", index=False, header=write_header)
        write_header = False

        if pbar:
            pbar.update(len(batch))


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
