import os
from typing import Iterator

from git import Repo, Commit
import pandas as pd


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


def export_commit_to_csv(commit: Commit, destination: str):
    df = pd.DataFrame(
        [
            {
                "message": commit.message,
                "sha": commit.hexsha,
                "remote_url": commit.repo.remotes.origin.url,
                "label": "",
            }
        ]
    )
    # add header if doesn't exist
    df.to_csv(destination, mode="a", index=False, header=False)
