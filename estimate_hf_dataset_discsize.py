#!/usr/bin/env python3
"""
Show total and per-top-level-directory sizes for one or more Hugging Face datasets.

Usage:
    # With defaults (nvidia/ClimbLab and OptimalScale/ClimbLab)
    python estimate_hf_dataset_discsize.py

    # Explicit list (maximum of 10 datasets allowed)
    python estimate_hf_dataset_discsize.py --datasets "repo1, repo2, repo3"

Requirements:
    - huggingface_hub
"""

import argparse

from collections import defaultdict
from huggingface_hub import HfApi


REPO_TYPE = "dataset"
REPO_DEFAULTS = ["nvidia/ClimbLab", "OptimalScale/ClimbLab"]


def humanize_disc_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    for u in units:
        if num_bytes < 1024 or u == units[-1]:
            return f"{num_bytes:.2f} {u}"
            
        num_bytes /= 1024


def list_files_with_sizes(repo_id: str, repo_type: str = REPO_TYPE) -> list[tuple[str, int]]:
    """
    Retrieve (path, size_bytes) for files in a repo with metadata available.
    """
    api = HfApi()
    info = api.repo_info(repo_id=repo_id, repo_type=repo_type, files_metadata=True)

    files_metadata: list[tuple[str, int]] = []
    
    for sib in info.siblings:
        size = getattr(sib, "size", None)
        rfilename = getattr(sib, "rfilename", None)
        
        if size is not None and rfilename:
            files_metadata.append((rfilename, int(size)))
            
    return files_metadata


def parse_datasets_arg(raw: str | None) -> list[str]:
    """
    Split a comma-separated list, trim spaces, drop empties, dedupe (preserve order).
    If raw is None or results in an empty list, return defaults.
    """
    if not raw:
        return REPO_DEFAULTS[:]

    items = [part.strip() for part in raw.split(",")]
    cleaned: list[str] = []
    seen = set()
    
    for item in items:
        if not item or item in seen:
            continue
            
        cleaned.append(item)
        seen.add(item)
        
    return cleaned or REPO_DEFAULTS[:]  # fallback if user passed only commas/spaces


def ask_yes_no(prompt: str, default: str = "n") -> bool:
    """
    Prompt the user with a yes/no question. Returns True for yes, False for no.
    default: 'y' or 'n' used if user just hits Enter or input is invalid repeatedly.
    """
    default = default.lower()
    suffix = " [Y/n] " if default == "y" else " [y/N] "
    
    try:
        ans = input(prompt + suffix).strip().lower()
    except EOFError:
        ans = ""

    if ans in {"y", "yes"}:
        return True
        
    if ans in {"n", "no"}:
        return False
        
    # If ans equals none of the valid options, fall back to default
    return default == "y"


def show_discsizes(repos: list[str], repo_type: str = REPO_TYPE) -> None:
    if not repos:
        print("No repositories provided.")
        return

    files_by_repo = {repo_id: list_files_with_sizes(repo_id, repo_type) for repo_id in repos}

    if all(not lst for lst in files_by_repo.values()):
        print("No file metadata found (repos empty or rate-limited).")
        return

    for repo_id, file_list in files_by_repo.items():
        print(f"\n\n---- Repository: {repo_id} ({repo_type}) ----")

        if not file_list:
            print("No files found for this repository.")
            continue

        total_bytes = sum(sz for _, sz in file_list)
        print(f"Total on-hub file size: {total_bytes} bytes ({humanize_disc_size(total_bytes)})\n")

        per_dir = defaultdict(int)
        for path, sz in file_list:
            top = path.split("/", 1)[0] if "/" in path else "(root)"
            per_dir[top] += sz

        print("Per-directory totals:")
        for d in sorted(per_dir):
            bytes_ = per_dir[d]
            print(f"  {d:<15} {humanize_disc_size(bytes_)} ({bytes_} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show dataset sizes (total and per top-level directory) for HF Hub repos."
    )
    parser.add_argument(
        "--datasets",
        help='Comma-separated list of repos, e.g. "nvidia/ClimbLab, OptimalScale/ClimbLab". '
             "If omitted or empty, defaults to the two ClimbLab repos.",
    )
    args = parser.parse_args()

    repos = parse_datasets_arg(args.datasets)

    if len(repos) > 10:
        print(f"You provided {len(repos)} datasets. The maximum allowed is 10.")
        proceed = ask_yes_no("Do you want to run for the first 10 datasets?", default="n")
        
        if not proceed:
            raise SystemExit("Aborted by user.")
            
        repos = repos[:10]
        print("Proceeding with the first 10 datasets:")

    print(", ".join(repos))
    show_discsizes(repos, REPO_TYPE)


if __name__ == "__main__":
    main()
