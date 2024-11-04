# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
from pathlib import Path
from typing import Optional
from requests.exceptions import HTTPError


def hf_download(
    repo_id: str, tokenizer_path: str, local_dir: str, hf_token: Optional[str] = None
) -> None:
    from huggingface_hub import hf_hub_download

    # Construct tokenizer path
    tokenizer_path = (
        f"{tokenizer_path}/tokenizer.model" if tokenizer_path else "tokenizer.model"
    )

    # Convert to Path objects
    local_dir_path = Path(local_dir)
    target_path = local_dir_path / os.path.basename(tokenizer_path)

    # Ensure the target directory exists
    local_dir_path.mkdir(parents=True, exist_ok=True)

    # Clean up existing path if necessary
    if target_path.exists():
        if target_path.is_dir():
            shutil.rmtree(target_path)
        else:
            target_path.unlink()

    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=tokenizer_path,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
        )
    except HTTPError as e:
        if e.response.status_code == 401:
            print(
                "You need to pass a valid `--hf_token=...` to download private checkpoints."
            )
        else:
            raise e


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download tokenizer from HuggingFace.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Repository ID to download from. default to Llama-3-8B",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="original",
        help="the tokenizer.model path relative to repo_id",
    )
    parser.add_argument(
        "--hf_token", type=str, default=None, help="HuggingFace API token"
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="torchtitan/datasets/tokenizer/",
        help="local directory to save the tokenizer.model",
    )

    args = parser.parse_args()
    hf_download(args.repo_id, args.tokenizer_path, args.local_dir, args.hf_token)