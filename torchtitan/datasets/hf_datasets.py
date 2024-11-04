import pickle
from typing import Any, Dict, List, Optional
import numpy as np
import os

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchtitan.logging import logger
from torchtitan.datasets.tokenizer import Tokenizer
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node

_supported_datasets = {
    "c4_test": "test/assets/c4_test",
    "c4": "allenai/c4",
    "smollm1": ("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup"),
}

class HuggingFaceDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        tokenizer: Tokenizer,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
        seed: Optional[int] = 42,
        buffer_size: int = 10_000,  # Added buffer_size parameter
    ) -> None:
        if dataset_name not in _supported_datasets:
            raise ValueError(
                f"Dataset {dataset_name} is not supported. "
                f"Supported datasets are: {list(_supported_datasets.keys())}"
            )

        self.dataset_name = dataset_name
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self.world_size = world_size
        self.rank = rank

        # Load dataset
        if dataset_name == "smollm1":
            repo_name, subset = _supported_datasets[dataset_name]
            logger.info(f"Loading SmolLM dataset {subset} from {repo_name}")
            ds = load_dataset(repo_name, subset, split="train", streaming=True)
        elif dataset_name == "c4":
            path = dataset_path or _supported_datasets[dataset_name]
            logger.info(f"Preparing C4 dataset from {path}")
            ds = load_dataset(path, name="en", split="train", streaming=True)
        else:
            path = dataset_path or _supported_datasets[dataset_name]
            logger.info(f"Preparing {dataset_name} dataset from {path}")
            ds = load_dataset(path, split="train")

        # Apply shuffling for streaming datasets
        if isinstance(ds, Dataset):
            # Non-streaming dataset
            self._data = split_dataset_by_node(ds, rank, world_size)
        else:
            # Streaming dataset - shuffle then shard
            logger.info(f"Shuffled streaming dataset!")
            shuffled = ds.shuffle(seed=seed, buffer_size=buffer_size)
            self._data = split_dataset_by_node(shuffled, rank, world_size)

        self._sample_idx = 0
        self._all_tokens: List[int] = []
        
        if rank == 0:
            logger.info(
                f"Dataset {dataset_name} initialized with seed {seed} "
                f"and buffer_size {buffer_size}"
            )

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                text = sample["text"] if isinstance(sample["text"], str) else sample["text"].decode('utf-8')
                tokens = self._tokenizer.encode(text, bos=True, eos=True)
                self._all_tokens.extend(tokens)
                self._sample_idx += 1

                while len(self._all_tokens) >= max_buffer_token_len:
                    x = torch.LongTensor(self._all_tokens[:max_buffer_token_len])
                    self._all_tokens = self._all_tokens[max_buffer_token_len:]
                    yield x[:-1], x[1:]

            if not self.infinite:
                break
            
            self._sample_idx = 0

    def _get_data_iter(self):
        if self._sample_idx == 0:
            return iter(self._data)
        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])
        return iter(self._data.skip(self._sample_idx))

    def state_dict(self):
        return {
            "sample_idx": self._sample_idx,
            "token_buffer": self._all_tokens,
        }

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._all_tokens = state_dict["token_buffer"]

class DPAwareDataLoader(StatefulDataLoader, Stateful):
    """
    A wrapper around the StatefulDataLoader that ensures that the state is stored only once per DP rank.
    """

    def __init__(self, dp_rank: int, hf_ds: IterableDataset, batch_size: int):
        super().__init__(hf_ds, batch_size)
        self._dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> Dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions
        return {self._rank_id: pickle.dumps(super().state_dict())}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # State being empty is valid
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(
                f"DataLoader state is empty for dp rank {self._dp_rank}, expected key {self._rank_id}"
            )
            return
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))


def build_hf_data_loader(
    dataset_name: str,
    dataset_path: Optional[str],
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    world_size,
    rank,
    infinite: bool = True,
):
    hf_ds = HuggingFaceDataset(
        dataset_name, dataset_path, tokenizer, seq_len, world_size, rank, infinite
    )

    return DPAwareDataLoader(rank, hf_ds, batch_size=batch_size)
