import pickle
from typing import Any, Dict, List, Optional
import numpy as np
import os

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchdata.stateful_dataloader import StatefulDataLoader
from torchtitan.datasets.tokenizer import Tokenizer
from torchtitan.logging import logger

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node

_supported_datasets = {
    "c4_test": "test/assets/c4_test",
    "c4": "allenai/c4",
    "fineweb10b": "test/assets/fineweb10B",
}

class HuggingFaceDataset(IterableDataset, Stateful):
    """PyTorch Representation of the HuggingFace Dataset or FineWeb10B Dataset.

    Args:
        dataset_name (str): name of the dataset to load ("c4_test", "c4", or "fineweb10b")
        dataset_path (Optional[str]): override default dataset path if needed
        tokenizer (Tokenizer): tokenizer used to encode data
        seq_len (int): max sequence length
        world_size (int): number of data parallel processes
        rank (int): rank of the current process
        infinite (bool): whether to loop infinitely over the dataset
    """
    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        tokenizer: Tokenizer,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
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

        # Set up dataset specific attributes
        self.is_fineweb = dataset_name == "fineweb10b"
        if self.is_fineweb:
            self.data_dir = dataset_path or _supported_datasets[dataset_name]
            self._current_chunk = 1
            self.num_chunks = 103
            logger.info(f"Preparing FineWeb10B dataset from {self.data_dir}")
        else:
            path = dataset_path or _supported_datasets[dataset_name]
            logger.info(f"Preparing {dataset_name} dataset from {path}")
            if dataset_name == "c4":
                ds = load_dataset(path, name="en", split="train", streaming=True)
            else:
                ds = load_dataset(path, split="train")
            self._data = split_dataset_by_node(ds, rank, world_size)

        # Checkpointing state
        self._sample_idx = 0
        self._chunk_position = 0
        self._all_tokens: List[int] = []

    def _load_fineweb_chunk(self, chunk_idx: int) -> np.ndarray:
        chunk_path = os.path.join(
            self.data_dir,
            f"fineweb_train_{chunk_idx:06d}.bin"
        )
        return np.fromfile(chunk_path, dtype=np.uint16)

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            if self.is_fineweb:
                for chunk_idx in range(self._current_chunk, self.num_chunks + 1):
                    chunk_data = self._load_fineweb_chunk(chunk_idx)
                    
                    chunk_size = len(chunk_data)
                    per_rank_size = chunk_size // self.world_size
                    start_idx = self.rank * per_rank_size
                    end_idx = start_idx + per_rank_size if self.rank != self.world_size - 1 else chunk_size
                    
                    for pos in range(self._chunk_position, end_idx - start_idx):
                        self._all_tokens.append(int(chunk_data[start_idx + pos]))
                        
                        while len(self._all_tokens) >= max_buffer_token_len:
                            x = torch.LongTensor(self._all_tokens[:max_buffer_token_len])
                            self._all_tokens = self._all_tokens[max_buffer_token_len:]
                            yield x[:-1], x[1:]
                    
                    self._current_chunk = chunk_idx + 1
                    self._chunk_position = 0
            else:
                for sample in self._get_data_iter():
                    tokens = self._tokenizer.encode(sample["text"], bos=True, eos=True)
                    self._all_tokens.extend(tokens)
                    self._sample_idx += 1

                    while len(self._all_tokens) >= max_buffer_token_len:
                        x = torch.LongTensor(self._all_tokens[:max_buffer_token_len])
                        self._all_tokens = self._all_tokens[max_buffer_token_len:]
                        yield x[:-1], x[1:]

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            
            # Reset for next iteration
            if self.is_fineweb:
                self._current_chunk = 1
                self._chunk_position = 0
            else:
                self._sample_idx = 0
            logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def _get_data_iter(self):
        if self._sample_idx == 0:
            return iter(self._data)
        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])
        return iter(self._data.skip(self._sample_idx))

    def state_dict(self):
        state = {"token_buffer": self._all_tokens}
        if self.is_fineweb:
            state.update({
                "current_chunk": self._current_chunk,
                "chunk_position": self._chunk_position
            })
        else:
            state["sample_idx"] = self._sample_idx
        return state

    def load_state_dict(self, state_dict):
        self._all_tokens = state_dict["token_buffer"]
        if self.is_fineweb:
            self._current_chunk = state_dict.get("current_chunk", 1)
            self._chunk_position = state_dict.get("chunk_position", 0)
        else:
            self._sample_idx = state_dict["sample_idx"]

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
