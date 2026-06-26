import os
import random
from glob import glob

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset
from huggingface_hub import snapshot_download


MAX_TRAIN_SHARDS = 1
PILE_REPO_ID = "Geonwoohong/pile-uncopyrighted-train-tokenized-gpt2"
PILE_TEST_REPO_ID = "Geonwoohong/pile-uncopyrighted-test-tokenized-gpt2"


def _is_rank0():
    return int(os.environ.get("RANK", "0")) == 0


def worker_init_fn(worker_id):
    rank = int(os.environ.get("RANK", "0"))

    worker_seed = 42 + worker_id + rank
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader_generator(rank=0):
    generator = torch.Generator()
    generator.manual_seed(42 + rank)
    return generator


def _distributed_barrier(local_rank):
    if not (dist.is_available() and dist.is_initialized()):
        return

    if torch.cuda.is_available():
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def load_or_prepare_pile(verbose=True):
    cache_dir_ds = os.environ.get("HF_DATASETS_CACHE", None)
    cache_dir_hub = os.environ.get("HF_HOME", None) or cache_dir_ds

    is_distributed = dist.is_available() and dist.is_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if MAX_TRAIN_SHARDS is None:
        if verbose and _is_rank0():
            print(f"Loading {PILE_REPO_ID} (cache_dir={cache_dir_ds}) [ALL shards]")

        if is_distributed:
            rank = dist.get_rank()

            if rank == 0:
                print("[Rank 0] Loading full dataset to cache...")
                load_dataset(PILE_REPO_ID, cache_dir=cache_dir_ds)
                print("[Rank 0] Finished loading full dataset.")

            print(f"[Rank {rank}] Waiting at barrier...")
            _distributed_barrier(local_rank)

        dataset = load_dataset(PILE_REPO_ID, cache_dir=cache_dir_ds)
        return dataset["train"], dataset["validation"]

    if verbose and _is_rank0():
        print(
            f"Downloading first {MAX_TRAIN_SHARDS} train shards "
            f"from {PILE_REPO_ID} (cache_dir={cache_dir_hub})"
        )

    allow_patterns = [
        *(f"train/train.{i:05d}.parquet" for i in range(MAX_TRAIN_SHARDS)),
        "validation/*.parquet",
    ]

    if is_distributed:
        rank = dist.get_rank()

        if rank == 0:
            snapshot_download(
                repo_id=PILE_REPO_ID,
                repo_type="dataset",
                cache_dir=cache_dir_hub,
                allow_patterns=allow_patterns,
            )

        _distributed_barrier(local_rank)

    repo_path = snapshot_download(
        repo_id=PILE_REPO_ID,
        repo_type="dataset",
        cache_dir=cache_dir_hub,
        allow_patterns=allow_patterns,
    )

    train_files = [
        os.path.join(repo_path, f"train/train.{i:05d}.parquet")
        for i in range(MAX_TRAIN_SHARDS)
    ]

    valid_files = sorted(glob(os.path.join(repo_path, "validation", "*.parquet")))

    if verbose and _is_rank0():
        print(f"Train shards: {len(train_files)} -> {train_files[0]}")
        print(f"Validation files: {len(valid_files)}")

    if is_distributed:
        rank = dist.get_rank()

        if rank == 0:
            print("[Rank 0] Processing train dataset cache...")
            load_dataset(
                "parquet",
                data_files={"train": train_files},
                num_proc=1,
            )

            print("[Rank 0] Processing validation dataset cache...")
            load_dataset(
                "parquet",
                data_files={"validation": valid_files},
                num_proc=1,
            )

            print("[Rank 0] Dataset caching complete.")

        print(f"[Rank {rank}] Waiting for dataset cache...")
        _distributed_barrier(local_rank)
        print(f"[Rank {rank}] Loading dataset from cache.")

    train_dataset = load_dataset(
        "parquet",
        data_files={"train": train_files},
    )["train"]

    valid_dataset = load_dataset(
        "parquet",
        data_files={"validation": valid_files},
    )["validation"]

    return train_dataset, valid_dataset


def load_pile_test(verbose=True):
    cache_dir = os.environ.get("HF_DATASETS_CACHE", None)

    if verbose:
        print(f"Loading {PILE_TEST_REPO_ID} (cache_dir={cache_dir})")

    dataset = load_dataset(
        PILE_TEST_REPO_ID,
        cache_dir=cache_dir,
    )

    return dataset["test"]