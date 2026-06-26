# config.py
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

WORKSPACE_ROOT = os.path.join(CURRENT_DIR, "workspace")

HF_HOME = os.path.join(WORKSPACE_ROOT, "hf_cache")
HF_DATASETS_CACHE = os.path.join(HF_HOME, "datasets")

CHECKPOINTS_DIR = os.path.join(WORKSPACE_ROOT, "checkpoints")

MODEL_SPECS = {
    "small": {
        "n_embd": 512,
        "n_layer": 6,
        "n_head": 8,
        "d_ff": 2048,
    },
    "base": {
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "d_ff": 3072,
    },
    "large": {
        "n_embd": 1024,
        "n_layer": 24,
        "n_head": 16,
        "d_ff": 4096,
    },
}

os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", HF_DATASETS_CACHE)

os.makedirs(WORKSPACE_ROOT, exist_ok=True)
os.makedirs(HF_HOME, exist_ok=True)
os.makedirs(HF_DATASETS_CACHE, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
