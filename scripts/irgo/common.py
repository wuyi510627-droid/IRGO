"""Portable release helpers; the training algorithm lives in recipe/GraphGPO.

Only paths, log destinations and explicit CLI overrides change here. Keep the
archived scientific settings in configs/irgo/full150.yaml as the source of truth.
"""

import hashlib
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WEBSHOP = REPO / "agent_system/environments/env_package/webshop/webshop"
PRODUCT_HASHES = {
    "items_shuffle_1000.json": "30a4765c3a327af72d9a9a95a6b2486d516f0fa1d3ecd83681901ce82a21b269",
    "items_ins_v2_1000.json": "f88a36314a397b53b3d9c3fa5878e5f7b26d35019a51ec83fbedeca61a948f6f",
}
BASE_MODEL_HASH = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def training_config(model, data_dir, output, profile="full150"):
    """Build a relocatable config without importing Ray, CUDA or the trainer."""
    text = (REPO / "configs/irgo/full150.yaml").read_text()
    values = {
        "IRGO_MODEL_PATH": str(Path(model).resolve()),
        "IRGO_TRAIN_FILE": str(Path(data_dir).resolve() / "train.parquet"),
        "IRGO_VAL_FILE": str(Path(data_dir).resolve() / "test.parquet"),
        "IRGO_RUN_DIR": str(Path(output).resolve() / "run"),
        "IRGO_REPO": str(REPO),
    }

    # Load YAML before substitution so paths containing quotes, # or ':' remain
    # plain values instead of being parsed as YAML syntax.
    def substitute(value):
        if isinstance(value, dict):
            return {k: substitute(v) for k, v in value.items()}
        if isinstance(value, list):
            return [substitute(v) for v in value]
        if isinstance(value, str):
            for key, replacement in values.items():
                value = value.replace("${oc.env:" + key + "}", replacement)
        return value

    cfg = substitute(yaml.safe_load(text))
    if profile == "smoke":
        cfg["data"].update(train_batch_size=4, gen_batch_size=4, val_batch_size=8)
        cfg["env"]["rollout"]["n"] = 2
        cfg["algorithm"]["u"].update(max_probes=4, max_probes_per_task=2, max_extra_env_calls=60)
        cfg["trainer"].update(
            total_training_steps=2,
            total_epochs=2,
            save_freq=2,
            test_freq=0,
            val_before_train=False,
            experiment_name="irgo_smoke",
        )
    elif profile != "full150":
        raise ValueError(f"Unknown profile: {profile}")
    return cfg


def runtime_environment(profile="full150"):
    """Reproduce the two-GPU runtime while accepting the caller's CUDA/W&B setup."""
    env = dict(os.environ)
    defaults = {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_ATTENTION_BACKEND": "XFORMERS",
        "VLLM_USE_V1": "0",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "MPLBACKEND": "Agg",
        "WANDB_MODE": "offline",
        "HYDRA_FULL_ERROR": "1",
        "WEBSHOP_ENVS_PER_ACTOR": "8" if profile == "smoke" else "32",
        "WEBSHOP_ENVS_PER_ACTOR_VAL": "8" if profile == "smoke" else "64",
    }
    for key, value in defaults.items():
        env.setdefault(key, value)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def verify_webshop():
    """Reject a silent change to the 1,000-product dataset before launching GPUs."""
    import json

    paths = {
        "items_shuffle_1000.json": Path(
            os.environ.get("WEBSHOP_FILE_PATH", WEBSHOP / "data/items_shuffle_1000.json")
        ),
        "items_ins_v2_1000.json": Path(
            os.environ.get("WEBSHOP_ATTR_PATH", WEBSHOP / "data/items_ins_v2_1000.json")
        ),
    }
    for name, path in paths.items():
        if not path.is_file() or sha256(path) != PRODUCT_HASHES[name]:
            raise ValueError(f"Missing or different WebShop-1k asset: {path}")
    if len(json.loads(paths["items_shuffle_1000.json"].read_text())) != 1000:
        raise ValueError("Expected exactly 1000 products")
    # use_small=True loads 1k JSON; this implementation uses num_products=None,
    # so the runtime opens search_engine/indexes, not indexes_1k.
    index = WEBSHOP / "search_engine/indexes"
    if not index.is_dir() or not any(index.iterdir()):
        raise ValueError(f"Build the WebShop Lucene index first: {index}")
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(str(index))
    products = json.loads(paths["items_shuffle_1000.json"].read_text())
    ids = {str(item["asin"]) for item in products}
    if searcher.num_docs != 1000 or any(searcher.doc(asin) is None for asin in ids):
        raise ValueError("Lucene index must contain exactly the same 1000 ASINs")
    return {name: PRODUCT_HASHES[name] for name in paths}
