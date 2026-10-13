"""CPU-only goal inventory from the actual, unchanged WebShop environment."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
p = argparse.ArgumentParser()
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
web = args.repo / "agent_system/environments/env_package/webshop/webshop"
sys.path.insert(0, str(web))
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

expected = {
    "items_shuffle_1000.json": "30a4765c3a327af72d9a9a95a6b2486d516f0fa1d3ecd83681901ce82a21b269",
    "items_ins_v2_1000.json": "f88a36314a397b53b3d9c3fa5878e5f7b26d35019a51ec83fbedeca61a948f6f",
}
for name, wanted in expected.items():
    assert (
        hashlib.sha256((web / "data" / name).read_bytes()).hexdigest() == wanted
    ), name
env = WebAgentTextEnv(
    seed=0,
    observation_mode="text_rich",
    num_products=None,
    human_goals=False,
    file_path=str(web / "data/items_shuffle_1000.json"),
    attr_path=str(web / "data/items_ins_v2_1000.json"),
    enable_ads=False,
)
assert len(env.server.all_products) == 1000
rows = [dict(index=i, **g) for i, g in enumerate(env.server.goals)]
data = dict(
    schema=1,
    environment_seed=0,
    products=1000,
    data_sha256=expected,
    goals=rows,
    source_sha256={
        str(p.relative_to(web)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            web / "web_agent_site/engine/goal.py",
            web / "web_agent_site/engine/engine.py",
            web / "web_agent_site/envs/web_agent_text_env.py",
        ]
    },
)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(data, ensure_ascii=False))
env.close()
print(json.dumps(dict(products=1000, goals=len(rows), output=str(args.output))))
