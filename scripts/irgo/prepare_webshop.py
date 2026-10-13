"""Download the upstream WebShop-1k assets and build their Lucene index.

Run explicitly with --download and/or --build-index. No dataset is committed
to this repository. Hash checks pin the products used by the reported run.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from common import PRODUCT_HASHES, WEBSHOP, sha256

DOWNLOADS = {
    "items_shuffle_1000.json": "1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib",
    "items_ins_v2_1000.json": "1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu",
    "items_human_ins.json": "14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--build-index", action="store_true")
    args = parser.parse_args()
    if not args.download and not args.build_index:
        parser.error("Select --download and/or --build-index")
    data = WEBSHOP / "data"
    data.mkdir(exist_ok=True)
    if args.download:
        import gdown

        for name, file_id in DOWNLOADS.items():
            target = data / name
            if target.exists():
                continue
            partial = Path(str(target) + ".partial")
            result = gdown.download(id=file_id, output=str(partial), quiet=False)
            if result is None:
                raise RuntimeError(f"Download failed: {name}")
            if name in PRODUCT_HASHES and sha256(partial) != PRODUCT_HASHES[name]:
                raise ValueError(
                    f"Upstream data changed: {name}; partial file retained for inspection"
                )
            partial.replace(target)
    for name, expected in PRODUCT_HASHES.items():
        if sha256(data / name) != expected:
            raise ValueError(f"Asset hash mismatch: {name}")
    if args.build_index:
        engine = WEBSHOP / "search_engine"
        index = engine / "indexes"
        if index.exists() and any(index.iterdir()):
            raise FileExistsError(
                "Index already exists; validate it instead of rebuilding in place"
            )
        for name in ["resources", "resources_100", "resources_1k", "resources_100k"]:
            (engine / name).mkdir(exist_ok=True)
        # Build from the assets just verified above, even when the caller has
        # environment variables pointing at another WebShop installation.
        env = dict(os.environ)
        env["WEBSHOP_FILE_PATH"] = str(data / "items_shuffle_1000.json")
        env["WEBSHOP_ATTR_PATH"] = str(data / "items_ins_v2_1000.json")
        subprocess.run(
            [sys.executable, "convert_product_file_format.py"], cwd=engine, env=env, check=True
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pyserini.index.lucene",
                "--collection",
                "JsonCollection",
                "--input",
                "resources",
                "--index",
                "indexes",
                "--generator",
                "DefaultLuceneDocumentGenerator",
                "--threads",
                "1",
                "--storePositions",
                "--storeDocvectors",
                "--storeRaw",
            ],
            cwd=engine,
            check=True,
        )
    print("WebShop asset preparation finished:", data)


if __name__ == "__main__":
    main()
