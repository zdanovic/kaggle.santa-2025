#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--dataset-slug", default="santa-2025-solver")
    parser.add_argument("--kernel-prefix", default="santa-2025-batch")
    parser.add_argument("--batches", default="batch_a,batch_b,batch_c,batch_d")
    args = parser.parse_args()

    dataset_id = f"{args.username}/{args.dataset_slug}"
    batches = [b.strip() for b in args.batches.split(",") if b.strip()]

    for batch in batches:
        suffix = batch.split("_")[-1]
        kernel_id = f"{args.username}/{args.kernel_prefix}-{suffix}"
        metadata = {
            "id": kernel_id,
            "title": f"Santa 2025 {batch}",
            "code_file": f"run_batch_{suffix}.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": False,
            "enable_internet": True,
            "dataset_sources": [dataset_id],
        }
        out_path = Path("kaggle") / f"kernel-metadata.{batch}.json"
        out_path.write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
