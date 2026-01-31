#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _render(template_path: Path, out_path: Path, replacements: dict) -> None:
    data = json.loads(template_path.read_text())
    text = json.dumps(data)
    for key, value in replacements.items():
        text = text.replace(key, value)
    out_path.write_text(json.dumps(json.loads(text), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--dataset-slug", default="santa-2025-solver")
    parser.add_argument("--kernel-slug", default="santa-2025-solver")
    parser.add_argument("--code-file", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--extra-datasets", default="")
    parser.add_argument("--out-dir", default="kaggle")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = f"{args.username}/{args.dataset_slug}"
    kernel_id = f"{args.username}/{args.kernel_slug}"

    template_dataset = Path("kaggle/dataset-metadata.template.json")
    template_kernel = Path("kaggle/kernel-metadata.template.json")

    _render(
        template_dataset,
        out_dir / "dataset-metadata.json",
        {"YOUR_KAGGLE_USERNAME/santa-2025-solver": dataset_id},
    )
    _render(
        template_kernel,
        out_dir / "kernel-metadata.json",
        {"YOUR_KAGGLE_USERNAME/santa-2025-solver": dataset_id},
    )

    # Replace kernel id separately to avoid accidental dataset replacement.
    kernel_meta = json.loads((out_dir / "kernel-metadata.json").read_text())
    kernel_meta["id"] = kernel_id
    extra = [d.strip() for d in args.extra_datasets.split(",") if d.strip()]
    seen = set()
    sources = []
    for source in [dataset_id, *extra]:
        if source in seen:
            continue
        seen.add(source)
        sources.append(source)
    kernel_meta["dataset_sources"] = sources
    if args.code_file:
        kernel_meta["code_file"] = args.code_file
    if args.title:
        kernel_meta["title"] = args.title
    (out_dir / "kernel-metadata.json").write_text(json.dumps(kernel_meta, indent=2))


if __name__ == "__main__":
    main()
