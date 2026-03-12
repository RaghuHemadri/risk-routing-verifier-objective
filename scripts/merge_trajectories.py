#!/usr/bin/env python3
"""Merge multiple trajectory JSONL files, deduplicating by episode_id.

Usage:
    python scripts/merge_trajectories.py \
        --inputs path/to/file1.jsonl path/to/file2.jsonl ... \
        --output path/to/final.jsonl \
        [--verbose]

The first occurrence of each episode_id is kept; later duplicates are dropped.
"""
import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="Input JSONL files (order matters: earlier files take priority).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    total_read = 0
    total_written = 0
    total_dupes = 0

    with open(output_path, "w") as out_f:
        for input_path in args.inputs:
            p = Path(input_path)
            if not p.exists():
                print(f"[WARN] Input not found, skipping: {p}", file=sys.stderr)
                continue
            file_read = 0
            file_written = 0
            with open(p) as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    total_read += 1
                    file_read += 1
                    ep = json.loads(line)
                    eid = ep.get("episode_id", "")
                    if eid in seen:
                        total_dupes += 1
                        continue
                    seen.add(eid)
                    out_f.write(line + "\n")
                    total_written += 1
                    file_written += 1
            if args.verbose:
                print(f"  {p.name}: {file_read} read, {file_written} written")

    print(
        f"Merged {len(args.inputs)} files: "
        f"{total_read} total lines, "
        f"{total_written} unique episodes written, "
        f"{total_dupes} duplicates dropped."
    )
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
