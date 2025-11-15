#!/usr/bin/env python3
"""Generate short-form theory cards from existing theory records."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.short_card_converter import ConversionContext, ShortCardConverter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert theory JSON files into short cards.")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["data/theories_v2.1"],
        help="One or more directories containing detailed theory JSON files.",
    )
    parser.add_argument(
        "--dest",
        default="cards",
        help="Directory where generated short-card JSON files will be written.",
    )
    parser.add_argument(
        "--schema",
        default="schemas/card.schema.json",
        help="Path to the JSON schema that defines the short-card shape.",
    )
    parser.add_argument(
        "--overrides",
        default=None,
        help="Optional JSON file containing manual overrides keyed by card id.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when scanning sources.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing card files instead of skipping them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema_path = Path(args.schema)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    override_context = ConversionContext.from_path(Path(args.overrides)) if args.overrides else ConversionContext.from_path(None)
    converter = ShortCardConverter(schema_path=schema_path, overrides=override_context)

    destination = Path(args.dest)
    total_converted = 0
    for source in args.sources:
        source_path = Path(source)
        if not source_path.exists():
            print(f"[WARN]  Source directory not found, skipping: {source_path}")
            continue
        count = converter.convert_directory(
            source_dir=source_path,
            destination_dir=destination,
            recursive=args.recursive,
            skip_existing=not args.force,
        )
        total_converted += count
        print(f"Converted {count} files from {source_path}")

    print(f"Completed short-card generation. Total new cards: {total_converted}")


if __name__ == "__main__":
    main()
