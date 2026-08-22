"""crawl -> chunk -> embed -> index, in one command.

    python -m scripts.build_all
    python -m scripts.build_all --skip-crawl   # reuse data/raw
    python -m scripts.build_all --recreate     # drop the collection first
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the index end to end.")
    parser.add_argument("--skip-crawl", action="store_true")
    parser.add_argument("--skip-chunk", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    if not args.skip_crawl:
        print("\n=== crawl ===")
        from ingestion.crawler import crawl

        crawl()

    if not args.skip_chunk:
        print("\n=== parse + chunk ===")
        from ingestion.chunker import build_chunks

        build_chunks()

    print("\n=== embed + index ===")
    from ingestion.indexer import index

    index(recreate=args.recreate)


if __name__ == "__main__":
    main()
