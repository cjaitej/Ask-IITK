"""Run tests/eval_questions.yaml through the pipeline and score what is
mechanical: did a retrieved passage come from the expected source, and was
that source cited?

Answer correctness is graded by hand from the JSON output — the script
prints the answer and a keyword hit rate, not a verdict.

Run:  python -m tests.run_eval             # retrieval only, no API key
      python -m tests.run_eval --generate  # full answers via Gemini
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import yaml

from rag.config import PROJECT_ROOT, get_settings
from rag.pipeline import get_pipeline

settings = get_settings()
QUESTIONS_PATH = Path(__file__).parent / "eval_questions.yaml"
GENERATE_DELAY_SECONDS = 4.5


def load_questions() -> List[Dict]:
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)["questions"]


def keyword_hits(answer: str, keywords: List[str]) -> List[bool]:
    lowered = answer.lower()
    return [kw.lower() in lowered for kw in keywords]


def run(generate: bool, out_path: Path | None) -> int:
    questions = load_questions()
    pipeline = get_pipeline()
    rows: List[Dict] = []

    print("\n{} eval questions | top_k={}\n".format(len(questions), settings.top_k))

    for item in questions:
        question = item["question"]
        expected = item["expected_source_id"]

        if generate:
            # 15 requests/minute on the Gemini free tier; 4.5s keeps a margin.
            if rows:
                time.sleep(GENERATE_DELAY_SECONDS)
            result = pipeline.answer(question)
            answer = result.answer
            retrieved_ids = [r["source_id"] for r in result.retrieved]
            cited_urls = [s["url"] for s in result.sources]
        else:
            passages = pipeline.retrieve(question)
            answer = ""
            retrieved_ids = [p.source_id for p in passages]
            cited_urls = []

        retrieval_ok = expected in retrieved_ids
        top1_ok = bool(retrieved_ids) and retrieved_ids[0] == expected
        hits = keyword_hits(answer, item.get("expected_answer_contains", []))

        rows.append(
            {
                "id": item["id"],
                "question": question,
                "expected_source_id": expected,
                "retrieval_ok": retrieval_ok,
                "top1_ok": top1_ok,
                "retrieved_source_ids": retrieved_ids,
                "answer": answer,
                "cited_urls": cited_urls,
                "keyword_hits": sum(hits),
                "keyword_total": len(hits),
                "answer_correct": None,  # <- fill in by hand
            }
        )

        mark = "PASS" if retrieval_ok else "MISS"
        line = "[{}] {:<4} {}".format(mark, item["id"], question[:66])
        print(line)
        print("        expected={:<18} got={}".format(expected, retrieved_ids))
        if generate:
            print(
                "        keywords {}/{} | {}".format(
                    sum(hits), len(hits), answer.replace("\n", " ")[:140]
                )
            )
        print()

    n = len(rows)
    recall = sum(r["retrieval_ok"] for r in rows)
    top1 = sum(r["top1_ok"] for r in rows)
    print("-" * 72)
    print(
        "retrieval@{}: {}/{} ({:.0%})   top-1 source: {}/{} ({:.0%})".format(
            settings.top_k, recall, n, recall / n, top1, n, top1 / n
        )
    )
    if generate:
        kw = sum(r["keyword_hits"] for r in rows)
        kw_total = sum(r["keyword_total"] for r in rows) or 1
        print(
            "expected-keyword coverage: {}/{} ({:.0%}) "
            "— a hint, not a grade; mark answer_correct by hand".format(
                kw, kw_total, kw / kw_total
            )
        )
    print("-" * 72)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), "utf-8")
        print("\nwrote {}  (set answer_correct: true/false by hand)".format(out_path))

    pipeline.close()  # release the local-mode Qdrant lock
    return 0 if recall == n else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the v1 eval set.")
    parser.add_argument(
        "--generate", action="store_true", help="also call Gemini for full answers"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "eval_run.json"),
        help="where to write the gradeable JSON report",
    )
    args = parser.parse_args()
    raise SystemExit(run(args.generate, Path(args.out) if args.out else None))
