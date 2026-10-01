"""Retrieval-only audit: no Groq call, so it is free and fast. Shows what retrieval returns."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.guard import classify
from app.retriever import resolve_scheme, resolve_topic, retrieve
from config import load_config

CASES: list[tuple[str, str]] = [
    ("A", "What is the expense ratio of HDFC Large Cap Fund - Direct Growth?"),
    ("A", "What is the minimum SIP for HDFC Equity (Flexi Cap) Fund?"),
    ("A", "What is the exit load on HDFC Small Cap Fund?"),
    ("A", "What is the benchmark of HDFC Small Cap Fund?"),
    ("A", "What is the expense ratio of HDFC Balanced Advantage Fund?"),
    ("A", "What is the exit load?"),
    ("A", "What is the AUM of HDFC Large Cap Fund?"),
    ("B", "What is the expense ratio of Parag Parflex Fund?"),
    ("B", "Who won the 2024 cricket world cup?"),
    ("B", "What is the price of Bitcoin today?"),
    ("B", "Who is the president of India?"),
    ("B", "How do I cook pasta?"),
    ("C", "Should I buy HDFC Large Cap Fund now?"),
    ("C", "How much did HDFC Flexi Cap return last year?"),
    ("C", "My PAN is ABCDE1234F, can you check my returns?"),
    ("D", "What is the lock-in period in HDFC ELSS Tax Saver Fund?"),
    ("D", "What is the riskometer level of HDFC Balanced Advantage Fund?"),
    ("D", "How do I download my capital gains statement?"),
    ("D", "What is the difference between Direct and Regular plan?"),
    ("D", "What is the investment objective of HDFC Equity (Flexi Cap) Fund?"),
]

GROUPS = {
    "A": "in-scope, expect the right scheme + an answer",
    "B": "out of scope, expect NO hits and NO LLM call",
    "C": "guard refusal, expect NO retrieval and NO LLM call",
    "D": "known corpus gap, expect hits but no answer",
}


def main() -> int:
    cfg = load_config()
    failures = 0
    for group, purpose in GROUPS.items():
        print(f"\n=== group {group}: {purpose}")
        for tag, question in CASES:
            if tag != group:
                continue
            decision = classify(question, cfg)
            if decision.intent != "FACTUAL":
                print(f"  [refuse] {decision.intent:<11} no retrieval, no LLM call | {question}")
                continue
            result = retrieve(question, decision, cfg)
            if result.status == "out_of_corpus":
                print(f"  [oos]    reason=scope     no LLM call | {question}")
                continue
            schemes = sorted({hit.chunk.scheme_name for hit in result.hits})
            topics = sorted({hit.chunk.topic for hit in result.hits})
            top = result.hits[0]
            expect = resolve_scheme(question, cfg)
            # Scheme confusion is only a defect when a scheme was actually named: a generic
            # question ("What is the exit load?") legitimately draws on several schemes.
            wrong = [s for s in schemes if s != expect] if expect else []
            if wrong:
                failures += 1
            print(
                f"  [ok]     n={len(result.hits)} scheme={expect or 'none (generic)'} "
                f"topic={resolve_topic(question, cfg) or 'none'}"
                f" filtered={result.filtered_topic} | {schemes[0][:34] if schemes else '-'}"
                f" | top={top.chunk.chunk_id} score={top.score:.4f}"
            )
            if wrong:
                print(f"           !! conflicting scheme in hits: {wrong}")
    print(f"\nscheme-confusion failures: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
