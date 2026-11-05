"""Compare retrieval configurations on the hand-judged queries in data/relevance.json.

    python evaluate.py              # summary table
    python evaluate.py --per-query  # also per-query average precision
    python evaluate.py -k 5         # cutoff for P@k / R@k / F1@k
"""
import argparse
import json
import os

from src.evaluation import (
    average_precision, f1_at_k, mean, precision_at_k, recall_at_k,
    reciprocal_rank,
)
from src.search_engine import SearchEngine

DOCS = os.path.join("data", "documents")
JUDGMENTS = os.path.join("data", "relevance.json")

CONFIGS = [
    ("raw tf", dict(stemming=False, sublinear_tf=False)),
    ("log tf", dict(stemming=False, sublinear_tf=True)),
    ("raw tf + stem", dict(stemming=True, sublinear_tf=False)),
    ("log tf + stem", dict(stemming=True, sublinear_tf=True)),
]


def load_queries():
    with open(JUDGMENTS, encoding="utf-8") as f:
        return json.load(f)["queries"]


def ranking_for(engine, query):
    return [r["filename"] for r in engine.search(query)["results"]]


def evaluate_config(engine, queries, k):
    rows = []
    for q in queries:
        relevant = set(q["relevant"])
        ranking = ranking_for(engine, q["query"])
        rows.append({
            "query": q["query"],
            "P": precision_at_k(ranking, relevant, k),
            "R": recall_at_k(ranking, relevant, k),
            "F1": f1_at_k(ranking, relevant, k),
            "AP": average_precision(ranking, relevant),
            "RR": reciprocal_rank(ranking, relevant),
        })
    return rows


def feedback_experiment(engine, queries, depth=5):
    """Residual-collection test of Rocchio.

    For each query, take the first relevant document in the top `depth` as the
    user's feedback, re-rank with Rocchio, then drop that document from both
    rankings (and from the relevant set) so feedback cannot win by returning
    the document it was given. Compare AP before and after on what is left.
    """
    before, after = [], []
    for q in queries:
        if q["query"].startswith('"'):
            continue  # a one-document phrase query has no residual to improve
        relevant = set(q["relevant"])
        base = ranking_for(engine, q["query"])
        seen = next((d for d in base[:depth] if d in relevant), None)
        if seen is None or len(relevant) < 2:
            continue
        fb = [r["filename"] for r in
              engine.feedback(q["query"], [seen])["results"]]
        residual = relevant - {seen}
        before.append(average_precision([d for d in base if d != seen], residual))
        after.append(average_precision([d for d in fb if d != seen], residual))
    return before, after


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=3)
    parser.add_argument("--per-query", action="store_true")
    args = parser.parse_args()

    queries = load_queries()
    print("{} queries, cutoff k={}\n".format(len(queries), args.k))
    header = "{:<16} {:>6} {:>6} {:>6} {:>6} {:>6}".format(
        "config", "P@k", "R@k", "F1@k", "MAP", "MRR")
    print(header)
    print("-" * len(header))

    per_query = {}
    engines = {}
    for name, opts in CONFIGS:
        engine = SearchEngine(DOCS, **opts)
        engine.build_index()
        engines[name] = engine
        rows = evaluate_config(engine, queries, args.k)
        per_query[name] = rows
        print("{:<16} {:>6.3f} {:>6.3f} {:>6.3f} {:>6.3f} {:>6.3f}".format(
            name, mean(r["P"] for r in rows), mean(r["R"] for r in rows),
            mean(r["F1"] for r in rows), mean(r["AP"] for r in rows),
            mean(r["RR"] for r in rows)))

    if args.per_query:
        print("\nAverage precision per query")
        names = [n for n, _ in CONFIGS]
        print("{:<46}".format("query") + "".join("{:>15}".format(n) for n in names))
        for i, q in enumerate(queries):
            print("{:<46}".format(q["query"][:44]) + "".join(
                "{:>15.3f}".format(per_query[n][i]["AP"]) for n in names))

    print("\nRocchio relevance feedback (residual collection, 1 judged doc, "
          "raw tf, no stemming)")
    before, after = feedback_experiment(engines["raw tf"], queries)
    print("  queries used: {}   MAP before: {:.3f}   MAP after: {:.3f}".format(
        len(before), mean(before), mean(after)))


if __name__ == "__main__":
    main()
