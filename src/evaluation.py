"""Standard IR effectiveness measures for a ranked list and a set of relevant docs."""


def precision_at_k(ranking, relevant, k):
    top = ranking[:k]
    return sum(1 for d in top if d in relevant) / k if k else 0.0


def recall_at_k(ranking, relevant, k):
    if not relevant:
        return 0.0
    return sum(1 for d in ranking[:k] if d in relevant) / len(relevant)


def f1_at_k(ranking, relevant, k):
    p = precision_at_k(ranking, relevant, k)
    r = recall_at_k(ranking, relevant, k)
    return 2 * p * r / (p + r) if p + r else 0.0


def average_precision(ranking, relevant):
    """Mean of precision@rank over the ranks where a relevant doc appears.

    Relevant documents that are never retrieved contribute 0, so the divisor
    is the total number of relevant documents.
    """
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, doc in enumerate(ranking, start=1):
        if doc in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def reciprocal_rank(ranking, relevant):
    for rank, doc in enumerate(ranking, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0
