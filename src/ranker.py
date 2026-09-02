import math
from collections import Counter


class Ranker:
    def __init__(self, index):
        self.index = index
        self.idf = {}
        self.doc_norm = {}
        self._prepare()

    def _prepare(self):
        n = self.index.document_count()

        # idf(term) = log(N / df(term)) + 1  (the +1 keeps terms that appear
        # in every document from collapsing to zero weight)
        for term, doc_ids in self.index.index.items():
            self.idf[term] = math.log(n / len(doc_ids)) + 1

        # precompute each document's tf-idf vector length for cosine similarity
        for doc_id in self.index.doc_ids:
            total = 0.0
            for term, freq in self.index.term_freq[doc_id].items():
                weight = freq * self.idf[term]
                total += weight * weight
            self.doc_norm[doc_id] = math.sqrt(total)

    def rank(self, query_terms, candidate_doc_ids):
        query_counts = Counter(query_terms)

        # build the query tf-idf vector
        query_vector = {}
        for term, freq in query_counts.items():
            if term in self.idf:
                query_vector[term] = freq * self.idf[term]

        query_norm = math.sqrt(sum(w * w for w in query_vector.values()))
        if query_norm == 0:
            return []

        scores = []
        for doc_id in candidate_doc_ids:
            dot = 0.0
            for term, q_weight in query_vector.items():
                tf = self.index.term_frequency(term, doc_id)
                if tf > 0:
                    dot += q_weight * (tf * self.idf[term])

            denom = query_norm * self.doc_norm[doc_id]
            score = dot / denom if denom != 0 else 0.0
            scores.append((doc_id, score))

        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores
