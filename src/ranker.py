import math


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

    def rank(self, query_groups, candidate_doc_ids):
        # Each group is one query term: a plain word is a group of one, a
        # wildcard is the group of terms it expanded to. Spread one unit of
        # term frequency across the group so a wildcard that expands to many
        # terms carries about the same weight as a single plain word.
        query_vector = {}
        for group in query_groups:
            in_vocab = [term for term in group if term in self.idf]
            if not in_vocab:
                continue
            share = 1.0 / len(in_vocab)
            for term in in_vocab:
                query_vector[term] = query_vector.get(term, 0.0) + share * self.idf[term]

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
