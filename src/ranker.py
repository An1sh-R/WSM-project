import math


class Ranker:
    """Vector space model: TF-IDF weights, cosine similarity.

    sublinear_tf=False: w(t, d) = tf * idf             (raw term frequency)
    sublinear_tf=True:  w(t, d) = (1 + log tf) * idf   (log-scaled term frequency)
    Cosine length normalization is applied in both cases.
    """

    def __init__(self, index, sublinear_tf=False):
        self.index = index
        self.sublinear_tf = sublinear_tf
        self.idf = {}
        self.doc_norm = {}
        self._prepare()

    def _tf_weight(self, tf):
        if tf <= 0:
            return 0.0
        return 1.0 + math.log(tf) if self.sublinear_tf else float(tf)

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
                weight = self._tf_weight(freq) * self.idf[term]
                total += weight * weight
            self.doc_norm[doc_id] = math.sqrt(total)

    # ----- vectors ----------------------------------------------------------

    def query_vector(self, query_groups):
        # Each group is one query term: a plain word is a group of one, a
        # wildcard is the group of terms it expanded to. Spread one unit of
        # term frequency across the group so a wildcard that expands to many
        # terms carries about the same weight as a single plain word.
        vector = {}
        for group in query_groups:
            in_vocab = [term for term in group if term in self.idf]
            if not in_vocab:
                continue
            share = 1.0 / len(in_vocab)
            for term in in_vocab:
                vector[term] = vector.get(term, 0.0) + share * self.idf[term]
        return vector

    def doc_vector(self, doc_id):
        """Full TF-IDF vector of one document, {term: weight}."""
        return {
            term: self._tf_weight(freq) * self.idf[term]
            for term, freq in self.index.term_freq[doc_id].items()
        }

    # ----- scoring ----------------------------------------------------------

    def rank_vector(self, query_vector, candidate_doc_ids):
        """Rank candidates by cosine similarity to an arbitrary query vector."""
        query_norm = math.sqrt(sum(w * w for w in query_vector.values()))
        if query_norm == 0:
            return []

        scores = []
        for doc_id in candidate_doc_ids:
            dot = 0.0
            for term, q_weight in query_vector.items():
                tf = self.index.term_frequency(term, doc_id)
                if tf > 0:
                    dot += q_weight * (self._tf_weight(tf) * self.idf[term])

            denom = query_norm * self.doc_norm[doc_id]
            score = dot / denom if denom != 0 else 0.0
            scores.append((doc_id, score))

        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores

    def rank(self, query_groups, candidate_doc_ids):
        return self.rank_vector(self.query_vector(query_groups), candidate_doc_ids)

    # ----- relevance feedback ----------------------------------------------

    def rocchio(self, query_vector, relevant_ids, nonrelevant_ids=(),
                alpha=1.0, beta=0.75, gamma=0.15, max_terms=25):
        """Rocchio: q' = a*q + b*mean(relevant) - g*mean(non-relevant).

        Negative weights are dropped, and only the max_terms heaviest terms of
        the feedback centroid are added so the expanded query stays focused.
        """
        def centroid(doc_ids):
            total = {}
            doc_ids = list(doc_ids)
            for doc_id in doc_ids:
                norm = self.doc_norm[doc_id] or 1.0
                for term, w in self.doc_vector(doc_id).items():
                    total[term] = total.get(term, 0.0) + w / norm  # unit length
            return {t: w / len(doc_ids) for t, w in total.items()} if doc_ids else {}

        rel = centroid(relevant_ids)
        non = centroid(nonrelevant_ids)

        # keep the strongest feedback terms only
        top = sorted(rel.items(), key=lambda kv: kv[1], reverse=True)[:max_terms]
        rel = dict(top)

        q_norm = math.sqrt(sum(w * w for w in query_vector.values())) or 1.0
        new = {}
        for term, w in query_vector.items():
            new[term] = alpha * w / q_norm
        for term, w in rel.items():
            new[term] = new.get(term, 0.0) + beta * w
        for term, w in non.items():
            if term in new:
                new[term] -= gamma * w
        return {t: w for t, w in new.items() if w > 0}
