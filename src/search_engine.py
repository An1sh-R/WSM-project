import re

from src.document_loader import load_documents
from src.indexer import InvertedIndex
from src.query_processor import QueryProcessor
from src.ranker import Ranker
from src.stemmer import stem


class SearchEngine:
    def __init__(self, documents_dir, stemming=False, sublinear_tf=False):
        self.documents_dir = documents_dir
        self.stemming = stemming
        self.sublinear_tf = sublinear_tf
        self.doc_filenames = {}
        self.doc_texts = {}
        self.index = None
        self.query_processor = None
        self.ranker = None

    def build_index(self):
        self.doc_filenames, self.doc_texts = load_documents(self.documents_dir)
        self.index = InvertedIndex(stemming=self.stemming)
        self.index.build(self.doc_texts)
        self.query_processor = QueryProcessor(self.index)
        self.ranker = Ranker(self.index, sublinear_tf=self.sublinear_tf)

    # ----- snippets ---------------------------------------------------------

    def preview(self, doc_id, length=200):
        text = re.sub(r"\s+", " ", self.doc_texts[doc_id]).strip()
        if len(text) > length:
            text = text[:length] + "..."
        return text

    def _first_match(self, lowered, doc_id, terms, phrases):
        """Character offset of the best spot to centre the snippet on."""
        # An exact phrase hit beats a scattering of single words.
        for phrase in phrases:
            if doc_id in phrase["counts"]:
                pattern = r"\W+".join(re.escape(w) for w in phrase["words"])
                m = re.search(pattern, lowered)
                if m:
                    return m.start()
                # stemmed index: the stored words are surface forms, so the
                # pattern above normally works; fall through otherwise.

        wanted = set(terms)
        for m in re.finditer(r"[a-z0-9]+", lowered):
            word = m.group()
            if (stem(word) if self.stemming else word) in wanted:
                return m.start()
        return -1

    def snippet(self, doc_id, terms, length=200, phrases=()):
        text = re.sub(r"\s+", " ", self.doc_texts[doc_id]).strip()
        lowered = text.lower()

        position = self._first_match(lowered, doc_id, terms, phrases)
        if position == -1:
            return self.preview(doc_id, length)

        start = max(0, position - length // 2)
        end = start + length
        piece = text[start:end].strip()
        if start > 0:
            piece = "..." + piece
        if end < len(text):
            piece = piece + "..."
        return piece

    # ----- searching --------------------------------------------------------

    def _format(self, ranked, terms, phrases=()):
        max_score = ranked[0][1] if ranked else 0.0
        results = []
        for doc_id, score in ranked:
            relative = (score / max_score * 100) if max_score > 0 else 0.0
            results.append({
                "filename": self.doc_filenames[doc_id],
                "score": score,
                "relative": relative,
                "preview": self.snippet(doc_id, terms, phrases=phrases),
            })
        return results

    def search(self, query):
        if query is None or query.strip() == "":
            return {"status": "empty", "results": []}

        candidate_docs, matched_groups, phrases = self.query_processor.process(query)
        matched_terms = [term for group in matched_groups for term in group]

        if not matched_groups:
            if phrases:
                # Phrase made only of stop words ("to be or not to be"): there
                # are no weighted terms, so rank by how often the phrase occurs.
                if not candidate_docs:
                    return {"status": "no_match", "results": []}
                ranked = sorted(
                    ((d, float(sum(p["counts"].get(d, 0) for p in phrases)))
                     for d in candidate_docs),
                    key=lambda pair: pair[1], reverse=True)
                return {"status": "ok",
                        "results": self._format(ranked, [], phrases)}
            if not candidate_docs:
                return {"status": "only_stopwords", "results": []}
            # Query had only exclusions (e.g. "-death"); there is nothing to score.
            return {"status": "no_positive_terms", "results": []}

        if not candidate_docs:
            return {"status": "no_match", "results": []}

        ranked = self.ranker.rank(matched_groups, candidate_docs)
        if not ranked:
            return {"status": "no_match", "results": []}

        return {"status": "ok",
                "results": self._format(ranked, matched_terms, phrases)}

    # ----- relevance feedback (Rocchio) --------------------------------------

    def feedback(self, query, relevant_filenames, nonrelevant_filenames=(),
                 alpha=1.0, beta=0.75, gamma=0.15, max_terms=25):
        """Re-rank the whole collection after the user marks documents relevant.

        The query vector is moved towards the centroid of the relevant documents
        (query expansion + reweighting). Because expansion adds new terms, every
        document is a candidate, not only those matching the original query.
        """
        by_name = {name: doc_id for doc_id, name in self.doc_filenames.items()}
        relevant = [by_name[n] for n in relevant_filenames if n in by_name]
        nonrelevant = [by_name[n] for n in nonrelevant_filenames if n in by_name]
        if not relevant:
            return self.search(query)

        _, groups, _ = (self.query_processor.process(query)
                        if query and query.strip() else (set(), [], []))
        original = self.ranker.query_vector(groups)
        expanded = self.ranker.rocchio(
            original, relevant, nonrelevant, alpha, beta, gamma, max_terms)

        ranked = self.ranker.rank_vector(expanded, self.index.doc_ids)
        ranked = [(d, s) for d, s in ranked if s > 0]
        if not ranked:
            return {"status": "no_match", "results": []}

        added = sorted(
            (t for t in expanded if t not in original),
            key=lambda t: expanded[t], reverse=True)
        top_terms = sorted(expanded, key=lambda t: expanded[t], reverse=True)[:12]
        return {
            "status": "ok",
            "results": self._format(ranked, top_terms),
            "expanded_terms": added[:10],
        }
