from src.text_processor import preprocess
from src.wildcard import has_wildcard, expand


class QueryProcessor:
    def __init__(self, index):
        self.index = index

    def _parse(self, query):
        positive = []
        negative = []
        use_or = False
        negate_next = False

        for raw in query.split():
            word = raw.lower()
            if word == "or":
                use_or = True
                continue
            if word == "not":
                negate_next = True
                continue
            if raw.startswith("-") and len(raw) > 1:
                negate_next = True
                raw = raw[1:]

            terms = [t for t in preprocess(raw) if t.strip("*") != ""]
            if terms:
                if negate_next:
                    negative.extend(terms)
                else:
                    positive.extend(terms)
            negate_next = False

        return positive, negative, use_or

    def _docs_for(self, term):
        if has_wildcard(term):
            expansions = expand(term, self.index.vocabulary())
            docs = set()
            for e in expansions:
                docs |= self.index.lookup(e)
            return docs, expansions
        return self.index.lookup(term), [term]

    def process(self, query):
        positive, negative, use_or = self._parse(query)

        if not positive and not negative:
            return set(), []

        # Each positive query term becomes one group: a plain word is a group
        # of one, a wildcard is the group of terms it expanded to. The ranker
        # weights per group, so a broad wildcard cannot outvote the plain
        # words next to it.
        matched_groups = []

        if positive:
            doc_sets = []
            for term in positive:
                docs, expansions = self._docs_for(term)
                if expansions:
                    matched_groups.append(expansions)
                doc_sets.append(docs)
            result_docs = set(doc_sets[0])
            for docs in doc_sets[1:]:
                result_docs = result_docs | docs if use_or else result_docs & docs
        else:
            result_docs = set(self.index.doc_ids)

        for term in negative:
            docs, _ = self._docs_for(term)
            result_docs = result_docs - docs

        return result_docs, matched_groups
