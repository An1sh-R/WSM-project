from src.text_processor import preprocess
from src.wildcard import has_wildcard, expand


class QueryProcessor:
    def __init__(self, index):
        self.index = index

    def process(self, query):
        terms = preprocess(query)
        terms = [t for t in terms if t.strip("*") != ""]

        if not terms:
            return set(), []

        vocabulary = self.index.vocabulary()
        doc_sets = []
        matched_terms = []

        for term in terms:
            if has_wildcard(term):
                expansions = expand(term, vocabulary)
                matched_terms.extend(expansions)
                docs = set()
                for e in expansions:
                    docs |= self.index.lookup(e)
            else:
                matched_terms.append(term)
                docs = self.index.lookup(term)
            doc_sets.append(docs)

        result_docs = set(doc_sets[0])
        for docs in doc_sets[1:]:
            result_docs = result_docs & docs

        return result_docs, matched_terms
