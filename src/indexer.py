from collections import Counter

from src.text_processor import preprocess


class InvertedIndex:
    def __init__(self):
        self.index = {}          # term -> set of doc IDs
        self.term_freq = {}      # doc ID -> Counter of term counts
        self.doc_length = {}     # doc ID -> number of tokens
        self.doc_ids = []

    def build(self, doc_texts):
        for doc_id, text in doc_texts.items():
            tokens = preprocess(text)
            self.doc_ids.append(doc_id)
            self.term_freq[doc_id] = Counter(tokens)
            self.doc_length[doc_id] = len(tokens)
            for term in set(tokens):
                if term not in self.index:
                    self.index[term] = set()
                self.index[term].add(doc_id)

    def lookup(self, term):
        return self.index.get(term, set())

    def vocabulary(self):
        return list(self.index.keys())

    def document_count(self):
        return len(self.doc_ids)

    def term_frequency(self, term, doc_id):
        return self.term_freq.get(doc_id, {}).get(term, 0)
