from collections import Counter

from src.text_processor import preprocess, positional_tokens


class InvertedIndex:
    def __init__(self, stemming=False):
        self.stemming = stemming
        self.index = {}          # term -> set of doc IDs
        self.term_freq = {}      # doc ID -> Counter of term counts
        self.doc_length = {}     # doc ID -> number of tokens
        self.doc_ids = []
        # Positional index: term -> {doc ID: [positions]}. Built over the token
        # stream *with* stop words, so "to be or not to be" is searchable even
        # though every word in it is a stop word.
        self.positions = {}

    def build(self, doc_texts):
        for doc_id, text in doc_texts.items():
            tokens = preprocess(text, self.stemming)
            self.doc_ids.append(doc_id)
            self.term_freq[doc_id] = Counter(tokens)
            self.doc_length[doc_id] = len(tokens)
            for term in set(tokens):
                if term not in self.index:
                    self.index[term] = set()
                self.index[term].add(doc_id)

            for pos, term in enumerate(positional_tokens(text, self.stemming)):
                self.positions.setdefault(term, {}).setdefault(doc_id, []).append(pos)

    def lookup(self, term):
        return self.index.get(term, set())

    def vocabulary(self):
        return list(self.index.keys())

    def document_count(self):
        return len(self.doc_ids)

    def term_frequency(self, term, doc_id):
        return self.term_freq.get(doc_id, {}).get(term, 0)

    def phrase_counts(self, phrase_terms):
        """{doc_id: occurrences} of the exact consecutive phrase.

        phrase_terms must already be normalized/stemmed like the index. Start
        from the postings of the first word, then keep only the positions p
        where word i occurs at p + i for every later word i.
        """
        if not phrase_terms:
            return {}
        first = self.positions.get(phrase_terms[0], {})
        counts = {}
        for doc_id, starts in first.items():
            valid = starts
            for offset, term in enumerate(phrase_terms[1:], start=1):
                later = self.positions.get(term, {}).get(doc_id)
                if not later:
                    valid = []
                    break
                later_set = set(later)
                valid = [p for p in valid if p + offset in later_set]
                if not valid:
                    break
            if valid:
                counts[doc_id] = len(valid)
        return counts
