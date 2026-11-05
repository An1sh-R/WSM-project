from bisect import bisect_left, bisect_right
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

    def proximity_counts(self, term_a, term_b, distance):
        """{doc_id: pairs} where term_a and term_b occur within `distance`
        positions of each other, in either order.

        Positions are over the token stream with stop words kept, so "love /3
        death" allows at most two words between them. For each position of
        term_a, binary-search the sorted positions of term_b for any inside the
        window [p - distance, p + distance] (excluding p itself, so a word does
        not count as being near itself).
        """
        a_docs = self.positions.get(term_a, {})
        b_docs = self.positions.get(term_b, {})
        counts = {}
        for doc_id, a_positions in a_docs.items():
            b_positions = b_docs.get(doc_id)
            if not b_positions:
                continue
            pairs = 0
            for p in a_positions:
                lo = bisect_left(b_positions, p - distance)
                hi = bisect_right(b_positions, p + distance)
                inside = hi - lo
                if term_a == term_b:
                    inside -= 1  # p itself is in the window
                pairs += max(inside, 0)
            if pairs:
                counts[doc_id] = pairs
        return counts
