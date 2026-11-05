import re

from src.text_processor import (
    preprocess, normalize, tokenize, remove_stop_words, stem_tokens,
)
from src.wildcard import has_wildcard, expand

# A quoted phrase, or any run of non-space characters.
_QUERY_TOKEN = re.compile(r'"([^"]*)"|(\S+)')


class QueryProcessor:
    def __init__(self, index):
        self.index = index

    def _phrase_words(self, text):
        # Wildcards are not supported inside a phrase; stop words are kept.
        return tokenize(normalize(text.replace("*", " ")))

    def _parse(self, query):
        """Return (positive, negative, use_or).

        Each entry is ("term", word) or ("phrase", [words]).
        """
        positive = []
        negative = []
        use_or = False
        negate_next = False

        for match in _QUERY_TOKEN.finditer(query):
            phrase, raw = match.group(1), match.group(2)

            if phrase is not None:
                words = self._phrase_words(phrase)
                if words:
                    (negative if negate_next else positive).append(("phrase", words))
                negate_next = False
                continue

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

            terms = [
                t for t in preprocess(raw, self.index.stemming)
                if t.strip("*") != ""
            ]
            if terms:
                target = negative if negate_next else positive
                target.extend(("term", t) for t in terms)
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

    def _phrase_docs(self, words):
        terms = stem_tokens(words) if self.index.stemming else words
        counts = self.index.phrase_counts(terms)
        # Ranking terms: the phrase's content words (stop words aren't indexed).
        content = remove_stop_words(words)
        content = stem_tokens(content) if self.index.stemming else content
        return counts, content

    def process(self, query):
        """Return (candidate doc IDs, matched term groups, matched phrases).

        A phrase entry is {"words": [...], "counts": {doc_id: occurrences}}.
        """
        positive, negative, use_or = self._parse(query)

        if not positive and not negative:
            return set(), [], []

        # Each positive query term becomes one group: a plain word is a group
        # of one, a wildcard is the group of terms it expanded to. The ranker
        # weights per group, so a broad wildcard cannot outvote the plain
        # words next to it. A phrase contributes one group per content word.
        matched_groups = []
        phrases = []

        if positive:
            doc_sets = []
            for kind, value in positive:
                if kind == "phrase":
                    counts, content = self._phrase_docs(value)
                    phrases.append({"words": value, "counts": counts})
                    matched_groups.extend([w] for w in content)
                    doc_sets.append(set(counts))
                else:
                    docs, expansions = self._docs_for(value)
                    if expansions:
                        matched_groups.append(expansions)
                    doc_sets.append(docs)
            result_docs = set(doc_sets[0])
            for docs in doc_sets[1:]:
                result_docs = result_docs | docs if use_or else result_docs & docs
        else:
            result_docs = set(self.index.doc_ids)

        for kind, value in negative:
            if kind == "phrase":
                docs = set(self._phrase_docs(value)[0])
            else:
                docs, _ = self._docs_for(value)
            result_docs = result_docs - docs

        return result_docs, matched_groups, phrases
