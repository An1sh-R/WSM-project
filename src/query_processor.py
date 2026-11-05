import re

from src.text_processor import (
    preprocess, normalize, tokenize, remove_stop_words, stem_tokens,
)
from src.wildcard import has_wildcard, expand

# A quoted phrase, or any run of non-space characters.
_QUERY_TOKEN = re.compile(r'"([^"]*)"|(\S+)')
_PROXIMITY = re.compile(r"^/\d+$")


class QueryProcessor:
    def __init__(self, index):
        self.index = index

    def _phrase_words(self, text):
        # Wildcards are not supported inside a phrase; stop words are kept.
        return tokenize(normalize(text.replace("*", " ")))

    def _parse(self, query):
        """Return (positive, negative, use_or).

        Each entry is ("term", word), ("phrase", [words]) or
        ("near", word_a, word_b, distance).
        """
        positive = []
        negative = []
        use_or = False
        negate_next = False

        lexemes = [(m.group(1), m.group(2)) for m in _QUERY_TOKEN.finditer(query)]
        i = -1
        while i + 1 < len(lexemes):
            i += 1
            phrase, raw = lexemes[i]

            # "a /N b": a and b within N words of each other, in either order.
            if phrase is None and i + 2 < len(lexemes):
                gap_phrase, gap = lexemes[i + 1]
                right_phrase, right = lexemes[i + 2]
                if (gap_phrase is None and _PROXIMITY.match(gap)
                        and int(gap[1:]) >= 1 and right_phrase is None):
                    left_words = self._phrase_words(raw.lstrip("-"))
                    right_words = self._phrase_words(right)
                    if len(left_words) == 1 and len(right_words) == 1:
                        item = ("near", left_words[0], right_words[0], int(gap[1:]))
                        (negative if negate_next else positive).append(item)
                        negate_next = False
                        i += 2
                        continue

            if phrase is None and _PROXIMITY.match(raw):
                continue  # a stray /N with no word on both sides is ignored

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

    def _near_docs(self, a, b, distance):
        content = remove_stop_words([a, b])      # words worth scoring
        if self.index.stemming:
            a, b = stem_tokens([a, b])
            content = stem_tokens(content)
        return self.index.proximity_counts(a, b, distance), content

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
            for item in positive:
                kind, value = item[0], item[1]
                if kind == "near":
                    counts, content = self._near_docs(item[1], item[2], item[3])
                    phrases.append({"words": [item[1], item[2]], "counts": counts,
                                    "near": item[3]})
                    matched_groups.extend([w] for w in content)
                    doc_sets.append(set(counts))
                elif kind == "phrase":
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

        for item in negative:
            kind, value = item[0], item[1]
            if kind == "near":
                docs = set(self._near_docs(item[1], item[2], item[3])[0])
            elif kind == "phrase":
                docs = set(self._phrase_docs(value)[0])
            else:
                docs, _ = self._docs_for(value)
            result_docs = result_docs - docs

        return result_docs, matched_groups, phrases
