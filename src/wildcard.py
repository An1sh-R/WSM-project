import re


def has_wildcard(term):
    return "*" in term


def expand(pattern, vocabulary):
    regex_text = "^" + re.escape(pattern).replace("\\*", ".*") + "$"
    regex = re.compile(regex_text)
    return [term for term in vocabulary if regex.match(term)]
