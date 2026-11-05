"""Porter stemmer (M. F. Porter, 1980), implemented from the original paper.

stem("relational") -> "relat", stem("loving") -> "love", stem("caresses") -> "caress"
"""


def _is_consonant(word, i):
    ch = word[i]
    if ch in "aeiou":
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(stem):
    """m in [C](VC){m}[V]: number of vowel-to-consonant transitions."""
    m = 0
    prev_vowel = False
    for i in range(len(stem)):
        vowel = not _is_consonant(stem, i)
        if prev_vowel and not vowel:
            m += 1
        prev_vowel = vowel
    return m


def _has_vowel(stem):
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(word):
    return (len(word) >= 2 and word[-1] == word[-2]
            and _is_consonant(word, len(word) - 1))


def _ends_cvc(word):
    """consonant-vowel-consonant where the last consonant is not w, x or y."""
    if len(word) < 3:
        return False
    return (_is_consonant(word, len(word) - 3)
            and not _is_consonant(word, len(word) - 2)
            and _is_consonant(word, len(word) - 1)
            and word[-1] not in "wxy")


def _replace(word, rules, min_measure):
    """Apply the first rule whose suffix matches (longest listed first).

    Returns (new_word, matched). A rule that matches the suffix but fails the
    measure condition still counts as matched, as in Porter's paper.
    """
    for suffix, replacement in rules:
        if word.endswith(suffix):
            stem = word[: len(word) - len(suffix)]
            if _measure(stem) > min_measure:
                return stem + replacement, True
            return word, True
    return word, False


def _step1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _step1b(word):
    if word.endswith("eed"):
        stem = word[:-3]
        return stem + "ee" if _measure(stem) > 0 else word

    for suffix in ("ed", "ing"):
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if not _has_vowel(stem):
                return word
            word = stem
            if word.endswith(("at", "bl", "iz")):
                return word + "e"
            if _ends_double_consonant(word) and word[-1] not in "lsz":
                return word[:-1]
            if _measure(word) == 1 and _ends_cvc(word):
                return word + "e"
            return word
    return word


def _step1c(word):
    if word.endswith("y") and _has_vowel(word[:-1]):
        return word[:-1] + "i"
    return word


_STEP2 = [
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
    ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
]

_STEP3 = [
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
]

_STEP4 = [
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
    "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
]


def _step4(word):
    # longest matching suffix wins
    for suffix in sorted(_STEP4, key=len, reverse=True):
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if suffix == "ion" and not stem.endswith(("s", "t")):
                return word
            return stem if _measure(stem) > 1 else word
    return word


def _step5(word):
    if word.endswith("e"):
        stem = word[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _ends_cvc(stem)):
            word = stem
    if _measure(word) > 1 and _ends_double_consonant(word) and word.endswith("l"):
        word = word[:-1]
    return word


def stem(word):
    if len(word) <= 2 or not word.isalpha():
        return word
    word = _step1a(word)
    word = _step1b(word)
    word = _step1c(word)
    word, _ = _replace(word, sorted(_STEP2, key=lambda r: -len(r[0])), 0)
    word, _ = _replace(word, sorted(_STEP3, key=lambda r: -len(r[0])), 0)
    word = _step4(word)
    word = _step5(word)
    return word
