import re

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "i", "in", "is", "it", "its",
    "me", "my", "no", "nor", "not", "o", "of", "on", "or", "our", "s", "she",
    "so", "than", "that", "the", "thee", "their", "them", "then", "there",
    "these", "they", "this", "thou", "thy", "to", "too", "us", "was", "we",
    "were", "what", "when", "which", "who", "will", "with", "would", "you",
    "your", "ye", "hath", "doth", "shall", "upon",
}


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s*]", " ", text)
    return text


def tokenize(text):
    return text.split()


def remove_stop_words(tokens):
    return [t for t in tokens if t not in STOP_WORDS]


def preprocess(text):
    text = normalize(text)
    tokens = tokenize(text)
    tokens = remove_stop_words(tokens)
    return tokens
