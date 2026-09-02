import re

from src.document_loader import load_documents
from src.indexer import InvertedIndex
from src.query_processor import QueryProcessor
from src.ranker import Ranker


class SearchEngine:
    def __init__(self, documents_dir):
        self.documents_dir = documents_dir
        self.doc_filenames = {}
        self.doc_texts = {}
        self.index = None
        self.query_processor = None
        self.ranker = None

    def build_index(self):
        self.doc_filenames, self.doc_texts = load_documents(self.documents_dir)
        self.index = InvertedIndex()
        self.index.build(self.doc_texts)
        self.query_processor = QueryProcessor(self.index)
        self.ranker = Ranker(self.index)

    def preview(self, doc_id, length=200):
        text = re.sub(r"\s+", " ", self.doc_texts[doc_id]).strip()
        if len(text) > length:
            text = text[:length] + "..."
        return text

    def snippet(self, doc_id, terms, length=200):
        text = re.sub(r"\s+", " ", self.doc_texts[doc_id]).strip()
        lowered = text.lower()

        position = -1
        for term in terms:
            found = lowered.find(term)
            if found != -1 and (position == -1 or found < position):
                position = found

        if position == -1:
            return self.preview(doc_id, length)

        start = max(0, position - length // 2)
        end = start + length
        piece = text[start:end].strip()
        if start > 0:
            piece = "..." + piece
        if end < len(text):
            piece = piece + "..."
        return piece

    def search(self, query):
        if query is None or query.strip() == "":
            return {"status": "empty", "results": []}

        candidate_docs, matched_terms = self.query_processor.process(query)

        if not candidate_docs:
            if not matched_terms:
                return {"status": "only_stopwords", "results": []}
            return {"status": "no_match", "results": []}

        if matched_terms:
            ranked = self.ranker.rank(matched_terms, candidate_docs)
        else:
            ranked = [(doc_id, 0.0) for doc_id in sorted(candidate_docs)]

        max_score = ranked[0][1] if ranked else 0.0

        results = []
        for doc_id, score in ranked:
            relative = (score / max_score * 100) if max_score > 0 else 0.0
            results.append({
                "filename": self.doc_filenames[doc_id],
                "score": score,
                "relative": relative,
                "preview": self.snippet(doc_id, matched_terms),
            })

        return {"status": "ok", "results": results}
