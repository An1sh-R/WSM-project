# Mini Information Retrieval System (Web Search Mining)

A small, college-level Information Retrieval / Web Search Mining system in Python.
It demonstrates the classical components of a search engine: an inverted index,
Boolean (AND) query processing, wildcard matching, and TF-IDF + cosine-similarity
ranking, with a simple Tkinter GUI.

The document corpus is 25 plain-text documents: five Shakespeare plays from the
Folger Shakespeare Library (*All's Well That Ends Well*, *Antony and Cleopatra*,
*As You Like It*, *Hamlet*, *Romeo and Juliet*), each split into one document per act.

## 1. Project Objective

Show how a traditional search engine turns a collection of text documents and a
user query into a ranked list of relevant documents, using only classical IR
techniques (no embeddings, no machine learning, no external search services).

## 2. Information Retrieval Pipeline

```
Document Loading
   -> Tokenization
   -> Stop-word Removal
   -> Normalization
   -> Inverted Index Construction
   -> Query Processing
   -> Wildcard Processing
   -> Document Retrieval
   -> Ranking (TF-IDF + cosine similarity)
   -> Query-focused Snippet Generation
   -> GUI Display
```

Text diagram:

```
              +-------------------+
  .txt files  |  document_loader  |  doc_id -> filename, doc_id -> text
------------->|                   |
              +---------+---------+
                        |
                        v
              +-------------------+
              |  text_processor   |  normalize -> tokenize -> remove stop words
              +---------+---------+
                        |
                        v
              +-------------------+
              |     indexer       |  term -> {doc_ids}, term frequencies
              +---------+---------+
                        |
   query                v
  "king*"      +-------------------+
------------->| query_processor   |  preprocess query, expand wildcards,
              |   + wildcard      |  intersect posting lists (AND)
              +---------+---------+
                        |
                        v
              +-------------------+
              |      ranker       |  TF-IDF vectors, cosine similarity
              +---------+---------+
                        |
                        v
              +-------------------+
              |  search_engine    |  returns [{filename, score, snippet}]
              +---------+---------+
                        |
                        v
              +-------------------+
              |       gui         |  Tkinter: input box, button, results area
              +-------------------+
```

## 3. Architecture

The IR logic is fully separated from the GUI. The GUI only calls
`SearchEngine.search()` and never touches the inverted index directly.

```
project/
|
├── data/
│   └── documents/            25 Shakespeare act files (*.txt)
|
├── src/
│   ├── document_loader.py    read .txt files, assign integer doc IDs
│   ├── text_processor.py     normalize / tokenize / stop-word removal
│   ├── indexer.py            InvertedIndex class
│   ├── query_processor.py    QueryProcessor class (AND queries + wildcards)
│   ├── wildcard.py           wildcard -> regex -> matching vocabulary terms
│   ├── ranker.py             Ranker class (TF-IDF + cosine similarity)
│   └── search_engine.py      SearchEngine class (interface used by the GUI)
|
├── gui.py                    Tkinter GUI
├── main.py                   builds the index, launches the GUI
├── requirements.txt          (standard library only)
├── pyrightconfig.json        editor type-checking settings
└── README.md
```

| Module | Responsibility |
| --- | --- |
| `document_loader` | Read every `.txt` file, skip empty/unreadable ones, return `{doc_id: filename}` and `{doc_id: text}`. |
| `text_processor` | One reusable `preprocess(text)` used for both documents and queries. |
| `indexer` | Build `term -> set(doc_ids)`, keep per-document term counts and lengths, expose the vocabulary. |
| `wildcard` | Detect `*` in a term and expand it to matching vocabulary terms via a regular expression. |
| `query_processor` | Preprocess the query, expand wildcard terms, retrieve documents with AND semantics. |
| `ranker` | Compute TF-IDF weights and rank candidate documents by cosine similarity to the query. |
| `search_engine` | Wire everything together, build the result snippets; the only class the GUI talks to. |
| `gui` | Input box, Search button, Enter key, read-only results area. |

## 4. Tokenization

`text_processor.tokenize()` splits normalized text on whitespace into a list of
word tokens. Nothing more complex is needed for this corpus.

## 5. Stop-word Removal

`text_processor.remove_stop_words()` drops very common English words (plus a few
archaic Shakespearean function words such as *thou*, *thee*, *hath*, *doth*)
using a small hard-coded `STOP_WORDS` set. Stop words carry little meaning and
inflate the index, so removing them improves both speed and ranking quality.

## 6. Normalization

`text_processor.normalize()` does two simple things:

1. Lowercase the text (so `King` and `king` are the same term).
2. Replace every character that is not a letter, digit, whitespace, or `*`
   with a space (so punctuation is dropped). `*` is kept so wildcard queries
   survive preprocessing.

No stemming or lemmatization is done, to keep the behaviour easy to explain.

## 7. Inverted Indexing

`indexer.InvertedIndex` builds, once at startup:

* `index`: `term -> set of document IDs` (the posting lists), e.g.

  ```python
  {
      "love":  {4, 5, 9, 12, ...},
      "death": {2, 4, 5, ...},
      "king":  {6, 7, 8, ...},
  }
  ```

* `term_freq`: `doc_id -> Counter` of how often each term occurs in that document
  (used by the ranker for TF).
* `doc_length`: `doc_id -> token count`.

Methods: `build()`, `lookup(term)`, `vocabulary()`, `document_count()`,
`term_frequency(term, doc_id)`.

## 8. Query Processing

`query_processor.QueryProcessor.process(query)`:

1. `preprocess()` the query (same normalization / tokenization / stop-word
   removal as documents).
2. Drop terms that are empty or only `*`.
3. For each remaining term:
   * if it contains `*`, expand it to all matching vocabulary terms and take the
     **union** of their posting lists;
   * otherwise, look up its posting list directly.
4. Intersect the posting lists of all query terms (**AND** semantics): a
   document is a candidate only if it matches every query term.
5. Return the candidate document IDs and the list of concrete terms used
   (wildcard expansions included), which the ranker then scores.

Example: `love death` returns only documents that contain both *love* and *death*.

## 9. Wildcard Processing

`wildcard.expand(pattern, vocabulary)`:

1. Take the vocabulary from the inverted index.
2. Convert the wildcard pattern to a regular expression: escape the pattern,
   then replace `*` with `.*` and anchor it with `^...$`.
3. Return every vocabulary term that matches the regex.

`*` means "zero or more characters", so:

| Query | Matches (examples) |
| --- | --- |
| `learn*` | learn, learned, learning |
| `*ing`   | king, thing, nothing, morning |
| `mach*ne` | machine |

No permuterm or k-gram index is used; a linear scan over ~10k vocabulary terms is
instant at this scale.

## 10. TF-IDF Ranking

`ranker.Ranker` uses the classic vector space model.

* **Term Frequency** `tf(t, d)` = number of times term `t` appears in document `d`
  (raw count).
* **Inverse Document Frequency** `idf(t) = log(N / df(t))`, where `N` is the
  total number of documents and `df(t)` is the number of documents containing
  `t`. Rare terms get a higher weight; a term appearing in every document gets
  `idf = 0`.
* **TF-IDF weight** `w(t, d) = tf(t, d) * idf(t)`.

Both documents and the query are represented as vectors of TF-IDF weights over
the vocabulary.

## 11. Cosine Similarity

The score of a document `d` for a query `q` is the cosine of the angle between
their TF-IDF vectors:

```
                 sum over t of ( w(t, q) * w(t, d) )
cosine(q, d) = -------------------------------------------
                        ||q||  *  ||d||
```

where `||q||` and `||d||` are the vector lengths (Euclidean norms). This
normalizes for document length, so long documents are not automatically favoured.
Each document norm is computed once when the index is built. Results are sorted
by decreasing cosine similarity.

Cosine values are naturally small for short queries (a 1-2 word query vector
against a document vector of thousands of terms). To make the ranking easier to
read, each result also carries a **relative score**:

```
relative(d) = cosine(q, d) / max_cosine * 100
```

so the best-matching document shows 100% and the rest are shown as a percentage
of it. This is only a display aid; the ranking order is still decided by the raw
cosine similarity.

## 12. Query-Focused Snippets

`search_engine.SearchEngine.snippet()` builds the short text shown under each
result. Instead of always taking the start of the file, it finds the first place
any matched term (wildcard expansions included) appears in the document and
returns a ~200-character window centred on it, with `...` added at the ends when
the text is trimmed. If none of the terms can be located in the raw text it falls
back to the first ~200 characters (`preview()`).

## 13. GUI

`gui.SearchGUI` (Tkinter):

* a single-line search input box,
* a **Search** button (and the **Enter** key) that runs the search,
* a read-only, word-wrapped results area.

For each hit it shows the rank, filename, cosine score, relative score (%), and a
query-focused snippet (see section 12).
It shows a plain message (never a crash) for: empty queries, queries containing
only stop words, no matching documents, and wildcard patterns that match nothing.

## 14. How to Run the Project

Requirements: Python 3.8+ with Tkinter (included in standard CPython on Windows
and macOS; on Linux install `python3-tk`). No packages to install.

```
python main.py
```

The console prints the document and vocabulary counts, then the GUI window opens.

Quick check without the GUI:

```
python -c "from src.search_engine import SearchEngine; e=SearchEngine('data/documents'); e.build_index(); print(e.search('king*')['results'][:3])"
```

## 15. Example Queries

| Query | What it demonstrates |
| --- | --- |
| `love death` | multi-term AND query; *Romeo and Juliet* acts rank highest |
| `king crown` | AND query across the history/tragedy acts |
| `king*` | prefix wildcard (king, kingly, kingdom, kingdoms, kings) |
| `*ing` | suffix wildcard (thing, nothing, morning, king) |
| `mach*ne` | infix wildcard (matches *machine* in *Hamlet* Act 2) |
| `nobl*` | prefix wildcard (noble, nobler, nobly, noblest) |
| `lo*e` | infix wildcard (love, lose, lone, ...) |
| `the and of` | only stop words -> handled message |
| `machine learning` | no matching documents -> handled message |

Several of these (e.g. `love death`, `king*`, `*ing`) return many documents with
clearly different scores, which is useful for showing ranking behaviour.

## 16. Limitations

* Only AND queries are supported (no OR, NOT, phrase, or proximity queries).
* No stemming, so `love` and `loving` are different terms (use `lov*` instead).
* Raw term-frequency weighting; no sublinear TF scaling or BM25.
* Wildcard expansion is a linear scan of the vocabulary.
* The index is in memory only and rebuilt on every startup (fine for 25 documents).
* Snippets show the text around the first matched term only, with no highlighting
  and no handling of multiple matches.
* Ranking is purely lexical: it has no notion of meaning or synonyms.

## 17. Possible Future Improvements

* Add OR / NOT / phrase queries and a small Boolean parser.
* Add stemming (e.g. a simple Porter stemmer) as an optional step.
* Store term positions to support phrase and proximity search.
* Use sublinear TF weighting (`1 + log(tf)`) or BM25 for better ranking.
* Highlight the matched terms in the snippet and widen it around several matches.
* Persist the index to disk so startup is instant.
* Show which terms matched (including wildcard expansions) in the GUI.
