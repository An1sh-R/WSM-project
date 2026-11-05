import argparse
import os

from src.search_engine import SearchEngine
from gui import SearchGUI

DOCUMENTS_DIR = os.path.join("data", "documents")


def main():
    parser = argparse.ArgumentParser(description="Shakespeare search")
    parser.add_argument("--stem", action="store_true",
                        help="apply the Porter stemmer to documents and queries")
    parser.add_argument("--log-tf", action="store_true",
                        help="use sublinear term frequency (1 + log tf)")
    args = parser.parse_args()

    engine = SearchEngine(DOCUMENTS_DIR, stemming=args.stem,
                          sublinear_tf=args.log_tf)
    engine.build_index()
    print("Index built. Documents:", engine.index.document_count(),
          "Vocabulary size:", len(engine.index.vocabulary()))

    gui = SearchGUI(engine)
    gui.run()


if __name__ == "__main__":
    main()
