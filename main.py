import argparse
import os
import threading
import webbrowser

from src.search_engine import SearchEngine

DOCUMENTS_DIR = os.path.join("data", "documents")


def main():
    parser = argparse.ArgumentParser(description="Shakespeare search")
    parser.add_argument("--stem", action="store_true",
                        help="apply the Porter stemmer to documents and queries")
    parser.add_argument("--log-tf", action="store_true",
                        help="use sublinear term frequency (1 + log tf)")
    parser.add_argument("--tk", action="store_true",
                        help="use the Tkinter desktop window instead of the web page")
    parser.add_argument("--port", type=int, default=8000,
                        help="port for the web page (default 8000)")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open the browser automatically")
    args = parser.parse_args()

    engine = SearchEngine(DOCUMENTS_DIR, stemming=args.stem,
                          sublinear_tf=args.log_tf)
    engine.build_index()
    print("Index built. Documents:", engine.index.document_count(),
          "Vocabulary size:", len(engine.index.vocabulary()))

    if args.tk:
        from gui import SearchGUI
        SearchGUI(engine).run()
        return

    from server import serve
    server = serve(engine, args.port)
    url = "http://127.0.0.1:{}/".format(args.port)
    print("Serving on", url, "(Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
