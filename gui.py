import tkinter as tk
from tkinter import ttk

# A calm, "old manuscript" palette that suits the Shakespeare corpus.
PAGE = "#faf8f4"
SURFACE = "#ffffff"
INK = "#2b2b2b"
MUTED = "#8a8175"
ACCENT = "#8c2f39"
ACCENT_SOFT = "#e9d6d8"
BORDER = "#e7e2d9"

UI_FONT = "Segoe UI"
TEXT_FONT = "Georgia"


class SearchGUI:
    def __init__(self, engine):
        self.engine = engine

        self.root = tk.Tk()
        self.root.title("Shakespeare Search")
        self.root.geometry("820x600")
        self.root.minsize(560, 420)
        self.root.configure(bg=PAGE)

        self._build_styles()
        self._build_header()
        self._build_search_bar()
        self._build_results()
        self._build_statusbar()

        self.entry.focus_set()

    # ----- construction -------------------------------------------------

    def _build_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Search.TEntry",
            fieldbackground=SURFACE,
            bordercolor=BORDER,
            lightcolor=BORDER,
            darkcolor=BORDER,
            insertcolor=INK,
            padding=8,
        )
        style.map("Search.TEntry", bordercolor=[("focus", ACCENT)])

        style.configure(
            "Search.TButton",
            background=ACCENT,
            foreground="#ffffff",
            font=(UI_FONT, 10, "bold"),
            borderwidth=0,
            focusthickness=0,
            padding=(18, 9),
        )
        style.map(
            "Search.TButton",
            background=[("active", "#73262f"), ("pressed", "#73262f")],
        )

    def _build_header(self):
        header = tk.Frame(self.root, bg=ACCENT)
        header.pack(fill="x")

        inner = tk.Frame(header, bg=ACCENT)
        inner.pack(fill="x", padx=24, pady=(16, 14))

        tk.Label(
            inner, text="Shakespeare Search", bg=ACCENT, fg="#ffffff",
            font=(TEXT_FONT, 19, "bold"),
        ).pack(anchor="w")

        count = self.engine.index.document_count()
        vocab = len(self.engine.index.vocabulary())
        tk.Label(
            inner,
            text="{} act-documents · {:,} distinct terms".format(count, vocab),
            bg=ACCENT, fg=ACCENT_SOFT, font=(UI_FONT, 9),
        ).pack(anchor="w", pady=(2, 0))

    def _build_search_bar(self):
        bar = tk.Frame(self.root, bg=PAGE)
        bar.pack(fill="x", padx=24, pady=(20, 6))

        self.entry = ttk.Entry(bar, style="Search.TEntry", font=(UI_FONT, 12))
        self.entry.pack(side="left", fill="x", expand=True, ipady=2)
        self.entry.bind("<Return>", self.on_search)

        ttk.Button(
            bar, text="Search", style="Search.TButton", command=self.on_search,
        ).pack(side="left", padx=(10, 0))

        tk.Label(
            self.root,
            text="love death  →  AND        love OR death        love NOT death  /  -death        king*  →  wildcard",
            bg=PAGE, fg=MUTED, font=(UI_FONT, 9),
        ).pack(anchor="w", padx=26, pady=(0, 14))

    def _build_results(self):
        frame = tk.Frame(self.root, bg=BORDER, bd=0)
        frame.pack(fill="both", expand=True, padx=24, pady=(0, 12))

        self.results = tk.Text(
            frame, wrap="word", state="disabled", bg=SURFACE, fg=INK,
            relief="flat", highlightthickness=1, highlightbackground=BORDER,
            highlightcolor=BORDER, padx=22, pady=18, cursor="arrow",
            font=(TEXT_FONT, 11), spacing1=2, spacing3=2,
        )
        scroll = ttk.Scrollbar(frame, command=self.results.yview)
        self.results.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.results.pack(side="left", fill="both", expand=True)

        self.results.tag_configure("rank", foreground=MUTED, font=(TEXT_FONT, 11))
        self.results.tag_configure("title", foreground=INK, font=(TEXT_FONT, 13, "bold"))
        self.results.tag_configure("meta", foreground=MUTED, font=(UI_FONT, 9))
        self.results.tag_configure("bar_fill", foreground=ACCENT, font=(UI_FONT, 9))
        self.results.tag_configure("bar_track", foreground=BORDER, font=(UI_FONT, 9))
        self.results.tag_configure(
            "snippet", foreground="#4a4a4a", font=(TEXT_FONT, 10, "italic"),
            lmargin1=14, lmargin2=14, spacing1=4, spacing3=10,
        )
        self.results.tag_configure(
            "notice", foreground=MUTED, font=(UI_FONT, 11), justify="center",
            spacing1=40,
        )

        self._show_notice("Enter a query above to search the collection.")

    def _build_statusbar(self):
        self.status = tk.Label(
            self.root, text="Ready", bg=PAGE, fg=MUTED, font=(UI_FONT, 9),
            anchor="w",
        )
        self.status.pack(fill="x", padx=26, pady=(0, 10))

    # ----- rendering --------------------------------------------------

    def _clear(self):
        self.results.config(state="normal")
        self.results.delete("1.0", "end")

    def _done(self):
        self.results.config(state="disabled")

    def _show_notice(self, text):
        self._clear()
        self.results.insert("end", text, "notice")
        self._done()

    def _relevance_bar(self, relative, cells=24):
        filled = int(round(relative / 100 * cells))
        filled = max(0, min(cells, filled))
        return "█" * filled, "█" * (cells - filled)

    def on_search(self, event=None):
        query = self.entry.get()
        response = self.engine.search(query)
        status = response["status"]

        notices = {
            "empty": "Please enter a query.",
            "only_stopwords": "Your query contains only stop words. Try different terms.",
            "no_positive_terms": "Your query only excludes terms. Add a word to search for.",
            "no_match": "No matching documents found.",
        }
        if status in notices:
            self._show_notice(notices[status])
            self.status.config(text="No results for “{}”".format(query.strip()))
            return

        results = response["results"]
        self._clear()
        for i, result in enumerate(results, start=1):
            fill, track = self._relevance_bar(result["relative"])
            self.results.insert("end", "{:>2}  ".format(i), "rank")
            self.results.insert("end", result["filename"] + "\n", "title")
            self.results.insert("end", fill, "bar_fill")
            self.results.insert("end", track + "  ", "bar_track")
            self.results.insert(
                "end",
                "{:.0f}%  relevance ·  score {:.4f}\n".format(
                    result["relative"], result["score"]),
                "meta",
            )
            self.results.insert("end", result["preview"] + "\n", "snippet")
        self._done()

        n = len(results)
        self.status.config(
            text="{} document{} for “{}”".format(
                n, "" if n == 1 else "s", query.strip()))

    def run(self):
        self.root.mainloop()
