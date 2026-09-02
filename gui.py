import tkinter as tk


class SearchGUI:
    def __init__(self, engine):
        self.engine = engine

        self.root = tk.Tk()
        self.root.title("Mini Information Retrieval System")
        self.root.geometry("700x500")

        top = tk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=10)

        tk.Label(top, text="Search:").pack(side="left")

        self.entry = tk.Entry(top)
        self.entry.pack(side="left", fill="x", expand=True, padx=5)
        self.entry.bind("<Return>", self.on_search)

        tk.Button(top, text="Search", command=self.on_search).pack(side="left")

        hint = tk.Label(
            self.root,
            text="love death (AND)    love OR death    love NOT death / -death    king* (wildcard)",
            fg="gray",
        )
        hint.pack(fill="x", padx=10)

        self.results = tk.Text(self.root, wrap="word", state="disabled")
        self.results.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def show_text(self, text):
        self.results.config(state="normal")
        self.results.delete("1.0", "end")
        self.results.insert("1.0", text)
        self.results.config(state="disabled")

    def on_search(self, event=None):
        query = self.entry.get()
        response = self.engine.search(query)
        status = response["status"]

        if status == "empty":
            self.show_text("Please enter a query.")
            return
        if status == "only_stopwords":
            self.show_text("Your query contains only stop words. Try different terms.")
            return
        if status == "no_match":
            self.show_text("No matching documents found.")
            return

        lines = []
        for i, result in enumerate(response["results"], start=1):
            lines.append("{}. {}     Score: {:.4f}     Relevance: {:.0f}%".format(
                i, result["filename"], result["score"], result["relative"]))
            lines.append("   " + result["preview"])
            lines.append("")

        self.show_text("\n".join(lines))

    def run(self):
        self.root.mainloop()
