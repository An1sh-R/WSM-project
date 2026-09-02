import os


def load_documents(directory):
    doc_filenames = {}
    doc_texts = {}
    next_id = 1

    filenames = sorted(f for f in os.listdir(directory) if f.endswith(".txt"))

    for filename in filenames:
        path = os.path.join(directory, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError):
            print("Skipping unreadable file:", filename)
            continue

        if text.strip() == "":
            print("Skipping empty file:", filename)
            continue

        doc_filenames[next_id] = filename
        doc_texts[next_id] = text
        next_id += 1

    return doc_filenames, doc_texts
