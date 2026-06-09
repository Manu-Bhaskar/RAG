
import os
import arxiv
import re
from urllib.error import HTTPError, URLError

SAVE_DIR = "./data/docs"
TARGET_PAPERS = 100

QUERY = "machine learning OR deep learning OR neural networks OR transformers"


def clean_filename(text):
    return re.sub(r'[\\/*?:"<>|]', "", text)[:150]


def count_existing_pdfs(folder):
    return len([f for f in os.listdir(folder) if f.lower().endswith(".pdf")])


def download_arxiv_papers():
    os.makedirs(SAVE_DIR, exist_ok=True)

    existing_count = count_existing_pdfs(SAVE_DIR)
    print(f"📁 PDFs already in corpus: {existing_count}")

    if existing_count >= TARGET_PAPERS:
        print("✅ Target already reached. No downloads needed.")
        return

    search = arxiv.Search(
        query=QUERY,
        max_results=500,
        sort_by=arxiv.SortCriterion.Relevance
    )

    downloaded_now = 0

    for paper in search.results():

        total_now = count_existing_pdfs(SAVE_DIR)
        if total_now >= TARGET_PAPERS:
            break

        filename = f"{paper.get_short_id()} - {clean_filename(paper.title)}.pdf"
        path = os.path.join(SAVE_DIR, filename)

        if os.path.exists(path):
            print(f"⏩ Exists: {filename}")
            continue

        try:
            print(f"📄 Downloading ({total_now+1}/{TARGET_PAPERS}): {paper.title}")
            paper.download_pdf(dirpath=SAVE_DIR, filename=filename)
            downloaded_now += 1

        except (HTTPError, URLError):
            print("⚠️ Skipped broken link")
            continue

    print(f"\n✅ New PDFs downloaded this run: {downloaded_now}")
    print(f"📚 Total PDFs now: {count_existing_pdfs(SAVE_DIR)}")


if __name__ == "__main__":
    download_arxiv_papers()
