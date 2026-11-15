"""Phase 5 helper: detect each page's language from its saved HTML.

Run: .venv/Scripts/python -m src.page_language

Safety: dataset HTML is untrusted. We parse it as a plain string with Python's
built-in html.parser (no browser, nothing executed, no network). <script> and
<style> contents are thrown away, and any text inside the page is treated as data
to measure, never as instructions.

Two signals per page:
  html_lang   the page's own <html lang="..."> attribute (if any), first 2 letters
  langdetect  language detected from the visible text (first 5,000 characters)
The final `language` is langdetect's answer when it has >= 30 letters of text,
otherwise html_lang, otherwise 'unknown'.

Output: data/processed/languages.csv  (path, html_lang, detected, detect_prob, n_text_chars, language)
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

MAX_HTML_BYTES = 2_000_000  # read at most 2 MB per page
MIN_LETTERS = 30


def page_language(html_path: str) -> dict:
    from langdetect import DetectorFactory, detect_langs
    DetectorFactory.seed = 42  # langdetect is randomised; fix it for reproducibility
    out = {"html_lang": "", "detected": "", "detect_prob": 0.0, "n_text_chars": 0}
    p = Path(html_path)
    if not p.is_file():
        return out
    raw = p.read_bytes()[:MAX_HTML_BYTES].decode("utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    if soup.html and soup.html.get("lang"):
        out["html_lang"] = str(soup.html.get("lang")).strip().lower()[:2]
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ").split())[:5000]
    out["n_text_chars"] = len(text)
    if sum(ch.isalpha() for ch in text) >= MIN_LETTERS:
        try:
            best = detect_langs(text)[0]
            out["detected"], out["detect_prob"] = best.lang[:2], round(best.prob, 3)
        except Exception:
            pass
    return out


def main() -> None:
    idx = pd.read_csv("data/processed/index.csv", keep_default_na=False)
    with ProcessPoolExecutor() as pool:
        res = list(tqdm(pool.map(page_language, idx.html_path, chunksize=16), total=len(idx), mininterval=15))
    df = pd.DataFrame(res)
    df.insert(0, "path", idx.path)
    df["language"] = df.detected.where(df.detected != "", df.html_lang).replace("", "unknown")
    df.to_csv("data/processed/languages.csv", index=False)
    print(df.language.value_counts().head(15).to_string())
    print(f"agreement detected vs html_lang (both present): "
          f"{(df.detected == df.html_lang)[(df.detected != '') & (df.html_lang != '')].mean():.3f}")


if __name__ == "__main__":
    main()
