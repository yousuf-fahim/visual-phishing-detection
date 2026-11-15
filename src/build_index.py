"""Phase 1, step 1: build one combined index of every sample.

Run: .venv/Scripts/python -m src.build_index

Outputs
  data/processed/index.csv           one row per sample (both datasets)
  data/processed/label_mapping.json  how folder names map to labels (PROVISIONAL)
  results/phase1/dataset_summary.json
  results/phase1/class_counts.png
  results/phase1/screenshots/sample_grid.png   (git-ignored: contains real screenshots)

Safety: URL files are read as plain text only. HTML files are never opened
here; we only check that they exist. Nothing is fetched from the internet.
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.domain_utils import registered_domain

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
OUT = Path("results/phase1")


def read_text(path: Path) -> str:
    """Read a small text file as an untrusted string. 'utf-8-sig' drops the BOM."""
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8-sig", errors="replace").strip()


def image_info(path: Path) -> dict:
    """Width, height and file size, plus whether the image decodes at all."""
    info = {"img_ok": False, "width": None, "height": None, "file_bytes": None}
    if not path.is_file():
        return info
    info["file_bytes"] = path.stat().st_size
    try:
        with Image.open(path) as im:
            info["width"], info["height"] = im.size
            im.verify()  # checks the file is not truncated / corrupt
        info["img_ok"] = True
    except Exception:
        pass
    return info


def index_phish360() -> list[dict]:
    """Phish360 layout: <split>/<ID>_<class>/{Label,URL,RAW-HTML,SCREEN-SHOT}/...

    The folder suffix is the label: 'legitimate', or a brand name (trainval),
    or the generic 'phish' (official test split, where brands are not given).
    """
    root = RAW / "phish360"
    rows = []
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for sample in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            sample_id, folder_class = sample.name.split("_", 1)
            is_legit = folder_class == "legitimate"
            label_txt = read_text(sample / "Label" / "label.txt")
            url = read_text(sample / "URL" / "url.txt")
            shot = sample / "SCREEN-SHOT" / "screen_shoot.png"
            rows.append({
                "dataset": "phish360",
                "sample_id": sample_id,
                "official_split": split_dir.name,          # trainval / test
                "path": shot.as_posix(),
                "html_path": (sample / "RAW-HTML" / "index.html").as_posix(),
                "html_present": (sample / "RAW-HTML" / "index.html").is_file(),
                "label_binary": 0 if is_legit else 1,     # 1 = phishing
                "brand": "" if is_legit or folder_class == "phish" else folder_class,
                "folder_class": folder_class,
                # label.txt should agree with the folder name; record when it doesn't
                "label_txt_agrees": label_txt.lower() == folder_class.lower(),
                "label_txt_empty": label_txt == "",
                "url": url,
                "registered_domain": registered_domain(url),
                "language": "",   # not provided; detected from page text in Phase 5
                "date": "",       # not provided per sample; dataset covers 2020-2023
                **image_info(shot),
            })
    return rows


def index_phish_iris() -> list[dict]:
    """Phish-IRIS has not been downloaded yet. Its layout is parsed once it's in
    data/raw/phish-iris (written after inspecting the real files, not guessed)."""
    if not (RAW / "phish-iris").exists():
        print("Phish-IRIS not found in data/raw/phish-iris: skipped for now.")
        return []
    raise NotImplementedError("Inspect the Phish-IRIS layout, then implement this parser.")


def save_class_chart(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 4]})
    # Left: binary label per dataset/split
    counts = df.groupby(["dataset", "official_split", "label_binary"]).size().unstack(fill_value=0)
    counts.columns = ["legitimate", "phishing"]
    counts.plot.bar(ax=axes[0], color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Samples per label")
    axes[0].set_xlabel("")
    # Right: phishing brands (only where brand is known)
    brands = df.loc[df.brand != "", "brand"].value_counts()
    brands.plot.bar(ax=axes[1], color="#DD8452")
    axes[1].set_title(f"Phishing samples per brand ({len(brands)} brands, brand known only)")
    fig.tight_layout()
    fig.savefig(OUT / "class_counts.png", dpi=120)
    plt.close(fig)


def save_sample_grid(df: pd.DataFrame, n_per_class: int = 6) -> None:
    """A small grid of real screenshots, for our eyes only (git-ignored folder)."""
    ok = df[df.img_ok]
    picks = pd.concat([
        ok[ok.label_binary == lab].sample(n_per_class, random_state=42) for lab in (0, 1)
    ])
    fig, axes = plt.subplots(2, n_per_class, figsize=(3 * n_per_class, 6))
    for ax, (_, row) in zip(axes.ravel(), picks.iterrows()):
        with Image.open(row.path) as im:
            im.thumbnail((400, 400))
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"{'phish: ' + (row.brand or '?') if row.label_binary else 'legit'}\n"
                     f"{row.width}x{row.height}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    (OUT / "screenshots").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "screenshots" / "sample_grid.png", dpi=100)
    plt.close(fig)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(index_phish360() + index_phish_iris())
    df.to_csv(PROCESSED / "index.csv", index=False)

    mapping = {
        "status": "PROVISIONAL: verify (see docs/QUESTIONS.md)",
        "label_binary": {"0": "legitimate", "1": "phishing"},
        "phish360": {
            "source": "sample folder name '<ID>_<class>'; label.txt only cross-checked",
            "legitimate": "folder class 'legitimate' (ID prefix L)",
            "phishing": "any other folder class: a brand name (trainval) or 'phish' (official test, brand hidden)",
        },
        "phish_iris": "pending download",
    }
    (PROCESSED / "label_mapping.json").write_text(json.dumps(mapping, indent=2))

    summary = {}
    for name, g in df.groupby("dataset"):
        summary[name] = {
            "samples": len(g),
            "by_official_split_and_label": {
                f"{s}/{'phish' if l else 'legit'}": int(n)
                for (s, l), n in g.groupby(["official_split", "label_binary"]).size().items()},
            "phishing_fraction": round(g.label_binary.mean(), 4),
            "brands_known": int((g.brand != "").sum()),
            "n_brands": int(g.loc[g.brand != "", "brand"].nunique()),
            "corrupt_or_missing_images": int((~g.img_ok).sum()),
            "missing_html": int((~g.html_present).sum()),
            "label_txt_empty": int(g.label_txt_empty.sum()),
            "label_txt_disagrees_nonempty": int((~g.label_txt_agrees & ~g.label_txt_empty).sum()),
            "empty_url": int((g.url == "").sum()),
            "empty_registered_domain": int((g.registered_domain == "").sum()),
            "unique_registered_domains": int(g.registered_domain.nunique()),
            "width_counts_top5": {str(k): int(v) for k, v in Counter(g.width).most_common(5)},
            "height_counts_top5": {str(k): int(v) for k, v in Counter(g.height).most_common(5)},
            "size_by_label_median": {
                ("phish" if l else "legit"): {"width": float(x.width.median()), "height": float(x.height.median()),
                                              "file_kb": round(x.file_bytes.median() / 1024, 1)}
                for l, x in g.groupby("label_binary")},
        }
    (OUT / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    save_class_chart(df)
    save_sample_grid(df)
    print(f"Wrote {PROCESSED/'index.csv'} ({len(df)} rows) and results/phase1/*")


if __name__ == "__main__":
    main()
