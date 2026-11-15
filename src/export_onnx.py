"""Phase 8: export the screenshot model for the browser demo, and prepare parity tests.

Run: .venv/Scripts/python -m src.export_onnx

1. Wrap the fine-tuned MobileNetV3 (full page, seed 42) so its input is the letterboxed
   224x224 image as float 0-255, NCHW. Scaling to 0-1 and ImageNet mean/std normalisation
   happen INSIDE the ONNX graph, so the browser only has to letterbox. Output: p(phishing).
2. Export to models/image_model.onnx and check it against PyTorch on 20 validation images.
3. Try dynamic int8 quantisation; keep it only if it shrinks the file meaningfully and val ROC-AUC
   moves by less than 0.002. (Dynamic quantisation mostly targets fully connected layers,
   so for a small conv net the gain may be small.)
4. Parity fixtures for web/test/parity.mjs (git-ignored, in data/processed/js_parity/): for 20
   validation screenshots, the decoded RGB pixels, Python's 224x224 letterbox and the probability.
   Plus a synthetic, non-phishing test image (web/public/selftest.png) and its expected
   probability, for the in-browser self-test.
5. web/public/model_card.json: every number the demo shows, read from results/ files.

Outputs: models/image_model.onnx, web/public/{image_model.onnx, model_card.json, selftest.png},
         results/phase8/export_check.json
"""
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch import nn

from src.metrics import binary_metrics
from src.preprocess_images import letterbox
from src.train_fusion import load_image_model

OUT = Path("results/phase8")
PUBLIC = Path("web/public")
PARITY = Path("data/processed/js_parity")


class BrowserModel(nn.Module):
    """float 0-255 NCHW in, p(phishing) out. Normalisation lives inside the graph."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        return torch.sigmoid(self.net((x - self.mean) / self.std)).squeeze(1)


def to_nchw(img224: np.ndarray) -> np.ndarray:
    return img224.transpose(2, 0, 1)[None].astype(np.float32)


def synthetic_page() -> Image.Image:
    """A plain drawn 'page' (header bar, boxes, text-like lines). Not a real site, not phishing."""
    im = Image.new("RGB", (1280, 960), (250, 250, 250))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 1280, 90], fill=(40, 70, 140))
    d.rectangle([60, 140, 760, 520], fill=(220, 230, 245), outline=(120, 120, 120), width=3)
    for i in range(12):
        d.rectangle([820, 150 + i * 40, 820 + 300 - (i % 4) * 50, 166 + i * 40], fill=(90, 90, 90))
    d.ellipse([500, 600, 760, 860], fill=(230, 120, 40))
    return im


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    PARITY.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)

    model = BrowserModel(load_image_model(42).net).eval()
    onnx_path = Path("models/image_model.onnx")
    torch.onnx.export(model, torch.zeros(1, 3, 224, 224), str(onnx_path), input_names=["image"],
                      output_names=["p_phish"], dynamic_axes={"image": {0: "batch"}, "p_phish": {0: "batch"}},
                      opset_version=17, dynamo=False)

    # ---- validation images: ONNX vs PyTorch, and quantisation check ----
    s = pd.read_csv("data/processed/splits.csv", keep_default_na=False)
    idx = pd.read_csv("data/processed/index.csv", keep_default_na=False)
    va = np.flatnonzero((s.dataset == "phish360") & (s.exclude == "") & (s.split == "val"))
    images = np.load("data/processed/img224_full.npy", mmap_mode="r")
    y = s.label_binary.to_numpy()[va]

    def onnx_scores(path, rows):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return np.concatenate([sess.run(None, {"image": np.asarray(images[rows[i:i + 64]]).transpose(0, 3, 1, 2)
                                                .astype(np.float32)})[0] for i in range(0, len(rows), 64)])

    rng = np.random.default_rng(42)
    sample = np.sort(np.concatenate([rng.choice(va[y == 0], 10, replace=False), rng.choice(va[y == 1], 10, replace=False)]))
    with torch.no_grad():
        torch_p = model(torch.from_numpy(np.asarray(images[sample]).transpose(0, 3, 1, 2).astype(np.float32))).numpy()
    onnx_p = onnx_scores(onnx_path, sample)
    full_fp32 = onnx_scores(onnx_path, va)

    from onnxruntime.quantization import QuantType, quantize_dynamic
    q_path = Path("models/image_model_int8.onnx")
    quantize_dynamic(str(onnx_path), str(q_path), weight_type=QuantType.QInt8)
    full_int8 = onnx_scores(q_path, va)
    auc32, auc8 = binary_metrics(y, full_fp32)["roc_auc"], binary_metrics(y, full_int8)["roc_auc"]
    size32, size8 = onnx_path.stat().st_size, q_path.stat().st_size
    use_int8 = (size8 < 0.6 * size32) and abs(auc32 - auc8) < 0.002
    chosen = q_path if use_int8 else onnx_path
    shutil.copy(chosen, PUBLIC / "image_model.onnx")

    check = {"torch_vs_onnx_max_abs_diff_20_samples": float(np.abs(torch_p - onnx_p).max()),
             "val_roc_auc_fp32": round(auc32, 4), "val_roc_auc_int8": round(auc8, 4),
             "size_mb_fp32": round(size32 / 1e6, 2), "size_mb_int8": round(size8 / 1e6, 2),
             "shipped": "int8" if use_int8 else "fp32"}
    print(json.dumps(check, indent=2))
    assert check["torch_vs_onnx_max_abs_diff_20_samples"] < 1e-4, "ONNX export does not match PyTorch"

    # ---- parity fixtures (real screenshots: git-ignored folder, never copied into web/) ----
    shipped = ort.InferenceSession(str(chosen), providers=["CPUExecutionProvider"])
    fixtures = []
    for k, r in enumerate(sample):
        with Image.open(idx.path[r]) as im:
            rgb = np.asarray(im.convert("RGB"))
        (PARITY / f"{k:02d}_rgb.bin").write_bytes(rgb.tobytes())
        lb = letterbox(Image.fromarray(rgb))
        assert (lb == images[r]).all()  # same function as the training arrays
        (PARITY / f"{k:02d}_224.bin").write_bytes(lb.tobytes())
        p = float(shipped.run(None, {"image": to_nchw(lb)})[0][0])
        fixtures.append({"id": k, "width": rgb.shape[1], "height": rgb.shape[0], "p_phish": p})
    # synthetic, odd-sized variants too, to exercise rounding and both resize passes
    page = synthetic_page()
    for k, size in enumerate([(1280, 960), (1003, 1500), (777, 333)], start=len(fixtures)):
        rgb = np.asarray(page.resize(size, Image.Resampling.BILINEAR))
        (PARITY / f"{k:02d}_rgb.bin").write_bytes(rgb.tobytes())
        lb = letterbox(Image.fromarray(rgb))
        (PARITY / f"{k:02d}_224.bin").write_bytes(lb.tobytes())
        fixtures.append({"id": k, "width": rgb.shape[1], "height": rgb.shape[0],
                         "p_phish": float(shipped.run(None, {"image": to_nchw(lb)})[0][0])})
    (PARITY / "fixtures.json").write_text(json.dumps(fixtures, indent=2))

    # browser self-test image (synthetic only) + expected output
    page.save(PUBLIC / "selftest.png")
    with Image.open(PUBLIC / "selftest.png") as im:
        p_self = float(shipped.run(None, {"image": to_nchw(letterbox(im.convert("RGB")))})[0][0])
    check["selftest_expected_p_phish"] = p_self
    (OUT / "export_check.json").write_text(json.dumps(check, indent=2))

    # ---- model card: every number shown in the UI, read from results/ ----
    final = pd.read_csv("results/final_in_distribution.csv").set_index(["model", "url_features"])
    hard = pd.read_csv("results/phase5/final_shortcut_hard.csv").set_index(["model", "url_features"])
    cd = pd.read_csv("results/phase5/cross_dataset.csv")
    cl = pd.read_csv("results/phase5/cross_language.csv")
    img = final.loc[("image_only", "-")]
    card = {
        "model": "MobileNetV3-small, ImageNet-pretrained, fine-tuned on Phish360 screenshots (seed 42)",
        "shipped_precision": check["shipped"],
        "selftest_expected_p_phish": round(p_self, 6),
        "test_roc_auc": round(img.roc_auc, 3),
        "test_tpr_at_1pct_fpr": round(img.tpr_at_1pct_fpr, 3),
        "shortcut_hard_roc_auc": round(hard.loc[("image_only", "-")].roc_auc, 3),
        "cross_dataset_roc_auc": round(cd[(cd.train_dataset == "phish360") & (cd.test_dataset == "phish-iris") &
                                          cd.condition.str.contains("removed")].roc_auc.iloc[0], 3),
        "english_trained_non_english_roc_auc": round(cl[(cl.trained_on == "English only") & (cl.model == "image_only") &
                                                        (cl.test_group == "non-English")].roc_auc.iloc[0], 3),
        "english_trained_english_roc_auc": round(cl[(cl.trained_on == "English only") & (cl.model == "image_only") &
                                                    (cl.test_group == "English")].roc_auc.iloc[0], 3),
        "late_fusion_test_roc_auc": round(final.loc[("late_fusion", "full")].roc_auc, 3),
        "sources": ["results/final_in_distribution.csv", "results/phase5/final_shortcut_hard.csv",
                    "results/phase5/cross_dataset.csv", "results/phase5/cross_language.csv",
                    "results/phase8/export_check.json"],
    }
    (PUBLIC / "model_card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
