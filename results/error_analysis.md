# Error analysis: screenshot model (Phase 7)

Model: fine-tuned MobileNetV3-small, full page. Data: Phish360 test split (1,590 pages).
Scores are seed-averaged image-only probabilities from `results/phase5/test_predictions.csv`.
Examples are the most confident errors, from `src/explain.py`. Per-example scores and Grad-CAM
region shares are in `results/phase7/screenshots/gradcam_examples.csv`, which is git-ignored because it sits next to real screenshots.
Sample IDs are Phish360's own IDs. **No URLs or domains are listed, on purpose.**

At threshold 0.5, the image model makes 41 false positives and 75 false negatives. Late fusion with the URL model
reduces that to 25 and 51 (`results/phase7/confusion_counts.csv`).

## Top false positives (legitimate pages flagged as phishing)

| Sample | Lang | p(image) | p(URL) | What the page is | Plausible reason |
|---|---|---|---|---|---|
| L06461 | en | 1.000 | 0.191 | A media site's "send us your video" submission form, with a few fields on a white page | Sparse form on a plain background looks exactly like a credential form |
| L03925 | en | 0.999 | 0.005 | A corporate portal sign-in box centred on an empty page | A legitimate login page: the very thing phishing pages copy |
| L06075 | es | 0.998 | 0.446 | A Spanish-language user-authentication page with two buttons | A legitimate login page; little else on the page |
| L04505 | ca | 0.994 | 0.771 | A webmail login for a regional telecom | A legitimate login page; the URL model is also unsure |
| L04951 | en | 0.992 | 0.017 | A full-screen photo background with a centred sign-in card | The classic phishing-kit layout, used here by the real site |

**Pattern:** every confident false positive is a *legitimate login or form page*. Phish360's legitimate class is mostly
homepages, so the model learned "sparse page with a form" means phishing. In 4 of the 5 cases the URL model correctly
scores the page as legitimate (p(URL) < 0.5), which is why fusion helps.

## Top false negatives (phishing pages missed)

| Sample | Brand | Lang | p(image) | p(URL) | What the page is | Plausible reason |
|---|---|---|---|---|---|---|
| P18997 | paypal | en | 0.000 | 0.031 | A copy of a busy customer-service information page, not a login form | Dense, content-rich page: looks like a homepage |
| P14120 | wellsfargo | en | 0.000 | 0.962 | A long text "disclaimer" page with the brand header | Text-heavy; the URL gives it away instead |
| P11923 | wellsfargo | en | 0.000 | 0.249 | A full copy of the bank's homepage, including the small login box | A faithful homepage clone; both models miss it |
| P14499 | apple | en | 0.000 | 0.552 | A product marketing page (large phone photo) | Looks like a normal brand homepage |
| P16279 | itau | pt | 0.000 | 0.992 | A full copy of the bank's homepage with promotional banners | Faithful clone; the URL model catches it |

**Pattern:** the missed phishing pages are *complete homepage clones* rather than bare login forms. They're dense,
colourful and multi-section, the opposite of the shortcut the model learned.

## What this means

The two error types are mirror images of the density shortcut found in Phase 1 (trivial features reach ROC-AUC 0.85) and
measured in Phase 5 (image ROC-AUC falls from 0.967 to 0.809 on shortcut-hard pages). Grad-CAM agrees: on correctly
detected phishing pages, the heatmap concentrates on the empty band below the form and the footer, not on the brand logo
(`results/phase7/region_summary.csv`). A detector meant for real use would need legitimate *login* pages in its training data.
