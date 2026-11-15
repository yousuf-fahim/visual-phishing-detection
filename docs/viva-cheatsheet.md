# Viva cheat sheet

Short answers you should be able to say in your own words. Numbers come from `results/` (see `docs/report.md`).
Read `docs/learning-log.md` once, top to bottom, the night before.

## The one-paragraph story

A screenshot CNN (MobileNetV3) detects phishing on Phish360 about as well as URL features (test ROC-AUC 0.967 vs 0.961), and averaging the two works best (0.984, catching 77% of phishing at a 1% false-alarm rate). But trivial statistics such as file size already get 0.853, because Phish360's legitimate pages are busy homepages and its phishing pages are sparse login forms. Where that shortcut fails, the CNN drops to 0.809, and Grad-CAM shows it looks at empty space below forms, not at logos. It also loses about half its strict-threshold detection rate on another dataset or on non-English pages. So screenshots are a useful complement to URLs, but not a standalone detector. Our contribution is measuring this honestly, not a new architecture.

## 20 likely questions

1. **What is your contribution, given Phish360's authors already published a multimodal detector?**
   A generalization and shortcut study. We show *how much* of screenshot performance is a dataset shortcut, and how it holds up across datasets and languages. Fusion is a reproduction-scale ablation, not a claim.

2. **Why not just report accuracy?**
   With 40% phishing, accuracy hides the trade-off. Flagging legitimate sites is costly, so we report ROC-AUC (overall ranking) and the TPR at 1% FPR (how much phishing we catch if only 1 in 100 legit pages may be flagged), plus precision, recall and F1.

3. **Why did you make your own splits instead of the official one?**
   The official split leaks: 429 of 2,771 test samples (15.5%) have a near-duplicate screenshot or the same registered domain in training. We grouped by pHash cluster and domain, and zero groups cross our splits.

4. **What is a perceptual hash?**
   A 64-bit fingerprint of an image's coarse brightness pattern. Similar-looking images differ in only a few bits. We called pairs within 6 bits near-duplicates.

5. **What is a shortcut, and how did you detect it?**
   An easy feature that predicts the label without being the real cause. We trained a model on file size, whiteness, colour and dimensions only: it reached 0.853 on Phish360 and 0.995 on Phish-IRIS. Every real model must beat that floor.

6. **Then is your CNN just measuring file size?**
   Partly, not entirely. It beats the floor clearly (0.967 vs 0.853), but on the 288 test pages where the floor is wrong, it drops to 0.809. Grad-CAM, occlusion and the border test confirm it relies on page density and layout more than on brand logos.

7. **Why letterbox instead of stretching to 224x224?**
   Stretching distorts pages differently depending on their size, and the model could learn the distortion. Letterboxing keeps the content undistorted. For Phish-IRIS we additionally crop to 4:3, because the band width would otherwise reveal the aspect ratio.

8. **Why no horizontal flips?**
   Layout carries meaning (logo top-left, left-to-right text). A mirrored login page never occurs in reality, so flips would teach the model about impossible inputs.

9. **Why does pretraining help so much?**
   ImageNet features (edges, text strokes, shapes) transfer. With ~7,500 training images, the pretrained model learned in 3 epochs what the scratch CNN barely reached in 15 (0.967 vs 0.939).

10. **Why MobileNetV3 and not something bigger?**
    No GPU. MobileNetV3-small fine-tunes in about 17 min per run on CPU. ResNet18 and DeiT would take over 40 min per run, so we used them as frozen feature extractors (0.954 and 0.943).

11. **Early vs late fusion: what's the difference, and which won?**
    Early: concatenate features and train one network. Late: train separately and average probabilities. Late fusion won (validation 0.989), has nothing to overfit, and degrades gracefully if one input is missing.

12. **Why was early fusion worse on fine-tuned embeddings than on frozen ones?**
    The fine-tuned CNN partly memorised its training images, so its embeddings look cleaner on training data than on new data. The fusion head learned to over-trust them (0.977 vs 0.987 on frozen ResNet18).

13. **What is your top URL feature, and is it legitimate?**
    `has_www` (26% of gain). That's a collection artifact: legitimate URLs come from a top-site list written as www.example.com. Removing all suspected artifacts costs 0.03 AUC, and we report both versions.

14. **How did you test across time?**
    Phish360 has no per-sample dates, so only a dataset-level contrast was possible: train on Phish360 (2020-23) and test on Phish-IRIS (2018), and vice versa. ROC-AUC drops to 0.931 and 0.910. That mixes time with different label definitions, and we say so.

15. **How did you get page language?**
    Offline, from the visible HTML text with langdetect (scripts stripped, HTML never rendered). English-trained image model: 0.964 on English vs 0.940 on non-English, and TPR@1%FPR halves. The URL model doesn't drop.

16. **What does Grad-CAM show?**
    For phishing pages, the attention is below the login form and in the footer, not on the logo: only 12% of heatmap mass is in the header, vs 20% for an even spread. False positives are real login pages, and false negatives are clones of busy homepages.

17. **How robust is it?**
    Robust to JPEG compression (0.960 at quality 10) and grayscale (0.961). Fragile to heavy blur (0.793), strong brightening (0.877) and seeing only the top fifth of the page (0.736). Late fusion stays at 0.953 or above throughout.

18. **Brand identification: how good, and what about unknown brands?**
    On Phish-IRIS: 15-class accuracy 0.872, macro-F1 0.802. With 3 brands hidden from training, low softmax confidence flags them with AUROC 0.828: better than chance, not reliable. Softmax is known to be over-confident on unfamiliar inputs.

19. **How do you know the browser demo gives the same answers as Python?**
    We ported Pillow's resize to JavaScript and tested it: bit-identical on 23 images, probabilities equal to within 1e-6, and a self-test passes in headless Edge and Chrome. Int8 quantisation was rejected (AUC 0.964 to 0.633).

20. **What are the ethics and safety measures?**
    Only downloaded datasets; no live phishing URL was visited or rendered. URLs and HTML were treated as untrusted text. No raw screenshots or URLs are published, the demo has no phishing examples, and it states it is not a security product.

**Bonus: Can you show it?** Yes: https://yousuf-fahim.github.io/visual-phishing-detection/ with the images in `test-images/`.
Wikipedia, BBC and the BUBT homepage score 0%, while the *genuine* GitHub, Google, Microsoft, PayPal and Facebook login pages score
97.6-99.95%. That is the shortcut, live. Say it before they do: "This is why it must be combined with the URL."

**Bonus: What would you do next?** Add legitimate *login* pages to training and test data (the key missing piece), dated samples for a real cross-time test, an adversarial evaluation, and a logo-vs-domain consistency check like Phishpedia's.

## 5-slide outline (about 7 minutes)

1. **Problem and questions.** Phishing pages copy brands visually. Can a CNN detect them from screenshots, and does it *really* look at phishing cues? State RQ1-RQ4 and the two datasets (Phish360 2020-23, Phish-IRIS 2018).
2. **Doing the data right.** The official split leaks 15.5%, so we used grouped splits. Trivial statistics reach 0.853 / 0.995 (show `results/phase1/shortcut_features.png`). This sets the floor.
3. **Main result (RQ1).** The test table: URL 0.961, image 0.967, late fusion 0.984. TPR@1%FPR rises from 47-59% to 77% (show `results/phase4/ablation.png`).
4. **What it actually learned (RQ4).** Shortcut-hard pages: 0.967 to 0.809. Grad-CAM mean maps and the error examples: legitimate login pages flagged, homepage clones missed (show `results/phase7/gradcam_mean.png`).
5. **Generalization and takeaways (RQ2-3).** Cross-dataset 0.931 / 0.910, non-English 0.940, TPR halves (show `results/phase5/generalization.png`). Takeaway: screenshots complement URLs; benchmarks need legitimate login pages and shortcut baselines. Live demo: drop a Wikipedia page (0%), then the real PayPal login (99.95%).
