# Phase 9: Report and viva prep (90-120 min)

Run after /clear. Paste everything below the line.

Final checklist: every number in the report traces to a file in results/, no raw screenshots or phishing content in the public repo, and you can explain every item in docs/viva-cheatsheet.md in your own words.

---

Phase 9: write-up.

1. README.md: project summary, research questions, results table, how to reproduce (exact commands, without distributing the data), dataset credits and citations, and the demo link if Phase 8 was done.
2. docs/report.md, 6-8 pages in lab-report style: Abstract, Introduction, Related work (only sources I can verify and cite; mention that Phish360's authors published a multimodal method, so ours is a generalization study), Datasets (including leakage and shortcut handling), Methods, Experimental setup, Results (ablation and generalization), Explainability and robustness, Limitations, Conclusion. Every figure and number must come from results/. Use only claims our experiments support.
3. Limitations must be honest: dataset bias and label noise, short-lived phishing pages, the Phish-IRIS label-mapping assumption, no adversarial evasion testing, results from specific collection periods, and that this is research, not a security product.
4. Ethics section: datasets used under their terms, no live phishing pages visited, no raw phishing content published.
5. Convert the report to PDF if a converter is available. Otherwise tell me what to install.
6. docs/viva-cheatsheet.md: the 20 most likely faculty questions with short answers based on our learning log and results, plus a 5-slide outline for a short presentation.
7. Final cleanup, make sure the repo runs from scratch with the README given the datasets, and commit.
