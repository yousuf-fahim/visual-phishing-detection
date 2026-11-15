"""Phase 1, step 3: near-duplicate clusters and leakage groups.

Run: .venv/Scripts/python -m src.dedup

Two samples end up in the same group if EITHER
  * their screenshots' 64-bit pHashes differ in <= HAMMING_MAX bits (near-duplicate), or
  * they share a registered domain (same site / same phishing kit host).
Groups are built with union-find, so links chain: A~B and B~C put A, B, C together.

Why: phishing kits are mass-deployed, so the "same" page appears many times. If one
copy is in train and another in test, the test score measures memory, not skill.

Outputs
  data/processed/groups.csv           path, phash_cluster, group_id
  results/phase1/leakage_report.json  cluster sizes, official-split overlap, cross-dataset overlap
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HAMMING_MAX = 6  # of 64 bits (~10%); a common near-duplicate threshold for pHash
OUT = Path("results/phase1")


class UnionFind:
    """Tiny union-find (disjoint sets): find() returns a group's representative."""

    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]  # path halving
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def near_duplicate_pairs(hashes: np.ndarray, max_bits: int) -> list[tuple[int, int]]:
    """All pairs (i < j) whose 64-bit hashes differ in <= max_bits bits.
    Done in chunks of rows so memory stays small (N x 512 comparisons at a time)."""
    pairs = []
    n = len(hashes)
    for start in range(0, n, 512):
        block = hashes[start:start + 512]
        dist = np.bitwise_count(block[:, None] ^ hashes[None, :])  # Hamming distance
        ii, jj = np.nonzero(dist <= max_bits)
        ii = ii + start
        keep = ii < jj
        pairs.extend(zip(ii[keep].tolist(), jj[keep].tolist()))
    return pairs


def main() -> None:
    idx = pd.read_csv("data/processed/index.csv", keep_default_na=False)
    feats = pd.read_csv("data/processed/image_features.csv", keep_default_na=False)
    assert (idx.path == feats.path).all(), "index.csv and image_features.csv out of sync"
    n = len(idx)
    hashes = np.array([int(h, 16) if h else 0 for h in feats.phash], dtype=np.uint64)

    # 1) near-duplicate screenshot clusters (pHash only)
    pairs = near_duplicate_pairs(hashes, HAMMING_MAX)
    uf_img = UnionFind(n)
    for a, b in pairs:
        uf_img.union(a, b)
    phash_cluster = np.array([uf_img.find(i) for i in range(n)])

    # 2) leakage groups: pHash links + same registered domain, WITHIN one dataset only.
    #    Splits are made per dataset, so a link to another dataset must not merge two
    #    groups here (that would reshuffle a dataset's split whenever another dataset is
    #    added). Cross-dataset duplicates are tracked through phash_cluster instead.
    ds = idx.dataset.to_numpy()
    uf = UnionFind(n)
    for a, b in pairs:
        if ds[a] == ds[b]:
            uf.union(a, b)
    for domain, members in idx.groupby("registered_domain").indices.items():
        if domain == "":  # no URL known (all of Phish-IRIS): only pHash links apply
            continue
        for m in members[1:]:
            uf.union(members[0], m)
    group = np.array([uf.find(i) for i in range(n)])

    out = pd.DataFrame({"path": idx.path, "phash_cluster": phash_cluster, "group_id": group})
    out.to_csv("data/processed/groups.csv", index=False)

    # ---- report ----
    df = idx.assign(phash_cluster=phash_cluster, group_id=group)
    clus_sizes = df.phash_cluster.value_counts()
    grp_sizes = df.group_id.value_counts()

    def describe_groups(col: str, top: int = 10) -> list[dict]:
        rows = []
        for gid, size in df[col].value_counts().head(top).items():
            g = df[df[col] == gid]
            rows.append({
                "size": int(size),
                "labels": {("phish" if k else "legit"): int(v) for k, v in g.label_binary.value_counts().items()},
                "top_domains": g.registered_domain.value_counts().head(3).to_dict(),
                "top_brands": g.loc[g.brand != "", "brand"].value_counts().head(3).to_dict(),
            })
        return rows

    # Groups that contain BOTH labels: a duplicate page labelled both ways is a label-noise warning.
    mixed = df.groupby("phash_cluster").label_binary.nunique()
    report = {
        "hamming_max_bits": HAMMING_MAX,
        "near_duplicate_pairs": len(pairs),
        "samples": n,
        "phash_clusters": int(len(clus_sizes)),
        "samples_in_multi_member_phash_clusters": int(clus_sizes[clus_sizes > 1].sum()),
        "phash_clusters_with_both_labels": int((mixed > 1).sum()),
        "samples_in_mixed_label_phash_clusters": int(df[df.phash_cluster.isin(mixed[mixed > 1].index)].shape[0]),
        "leakage_groups": int(len(grp_sizes)),
        "samples_in_multi_member_groups": int(grp_sizes[grp_sizes > 1].sum()),
        "largest_phash_clusters": describe_groups("phash_cluster"),
        "largest_groups": describe_groups("group_id"),
    }

    # Leakage across the OFFICIAL Phish360 split: how many test samples share a group with trainval?
    p = df[df.dataset == "phish360"]
    tv_groups = set(p.loc[p.official_split == "trainval", "group_id"])
    tv_clusters = set(p.loc[p.official_split == "trainval", "phash_cluster"])
    tv_domains = set(p.loc[p.official_split == "trainval", "registered_domain"])
    test = p[p.official_split == "test"]
    report["phish360_official_split_leakage"] = {
        "test_samples": len(test),
        "test_sharing_a_group_with_trainval": int(test.group_id.isin(tv_groups).sum()),
        "test_with_near_duplicate_screenshot_in_trainval": int(test.phash_cluster.isin(tv_clusters).sum()),
        "test_with_same_registered_domain_in_trainval": int(test.registered_domain.isin(tv_domains).sum()),
        "by_label_test_with_near_dup_in_trainval": {
            ("phish" if k else "legit"): int(v)
            for k, v in test[test.phash_cluster.isin(tv_clusters)].label_binary.value_counts().items()},
    }

    # Cross-dataset overlap (meaningful once Phish-IRIS is present)
    datasets = sorted(df.dataset.unique())
    if len(datasets) > 1:
        per_cluster = df.groupby("phash_cluster").dataset.nunique()
        report["cross_dataset_shared_phash_clusters"] = int((per_cluster > 1).sum())
    else:
        report["cross_dataset_shared_phash_clusters"] = f"n/a: only {datasets} present"

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "leakage_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
