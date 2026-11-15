"""Phase 2: lexical features computed from the URL *string* only.

Safety: URLs are untrusted text. We split them with urllib.parse and count
characters. Nothing here resolves DNS, connects, or fetches anything.

Run on its own to write data/processed/url_features.csv:
    .venv/Scripts/python -m src.url_features
"""
import ipaddress
import math
from collections import Counter
from urllib.parse import urlsplit

import pandas as pd

from src.domain_utils import _EXTRACT  # offline tldextract instance

# Words common in phishing URLs (classic lexical-feature literature). Lower-case match.
SUSPICIOUS_WORDS = ["login", "log-in", "signin", "sign-in", "verify", "verification", "account",
                    "update", "secure", "security", "confirm", "password", "banking", "webscr",
                    "ebayisapi", "wp-", "admin", "auth", "session", "billing", "unlock", "support"]

NUMERIC_FEATURES = [
    "url_len", "host_len", "path_len", "query_len", "fragment_len",
    "digit_ratio", "special_ratio", "upper_ratio",
    "n_subdomains", "n_dots", "n_hyphens", "n_hyphens_host", "n_slashes", "path_depth",
    "n_query_params", "has_at", "has_port", "host_is_ip", "is_https", "has_www",
    "url_entropy", "host_entropy", "n_suspicious_words", "reg_domain_len", "reg_domain_digits",
    "tld_len", "path_has_file_ext", "is_root_path",
]
CATEGORICAL_FEATURES = ["tld"]


def entropy(s: str) -> float:
    """Shannon entropy in bits per character: random-looking strings score higher."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def url_features(url: str) -> dict:
    url = (url or "").strip()
    # Without a scheme, urlsplit puts the host in the path; add one only for parsing.
    parse_target = url if "://" in url else "http://" + url
    try:
        parts = urlsplit(parse_target)
        host = (parts.hostname or "").lower()
        port = parts.port  # may raise ValueError on garbage ports
    except ValueError:
        parts, host, port = urlsplit(""), "", None
    path, query, frag = parts.path, parts.query, parts.fragment
    ext = _EXTRACT(host) if host else None
    subdomain = ext.subdomain if ext else ""
    reg_domain = f"{ext.domain}.{ext.suffix}" if ext and ext.suffix else (ext.domain if ext else "")
    n = max(len(url), 1)
    low = url.lower()
    return {
        "url_len": len(url),
        "host_len": len(host),
        "path_len": len(path),
        "query_len": len(query),
        "fragment_len": len(frag),
        "digit_ratio": sum(ch.isdigit() for ch in url) / n,
        "special_ratio": sum(not ch.isalnum() for ch in url) / n,
        "upper_ratio": sum(ch.isupper() for ch in url) / n,
        "n_subdomains": len([s for s in subdomain.split(".") if s]),
        "n_dots": url.count("."),
        "n_hyphens": url.count("-"),
        "n_hyphens_host": host.count("-"),
        "n_slashes": url.count("/"),
        "path_depth": len([p for p in path.split("/") if p]),
        "n_query_params": len([q for q in query.split("&") if q]),
        "has_at": int("@" in url),
        "has_port": int(port is not None),
        "host_is_ip": int(is_ip(host)),
        "is_https": int(low.startswith("https://")),
        "has_www": int(host.startswith("www.")),
        "url_entropy": entropy(url),
        "host_entropy": entropy(host),
        "n_suspicious_words": sum(w in low for w in SUSPICIOUS_WORDS),
        "reg_domain_len": len(reg_domain),
        "reg_domain_digits": sum(ch.isdigit() for ch in reg_domain),
        "tld_len": len(ext.suffix) if ext else 0,
        "path_has_file_ext": int("." in path.rsplit("/", 1)[-1]),
        "is_root_path": int(path in ("", "/") and not query),
        "tld": (ext.suffix if ext and ext.suffix else "(none)"),
    }


def build_table(urls: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([url_features(u) for u in urls])


if __name__ == "__main__":
    idx = pd.read_csv("data/processed/index.csv", keep_default_na=False)
    table = build_table(idx.url)
    table.insert(0, "path", idx.path)
    table.to_csv("data/processed/url_features.csv", index=False)
    print(f"Wrote data/processed/url_features.csv: {table.shape}")
