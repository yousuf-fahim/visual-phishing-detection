"""Helpers for turning a URL string into its registered domain, fully offline.

Safety: URLs from the datasets are untrusted. We only parse them as text here.
Nothing in this module ever connects to the URL or to the internet.

tldextract normally downloads the latest Public Suffix List on first use.
Passing suffix_list_urls=() turns that off, so it uses the snapshot that ships
inside the tldextract package. cache_dir=None means it writes nothing to disk.
"""
import tldextract

# One shared, offline extractor for the whole project.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


def registered_domain(url: str) -> str:
    """Return e.g. 'example.co.uk' for 'http://login.example.co.uk/x'.

    Falls back to the bare host (e.g. an IP address) when there is no
    recognised public suffix, and to '' for empty or garbage input.
    """
    if not isinstance(url, str) or not url.strip():
        return ""
    parts = _EXTRACT(url.strip())
    if parts.domain and parts.suffix:
        return f"{parts.domain}.{parts.suffix}".lower()
    return (parts.domain or "").lower()
