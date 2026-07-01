#!/usr/bin/env python3
"""
Offline scraper for sasi.ac.in. Run once (or whenever the site changes) to
refresh data/pages.jsonl - the raw text corpus that build_index.py embeds.

Not run by the bridge at call time: the bridge only ever reads the
already-built local index (data/index.json), so a live call never depends
on the college website being reachable.
"""
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

SITEMAP_URL = "https://sasi.ac.in/page-sitemap.xml"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
OUTPUT_PATH = "data/pages.jsonl"
REQUEST_DELAY_SECONDS = 0.5

# Pages that are administrative/legal boilerplate rather than content a
# caller would ask a voice agent about.
SKIP_URL_PATTERNS = (
    "grievance", "feedback", "-form", "error-test", "mandatory-disclosure",
    "hr-manual", "site-committees", "ic-committee",
)


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8", errors="replace")


def list_page_urls():
    xml_text = fetch(SITEMAP_URL)
    root = ET.fromstring(xml_text)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [loc.text.strip() for loc in root.findall(".//sm:loc", ns)]
    return [u for u in urls if not any(p in u for p in SKIP_URL_PATTERNS)]


def html_to_text(html):
    html = re.sub(r"<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.S | re.I)
    title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return title, text.strip()


def main():
    urls = list_page_urls()
    print(f"Found {len(urls)} pages to scrape", file=sys.stderr)
    with open(OUTPUT_PATH, "w") as out:
        for i, url in enumerate(urls, 1):
            try:
                html = fetch(url)
            except Exception as exc:
                print(f"[{i}/{len(urls)}] FAILED {url}: {exc}", file=sys.stderr)
                continue
            title, text = html_to_text(html)
            if len(text) < 100:
                print(f"[{i}/{len(urls)}] skipping {url} (too little text)", file=sys.stderr)
                continue
            out.write(json.dumps({"url": url, "title": title, "text": text}) + "\n")
            print(f"[{i}/{len(urls)}] {url} ({len(text)} chars)", file=sys.stderr)
            time.sleep(REQUEST_DELAY_SECONDS)


if __name__ == "__main__":
    main()
