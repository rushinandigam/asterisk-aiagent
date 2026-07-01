#!/usr/bin/env python3
"""
Parses the EAPCET rank-cutoff PDF (previous year's branch/category/gender-wise
admission rank ranges) and upserts one Q&A-framed chunk per branch into the
*existing* Qdrant collection, alongside the scraped website content.

Unlike upload_qdrant.py (which recreate_collection()s from data/index.json),
this script only ever upserts - it must never wipe the collection, since the
scraped sasi.ac.in content already lives there.

Usage:
    OPENAI_API_KEY=sk-... QDRANT_URL=... QDRANT_API_KEY=... \
      .venv/bin/python ingest_admissions_pdf.py "/path/to/RANK POSITION 2025.pdf"
"""
import json
import os
import sys
import urllib.request
import uuid

import pdfplumber
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "sasi_college")
SOURCE_LABEL = "EAPCET 2025 Rank Cutoffs"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

# Column order as laid out in the PDF table, split across its two pages.
CATEGORIES = ["OC", "EWS", "BC-A", "BC-B", "BC-C", "BC-D", "BC-E"]


def _cell(row, i):
    val = row[i] if i < len(row) else None
    val = (val or "").strip()
    return val if val and val != "-" else None


def parse_pdf(path):
    """Returns [{"branch": "CSE", "rank_type": "Min"/"Max", category: {"M":.., "F":..}}, ...]"""
    with pdfplumber.open(path) as pdf:
        page1_rows = pdf.pages[0].extract_tables()[0]
        page2_rows = pdf.pages[1].extract_tables()[0] if len(pdf.pages) > 1 else []

    # page1: S.NO | BRANCH | RANK | OC-M | OC-F | EWS-M | EWS-F | BCA-M | BCA-F |
    #        BCB-M | BCB-F | BCC-M | BCC-F | BCD-M | BCD-F | BCE-M  (16 cols)
    data_rows = [r for r in page1_rows if _cell(r, 2) in ("Min", "Max")]
    # page2: just the BC-E "F" column, same row order, header "F" in row 0
    bce_f_values = [_cell(r, 0) for r in page2_rows[1:]] if page2_rows else []

    records = []
    branch = None
    for idx, row in enumerate(data_rows):
        branch = _cell(row, 1) or branch  # branch name only present on the Min row
        rank_type = _cell(row, 2)
        cats = {}
        for ci, cat in enumerate(CATEGORIES):
            m = _cell(row, 3 + ci * 2)
            f = _cell(row, 4 + ci * 2) if cat != "BC-E" else (
                bce_f_values[idx] if idx < len(bce_f_values) else None
            )
            if cat == "BC-E":
                f = bce_f_values[idx] if idx < len(bce_f_values) else None
            if m or f:
                cats[cat] = {"M": m, "F": f}
        records.append({"branch": branch, "rank_type": rank_type, "categories": cats})
    return records


def frame_branch_text(branch, min_row, max_row):
    # Mirror the source PDF's own "Min"/"Max" rank columns exactly rather than
    # interpreting them as opening/closing rank - in a few rows Max is lower
    # than Min (e.g. CIC OC-Male, AIML EWS), so that assumption doesn't hold
    # and would misstate the source data.
    lines = [
        f"Q: What was the EAPCET 2025 rank position for the {branch} branch at SASI "
        f"college, category and gender wise?",
        f"A: Based on last year's (EAPCET 2025) admission counselling, the {branch} branch "
        f"at SASI recorded the following Min and Max ranks of admitted candidates by "
        f"category and gender (as published in the official rank position sheet). A lower "
        f"rank number means better merit.",
    ]
    all_cats = sorted(set(min_row.get("categories", {})) | set(max_row.get("categories", {})))
    any_data = False
    for cat in all_cats:
        for gender, label in (("M", "Male"), ("F", "Female")):
            lo = (min_row.get("categories", {}).get(cat) or {}).get(gender)
            hi = (max_row.get("categories", {}).get(cat) or {}).get(gender)
            if not lo and not hi:
                continue
            any_data = True
            if lo and hi:
                lines.append(f"- {cat} {label}: Min rank {lo}, Max rank {hi}")
            elif lo:
                lines.append(f"- {cat} {label}: Min rank {lo}")
            else:
                lines.append(f"- {cat} {label}: Max rank {hi}")
    if not any_data:
        return None
    lines.append(
        "Note: these are last year's (2025) figures, provided for reference only - actual "
        "ranks for the current admission year depend on this year's competition and seat "
        "availability, and can go up or down. Encourage the caller to apply and check with "
        "the admissions department for the latest expected cutoffs."
    )
    return "\n".join(lines)


def embed_batch(texts):
    body = json.dumps({"model": EMBEDDING_MODEL, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=body,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return [item["embedding"] for item in result["data"]]


def main():
    if len(sys.argv) != 2:
        print("usage: ingest_admissions_pdf.py <path-to-pdf>", file=sys.stderr)
        sys.exit(1)
    if not (OPENAI_API_KEY and QDRANT_URL and QDRANT_API_KEY):
        print("OPENAI_API_KEY, QDRANT_URL, and QDRANT_API_KEY must all be set", file=sys.stderr)
        sys.exit(1)

    records = parse_pdf(sys.argv[1])
    by_branch = {}
    for rec in records:
        by_branch.setdefault(rec["branch"], {})[rec["rank_type"]] = rec

    chunks = []
    for branch, rows in by_branch.items():
        text = frame_branch_text(branch, rows.get("Min", {}), rows.get("Max", {}))
        if text:
            chunks.append({"url": "internal://eapcet-2025-rank-cutoffs", "title": f"{SOURCE_LABEL} - {branch}", "text": text})

    print(f"framed {len(chunks)} branch chunks: {[c['title'] for c in chunks]}", file=sys.stderr)

    vectors = embed_batch([c["text"] for c in chunks])
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=vec, payload={"url": c["url"], "title": c["title"], "text": c["text"]})
        for c, vec in zip(chunks, vectors)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"upserted {len(points)} admissions chunks - collection '{COLLECTION_NAME}' now has {count} points", file=sys.stderr)


if __name__ == "__main__":
    main()
