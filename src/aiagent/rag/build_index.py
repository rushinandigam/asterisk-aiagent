#!/usr/bin/env python3
"""
Builds data/index.json - the local vector index - from data/pages.jsonl.
Run offline, whenever pages.jsonl changes. The bridge only ever reads the
resulting index.json at startup; it never calls this script itself.
"""
import json
import os
import sys
import urllib.request

INPUT_PATH = "data/pages.jsonl"
OUTPUT_PATH = "data/index.json"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150
EMBED_BATCH_SIZE = 64

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def chunk_text(text, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP_CHARS):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if len(c.strip()) > 50]


def embed_batch(texts):
    body = json.dumps({"model": EMBEDDING_MODEL, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=body,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return [item["embedding"] for item in result["data"]]


def main():
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    chunks = []
    with open(INPUT_PATH) as f:
        for line in f:
            page = json.loads(line)
            for chunk in chunk_text(page["text"]):
                chunks.append({"url": page["url"], "title": page["title"], "text": chunk})

    print(f"{len(chunks)} chunks from pages.jsonl, embedding in batches of {EMBED_BATCH_SIZE}...", file=sys.stderr)
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i:i + EMBED_BATCH_SIZE]
        vectors = embed_batch([c["text"] for c in batch])
        for chunk, vector in zip(batch, vectors):
            chunk["embedding"] = vector
        print(f"embedded {min(i + EMBED_BATCH_SIZE, len(chunks))}/{len(chunks)}", file=sys.stderr)

    with open(OUTPUT_PATH, "w") as out:
        json.dump(chunks, out)
    print(f"wrote {OUTPUT_PATH} ({len(chunks)} chunks)", file=sys.stderr)


if __name__ == "__main__":
    main()
