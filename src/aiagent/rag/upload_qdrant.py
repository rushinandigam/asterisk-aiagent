#!/usr/bin/env python3
"""
Uploads data/index.json (built by build_index.py) into a Qdrant collection.
Run offline, once after build_index.py and again whenever it's rebuilt.
The bridge only ever queries Qdrant at call time - this script is the only
thing that writes to it.
"""
import json
import os
import sys
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

INPUT_PATH = "data/index.json"
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "sasi_college")
EMBEDDING_DIM = 1536
UPSERT_BATCH_SIZE = 128

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")


def main():
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("QDRANT_URL and QDRANT_API_KEY must be set", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_PATH) as f:
        chunks = json.load(f)
    print(f"loaded {len(chunks)} chunks from {INPUT_PATH}", file=sys.stderr)

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    for i in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch = chunks[i:i + UPSERT_BATCH_SIZE]
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk["embedding"],
                payload={"url": chunk["url"], "title": chunk["title"], "text": chunk["text"]},
            )
            for chunk in batch
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"uploaded {min(i + UPSERT_BATCH_SIZE, len(chunks))}/{len(chunks)}", file=sys.stderr)

    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"done - collection '{COLLECTION_NAME}' has {count} points", file=sys.stderr)


if __name__ == "__main__":
    main()
