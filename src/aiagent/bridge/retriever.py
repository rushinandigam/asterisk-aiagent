"""
Vector search for SASI college content, backed by Qdrant Cloud. The
collection is populated offline by src/aiagent/rag/upload_qdrant.py; this
module only ever reads from it - no scraping or writing happens at call time.
"""
import json
import os
import urllib.request

from qdrant_client import QdrantClient

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "sasi_college")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

_client = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_API_KEY"),
)


def _embed_query(text):
    body = json.dumps({"model": EMBEDDING_MODEL, "input": [text]}).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=body,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["data"][0]["embedding"]


def search(query, top_k=4, min_score=0.25):
    query_vector = _embed_query(query)
    result = _client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        score_threshold=min_score,
    )
    return [
        {"url": hit.payload["url"], "title": hit.payload["title"], "text": hit.payload["text"], "score": round(hit.score, 3)}
        for hit in result.points
    ]
