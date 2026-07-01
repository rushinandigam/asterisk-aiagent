# aiagent

Turns extension 1001 into an AI voice agent that answers questions about
SASI college (sasi.ac.in) using a locally-stored knowledge base. Calls
dialed to 1001 are answered and bridged, via Asterisk's AudioSocket app, to
the `aiagent-bridge` service, which relays the audio to OpenAI's Realtime
API (speech-to-speech) and streams the spoken reply back into the call.

## How it works

- `src/privatedial/config/extensions_local.conf` adds an exact-match
  `exten => 1001` in `[dp_entry_call_inout]` that answers the call and runs
  `AudioSocket(${UUID()},127.0.0.1:9092)` instead of falling through to
  `sub_dial_term`/`sub_voicemail`. Every other extension dials as before.
  (`${UUID()}` is required - `AudioSocket()` rejects `${UNIQUEID}`, which
  isn't a real UUID.)
- `src/aiagent/bridge/bridge.py` is a small asyncio TCP server speaking the
  AudioSocket protocol on one side and a WebSocket client to
  `wss://api.openai.com/v1/realtime` on the other. It resamples audio
  8kHz (Asterisk, ulaw) <-> 24kHz (OpenAI Realtime PCM16) with stdlib
  `audioop`, greets the caller immediately on connect, handles barge-in by
  cancelling the in-flight response when the caller starts talking over the
  agent, and fills idle gaps with silence so Asterisk's hardcoded 2s
  AudioSocket inactivity timeout doesn't fire while waiting on OpenAI.
- `src/aiagent/bridge/retriever.py` queries a Qdrant Cloud collection
  (`sasi_college` by default) for similarity search; the Realtime session
  is configured with a `search_college_info` function tool, the model calls
  it for any factual question, and the bridge runs the search and feeds the
  results back via `conversation.item.create` (`function_call_output`).
  Only the live query gets embedded at call time (one OpenAI embeddings
  call); the corpus itself lives entirely in Qdrant.
- `src/aiagent/rag/scrape.py`, `build_index.py`, and `upload_qdrant.py` are
  the **offline** pipeline that builds and uploads that data from
  sasi.ac.in - the bridge never scrapes the website or writes to Qdrant at
  call time, only reads. Re-run them (see below) whenever the website
  content changes.
- In `demo/docker-compose.yml`, `aiagent-bridge` runs with
  `network_mode: "service:tele"` so it shares the Asterisk container's
  network namespace and is reachable at `127.0.0.1:9092` — `tele` itself
  uses the legacy `bridge` network mode, which has no built-in service-name
  DNS, so a normal compose network alias would not have resolved.
- `src/privatedial/config/pjsip_transport.conf`'s `external_signaling_address`
  / `external_media_address` must be set to an address your softphone can
  actually reach (not `127.0.0.1`/`sip.example.com` placeholders) - otherwise
  Asterisk tells the caller's phone to send RTP audio to an unreachable
  address and you get one-way audio (agent audible, caller's voice never
  arrives). See the NAT section in `src/privatedial/doc/privatedial.md`.

## Refreshing the SASI knowledge base

```
cd src/aiagent/rag
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scrape.py                                          # writes data/pages.jsonl
OPENAI_API_KEY=sk-... .venv/bin/python build_index.py                # writes data/index.json
QDRANT_URL=... QDRANT_API_KEY=... OPENAI_API_KEY=sk-... \
  .venv/bin/python upload_qdrant.py                                 # uploads to Qdrant
```

`scrape.py` pulls every URL in `https://sasi.ac.in/page-sitemap.xml` (minus
admin/feedback-form boilerplate pages), strips HTML to plain text.
`build_index.py` chunks each page (~1200 chars, 150 char overlap) and embeds
every chunk with `text-embedding-3-small`. `upload_qdrant.py` recreates the
`sasi_college` collection (1536-dim, cosine distance) and upserts every
chunk with its embedding as the vector and `{url, title, text}` as payload.
No rebuild/restart of `aiagent-bridge` is needed after this - it queries
Qdrant live, so a re-upload takes effect on the next call.

## Required setup

1. An OpenAI account with Realtime API access, and an API key:
   `OPENAI_API_KEY=sk-...` (same key is reused for embeddings).
2. A Qdrant Cloud cluster (or self-hosted instance) URL and API key.
3. Add to `demo/.env` (or export before `docker compose up`):
   ```
   OPENAI_API_KEY=sk-...
   QDRANT_URL=https://xxxxx.gcp.cloud.qdrant.io
   QDRANT_API_KEY=...
   # optional overrides:
   QDRANT_COLLECTION=sasi_college
   OPENAI_REALTIME_MODEL=gpt-realtime
   OPENAI_REALTIME_VOICE=alloy
   OPENAI_REALTIME_SPEED=0.85
   AIAGENT_INSTRUCTIONS=...
   AIAGENT_GREETING=Welcome to SASI. How may I help you today?
   ```
3. Build and start: `docker compose up --build` from `demo/`.
4. Register a softphone as extension 1002 (see `pjsip_endpoint.conf`,
   `inbound_auth/username = 1002`, password `1234`) and dial `1001`.

## Verifying

- `docker compose logs -f aiagent-bridge` shows each call's lifecycle:
  UUID received, OpenAI Realtime session created, `1002 said: ...` /
  `agent said: ...` transcripts, tool calls (`tool call
  search_college_info(...)`, `retrieved N matching chunk(s)`), call ended.
- If the call connects but you hear silence, double-check
  `OPENAI_API_KEY` is set in the `aiagent-bridge` container
  (`docker compose exec aiagent-bridge env | grep OPENAI`) — the bridge
  logs an error and hangs up immediately if it's missing.
- If you hear the agent but it never hears you (no `1002 said:` lines, ever),
  check `external_media_address` in `pjsip_transport.conf` first - this is
  the most common cause and is unrelated to the bridge.
- Treat `QDRANT_API_KEY`/`OPENAI_API_KEY` as live secrets: keep them only in
  `demo/.env` (untracked by git), never in code or commit messages, and
  rotate immediately if either is ever pasted somewhere it could leak.

## Known rough edges to verify against a live call

- OpenAI's Realtime API event names have shifted between beta and GA
  (`response.audio.delta` -> `response.output_audio.delta`, etc.) - the
  code handles both spellings for the events found in testing, but the API
  may continue to evolve. The bridge logs every event type it doesn't
  explicitly handle, which is the fastest way to spot a renamed field.
- `search_college_info` results are filtered to cosine similarity >= 0.25
  (see `retriever.search`'s `min_score`); tune if the agent says "I don't
  have that information" too often or too rarely.
