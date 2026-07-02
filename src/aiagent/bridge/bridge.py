#!/usr/bin/env python3
"""
aiagent-bridge

TCP server speaking the Asterisk AudioSocket protocol on one side, and a
3-stage AI voice pipeline on the other: Deepgram (speech-to-text) -> OpenAI
chat completions (reasoning + tool-calling) -> ElevenLabs (text-to-speech).

Asterisk's AudioSocket app() sends/receives raw signed-linear 16-bit PCM,
mono, at the channel's negotiated rate. extensions_local.conf answers the
call with the ulaw codec, so that's 8kHz here - this matches Deepgram's
linear16/8000 input format and ElevenLabs' pcm_8000 output format directly,
so no resampling is needed anywhere in this pipeline.

AudioSocket wire format (one TCP connection per call):
    1 byte kind | 2 bytes length (big-endian) | <length> bytes payload
Kinds used here: 0x01 UUID (sent once by Asterisk), 0x10 audio (slin),
0x00 hangup/terminate.
"""
import asyncio
import http.client
import json
import logging
import os
import queue
import re
import struct
import threading
import time

import websockets

import retriever

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("aiagent-bridge")

KIND_HANGUP = 0x00
KIND_UUID = 0x01
KIND_AUDIO = 0x10

LISTEN_HOST = os.environ.get("AIAGENT_LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("AIAGENT_LISTEN_PORT", "9092"))

SALES_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "sales_script.md")
try:
    with open(SALES_SCRIPT_PATH, "r", encoding="utf-8") as _f:
        SALES_SCRIPT = _f.read()
except FileNotFoundError:
    SALES_SCRIPT = ""

ASTERISK_SAMPLE_RATE = 8000
SAMPLE_WIDTH = 2  # 16-bit PCM
OUTBOUND_FRAME_MS = 20
OUTBOUND_FRAME_BYTES = ASTERISK_SAMPLE_RATE * SAMPLE_WIDTH * OUTBOUND_FRAME_MS // 1000
SILENCE_FRAME = b"\x00" * OUTBOUND_FRAME_BYTES

# app_audiosocket.so hangs up after a hardcoded 2000ms of no activity on the
# socket. The Deepgram -> OpenAI -> ElevenLabs round trip can exceed that
# during normal "thinking" pauses, so a keepalive writer fills gaps with
# silence, same as the previous single-hop Realtime bridge did.
KEEPALIVE_GAP_SECONDS = 0.5
KEEPALIVE_CHECK_SECONDS = 0.2

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
DEEPGRAM_MODEL = os.environ.get("DEEPGRAM_MODEL", "nova-3")
# "te" (fixed Telugu) transcribes Telugu-English code-mixed speech in native
# Telugu script far better than "multi" (Deepgram's official code-switching
# mode) - verified during development: "multi" romanized Telugu into
# near-unusable garbled text ("seshikarlá saleloh...") while "te" kept it in
# correct Telugu script and still picked up English technical terms
# reasonably. See aiagent.md for the comparison.
DEEPGRAM_LANGUAGE = os.environ.get("DEEPGRAM_LANGUAGE", "te")
# ms of trailing silence before Deepgram marks an utterance speech_final -
# this is our turn-taking boundary, equivalent to the old server_vad's
# silence_duration_ms. Starting value matches that just-tuned setting; will
# likely need retuning against real calls the same way that was.
DEEPGRAM_ENDPOINTING_MS = int(os.environ.get("DEEPGRAM_ENDPOINTING_MS", "800"))
# Grace period after the agent starts speaking during which barge-in
# detection is suppressed. On a phone/softphone without a headset (no
# acoustic echo cancellation), the agent's own TTS audio leaks back into the
# caller's mic and Deepgram transcribes it, which the bridge would otherwise
# mistake for the caller interrupting - self-echo risk is highest right at
# TTS onset, so this ignores that initial window. Same class of issue (and
# same fix shape) as the old OpenAI Realtime bridge's server_vad tuning.
BARGE_IN_GRACE_SECONDS = float(os.environ.get("BARGE_IN_GRACE_SECONDS", "1.2"))
DEEPGRAM_URL = (
    f"wss://api.deepgram.com/v1/listen?language={DEEPGRAM_LANGUAGE}"
    f"&model={DEEPGRAM_MODEL}&sample_rate={ASTERISK_SAMPLE_RATE}"
    f"&encoding=linear16&channels=1&endpointing={DEEPGRAM_ENDPOINTING_MS}"
    f"&interim_results=true"
)

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID")
# eleven_v3 is the only ElevenLabs model with confirmed-good Telugu output -
# the low-latency conversational models (flash_v2_5, multilingual_v2) don't
# officially support Telugu and were confirmed during development to
# mispronounce it. v3 also isn't available over ElevenLabs' streaming-input
# websocket (confirmed: connection rejected with HTTP 403), so this bridge
# calls the plain REST TTS endpoint instead, once per agent turn.
ELEVENLABS_TTS_MODEL = os.environ.get("ELEVENLABS_TTS_MODEL", "eleven_v3")
# Hybrid model selection: eleven_v3's slowness is only actually needed for
# replies that contain Telugu script - a reply that comes out purely in
# English (e.g. mirroring a caller who spoke English) has no reason to pay
# that latency tax, since the fast model handles English perfectly well.
# Checked per-turn against the actual reply text, not guessed in advance.
ELEVENLABS_TTS_MODEL_FAST = os.environ.get("ELEVENLABS_TTS_MODEL_FAST", "eleven_flash_v2_5")
TELUGU_SCRIPT_RE = re.compile(r"[ఀ-౿]")
ELEVENLABS_HOST = "api.elevenlabs.io"
ELEVENLABS_TTS_PATH_TMPL = "/v1/text-to-speech/{voice_id}?output_format=pcm_8000"

AIAGENT_INSTRUCTIONS = (os.environ.get("AIAGENT_INSTRUCTIONS") or (
    "You are a warm, friendly staff member at SASI (the college at sasi.ac.in) talking to a "
    "prospective student or parent on the phone - not a passive helpdesk, and NOT a brochure or "
    "advertisement read aloud. Talk like a real person having a genuine conversation, the way a "
    "helpful senior student or counselor would - never lead with or repeat formal taglines like "
    "'NAAC A+ accredited' or 'top private college' as an opener; save specific credentials for when "
    "they're actually relevant to what the caller asked or is worried about. Sound natural and "
    "varied, not like reciting the same marketing line every time. This call may be inbound (someone "
    "dialed in) or outbound (you are calling a prospective student/parent); either way, YOU drive "
    "the conversation - never open with or fall back to a generic question like 'how can I help you' "
    "or sit back waiting to be asked something - instead, proactively and warmly draw them in, one "
    "genuine point at a time. Directly address whatever the caller actually said or asked before "
    "adding anything else - don't just launch into generic pitch content that ignores their specific "
    "question or concern. "
    "CRITICAL: every reply must be very short - one, or at most two, short sentences. This is a "
    "live phone call with real-time voice synthesis, not a written essay - a long reply creates a "
    "long, awkward silence before the caller hears anything. Make ONE engaging point (a course, a "
    "placement stat, a facility) per turn, then stop and let the conversation continue naturally - "
    "don't stack multiple points or paragraphs into a single reply. "
    "For ANY factual claim about SASI - courses, departments, admissions, fees, facilities, "
    "placements, faculty, vision/mission, contact details, history, anything about the college - "
    "you MUST call the search_college_info tool first and base what you say strictly on what it "
    "returns. Never invent or guess facts about SASI. If the tool returns nothing relevant, say "
    "briefly that you don't have that detail and offer to connect them with the admissions "
    "department. If the caller asks about anything NOT related to SASI college (general knowledge, "
    "other organizations, personal topics, etc.), briefly explain you can only talk about SASI "
    "College and steer back to it."
)) + (
    " Speak primarily in Telugu, but naturally code-mix in English words and phrases the way "
    "people commonly do in colloquial Telugu conversation in Andhra Pradesh (Tenglish) - "
    "especially for technical, academic, or institutional terms (course names, 'placements', "
    "'faculty', 'admissions', 'campus', numbers, fees) - rather than translating everything "
    "into pure Telugu or switching languages entirely. Keep sentence structure and connecting "
    "words in Telugu, blending in English terms naturally within them. If the caller speaks "
    "primarily in English or Hindi, mirror their language choice for that stretch of the "
    "conversation, but default back to Telugu-English code-mixed speech otherwise. The "
    "college's name is spelled and pronounced \"Shasi\" - always use that spelling."
) + (
    (
        "\n\nBelow is a reference script pack of common caller intents, persona guidance, "
        "and follow-up hooks. Use it as your PRIMARY source for how to handle each topic and "
        "objection - only fall back to search_college_info for specific live facts/figures "
        "(exact fees, current placement numbers, cutoffs) or topics this pack doesn't cover. "
        "The pack's example answers are written as long paragraphs for a human reading them - "
        "you must NOT recite one verbatim in a single turn, since every reply must still stay "
        "to one or two short sentences. Instead, treat each entry as multiple conversation "
        "turns: make one point, then use its follow-up hook (asked as a short, natural "
        "question) to move the conversation forward, saving the rest of that entry's content "
        "for later turns as the caller responds. This pack has no WhatsApp/messaging "
        "capability on this phone call - wherever it says to send something on WhatsApp, "
        "instead offer to have a counselor call them back, or use search_college_info to "
        "answer it directly on the call if possible.\n\n"
        f"{SALES_SCRIPT}"
    ) if SALES_SCRIPT else ""
)
AIAGENT_GREETING = os.environ.get("AIAGENT_GREETING") or "శశి కళాశాలకు స్వాగతం!"

SEARCH_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "search_college_info",
            "description": (
                "Search SASI college's official website content (courses, departments, "
                "admissions, fees, facilities, placements, faculty, vision/mission, contact "
                "info, etc). Always call this before answering any factual question about SASI."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for, e.g. 'B.Tech CSE fee structure'",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

MAX_TOOL_ROUNDS = 3


async def read_audiosocket_packet(reader: asyncio.StreamReader):
    header = await reader.readexactly(3)
    kind, length = header[0], struct.unpack(">H", header[1:3])[0]
    payload = await reader.readexactly(length) if length else b""
    return kind, payload


def write_audiosocket_packet(writer: asyncio.StreamWriter, kind: int, payload: bytes = b""):
    writer.write(bytes([kind]) + struct.pack(">H", len(payload)) + payload)


# Pooled keep-alive HTTPS connections, one pool per host. A single AI turn
# can make 2-4 API calls in sequence (LLM, maybe a second LLM call after a
# tool result, then TTS) - urllib.request opens a brand new TCP+TLS
# connection for every single one, paying a full handshake each time. This
# reuses connections across calls instead, one small pool per host so
# multiple simultaneous phone calls don't serialize behind each other.
_HTTP_POOLS = {}
_HTTP_POOLS_LOCK = threading.Lock()
HTTP_POOL_MAX_SIZE = 4


def _get_pool(host):
    with _HTTP_POOLS_LOCK:
        pool = _HTTP_POOLS.setdefault(host, queue.Queue(maxsize=HTTP_POOL_MAX_SIZE))
    return pool


def _http_post(host, path, headers, body, timeout):
    """POST body (bytes) to host+path over a pooled keep-alive connection,
    with one retry on a fresh connection if the pooled one turned out to be
    stale (closed server-side after sitting idle)."""
    pool = _get_pool(host)
    try:
        conn = pool.get_nowait()
    except queue.Empty:
        conn = http.client.HTTPSConnection(host, timeout=timeout)

    for attempt in (1, 2):
        try:
            conn.request("POST", path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            status = resp.status
            break
        except (http.client.HTTPException, ConnectionError, OSError):
            conn.close()
            if attempt == 2:
                raise
            conn = http.client.HTTPSConnection(host, timeout=timeout)

    if status >= 400:
        conn.close()
        raise RuntimeError(f"{host}{path} returned HTTP {status}: {data[:500]!r}")

    try:
        pool.put_nowait(conn)
    except queue.Full:
        conn.close()
    return data


def _http_post_json(host, path, headers, payload, timeout=30):
    body = json.dumps(payload).encode("utf-8")
    data = _http_post(host, path, {**headers, "Content-Type": "application/json"}, body, timeout)
    return json.loads(data.decode("utf-8"))


def _openai_chat_sync(messages):
    return _http_post_json(
        "api.openai.com",
        "/v1/chat/completions",
        {"Authorization": f"Bearer {OPENAI_API_KEY}"},
        {
            "model": OPENAI_CHAT_MODEL,
            "messages": messages,
            "tools": SEARCH_TOOL,
            "tool_choice": "auto",
            # Hard cap on reply length, not just a prompt suggestion - eleven_v3
            # (the only ElevenLabs model with confirmed-good Telugu) is slow
            # enough that a multi-paragraph reply takes noticeably long to
            # synthesize, even though the keepalive prevents an AudioSocket
            # timeout. Measured on real calls: TTS generation time scales
            # with reply length, so a tighter cap directly cuts end-to-end
            # latency, on top of being better phone etiquette than an
            # essay-length answer.
            "max_tokens": 70,
        },
        timeout=30,
    )


async def run_llm_turn(call_id, history):
    """Runs the OpenAI chat-completions + tool-calling loop, mutating
    `history` in place (assistant/tool messages appended as they happen),
    and returns the final assistant text for this turn."""
    for _ in range(MAX_TOOL_ROUNDS):
        resp = await asyncio.to_thread(_openai_chat_sync, history)
        msg = resp["choices"][0]["message"]
        if msg.get("tool_calls"):
            history.append(msg)
            for tc in msg["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                query = args.get("query", "")
                log.info("[%s] tool call search_college_info(%r)", call_id, query)
                results = await asyncio.to_thread(retriever.search, query)
                log.info("[%s] retrieved %d matching chunk(s)", call_id, len(results))
                output = {"results": results} if results else {
                    "results": [],
                    "note": "No matching information found on the SASI website for this query.",
                }
                history.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(output),
                })
            continue
        text = (msg.get("content") or "").strip()
        history.append({"role": "assistant", "content": text})
        return text
    log.warning("[%s] hit MAX_TOOL_ROUNDS without a final answer", call_id)
    return ""


def _elevenlabs_tts_sync(text):
    # Only pay eleven_v3's latency tax when the reply actually contains
    # Telugu script - a reply that comes out purely in English/Latin script
    # (e.g. mirroring a caller who spoke English) synthesizes correctly on
    # the much faster model, so there's no reason to use the slow one.
    model = ELEVENLABS_TTS_MODEL if TELUGU_SCRIPT_RE.search(text) else ELEVENLABS_TTS_MODEL_FAST
    log.info("TTS model selected: %s", model)
    path = ELEVENLABS_TTS_PATH_TMPL.format(voice_id=ELEVENLABS_VOICE_ID)
    body = json.dumps({"text": text, "model_id": model}).encode("utf-8")
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    # eleven_v3 is meaningfully slower than the low-latency ElevenLabs
    # models (that's the whole reason it's used - it's the only one with
    # confirmed-good Telugu output), so this needs more headroom than a
    # typical REST call; the keepalive writer covers Asterisk's socket
    # during this wait regardless of how long it takes.
    return _http_post(ELEVENLABS_HOST, path, headers, body, timeout=60)


async def elevenlabs_tts(text):
    """Blocking REST call to ElevenLabs TTS (eleven_v3 doesn't support the
    streaming-input websocket), returns raw pcm_8000 bytes ready to frame
    straight into AudioSocket packets - no resampling needed."""
    return await asyncio.to_thread(_elevenlabs_tts_sync, text)


async def deepgram_stt_session():
    return await websockets.connect(
        DEEPGRAM_URL, additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    )


async def pump_asterisk_to_stt(call_id, reader, stt_ws, hangup_event):
    """Read raw SLIN frames from Asterisk and forward as binary websocket
    frames to Deepgram - continuously, regardless of whether the agent is
    currently speaking, so a barge-in can be detected at any time."""
    packets_in = 0
    bytes_in = 0
    try:
        while True:
            kind, payload = await read_audiosocket_packet(reader)
            if kind == KIND_HANGUP:
                log.info("[%s] caller hung up", call_id)
                break
            if kind != KIND_AUDIO or not payload:
                continue
            packets_in += 1
            bytes_in += len(payload)
            if packets_in % 100 == 0:
                log.info("[%s] received %d audio packets (%d bytes) from Asterisk so far", call_id, packets_in, bytes_in)
            await stt_ws.send(payload)
    except asyncio.IncompleteReadError:
        log.info("[%s] AudioSocket connection closed by Asterisk", call_id)
    finally:
        log.info("[%s] total received from Asterisk: %d packets, %d bytes", call_id, packets_in, bytes_in)
        hangup_event.set()


async def pump_stt_transcripts(call_id, stt_ws, transcript_queue, state, hangup_event):
    """Listen for Deepgram Results events. Any non-empty transcript while
    the agent is speaking is treated as a barge-in; a speech_final result
    (Deepgram's own end-of-utterance/endpointing signal) is a completed
    caller turn, queued for the LLM."""
    try:
        async for message in stt_ws:
            data = json.loads(message)
            if data.get("type") != "Results":
                continue
            alt = data["channel"]["alternatives"][0]
            transcript = alt["transcript"].strip()
            if not transcript:
                continue
            if (
                state["is_agent_speaking"]
                and not state["interrupted"]
                and time.monotonic() - state["speaking_started_at"] >= BARGE_IN_GRACE_SECONDS
            ):
                log.info("[%s] caller interrupted - barge-in detected", call_id)
                state["interrupted"] = True
            if data.get("speech_final"):
                await transcript_queue.put(transcript)
    except websockets.exceptions.ConnectionClosed:
        log.info("[%s] Deepgram STT connection closed", call_id)
    finally:
        hangup_event.set()
        await transcript_queue.put(None)  # unblocks process_turns on hangup


async def stream_pcm_to_asterisk(call_id, audio, writer, state, last_write):
    """Frame raw pcm_8000 bytes into 20ms AudioSocket packets, paced to
    roughly real-time playback so a barge-in (state['interrupted']) can
    actually cut off the remaining audio instead of it all being written to
    the socket instantly."""
    pending = audio
    bytes_out = 0
    while len(pending) >= OUTBOUND_FRAME_BYTES:
        if state["interrupted"]:
            log.info("[%s] TTS playback interrupted by barge-in, stopping early", call_id)
            break
        frame, pending = pending[:OUTBOUND_FRAME_BYTES], pending[OUTBOUND_FRAME_BYTES:]
        write_audiosocket_packet(writer, KIND_AUDIO, frame)
        bytes_out += len(frame)
        await writer.drain()
        last_write[0] = time.monotonic()
        await asyncio.sleep(OUTBOUND_FRAME_MS / 1000)
    if pending and not state["interrupted"]:
        frame = pending.ljust(OUTBOUND_FRAME_BYTES, b"\x00")
        write_audiosocket_packet(writer, KIND_AUDIO, frame)
        bytes_out += len(frame)
        await writer.drain()
        last_write[0] = time.monotonic()
    log.info("[%s] wrote %d bytes of agent audio to Asterisk", call_id, bytes_out)


async def speak_turn(call_id, history, writer, state, last_write):
    """Run one LLM turn to completion and speak the result, honoring
    barge-in throughout."""
    state["interrupted"] = False
    reply = await run_llm_turn(call_id, history)
    log.info("[%s] agent said: %s", call_id, reply)
    if not reply:
        return
    # is_agent_speaking suppresses the keepalive's silence filler, so it
    # must stay False during the (potentially slow, especially on
    # eleven_v3) LLM+TTS generation wait - otherwise nothing fills the
    # AudioSocket while we wait, and Asterisk's own hardcoded 2s inactivity
    # timeout kills the call before any audio is ever sent. Only flip it
    # once real audio is in hand and we're about to write it.
    try:
        audio = await elevenlabs_tts(reply)
    except Exception:
        log.exception("[%s] TTS generation error", call_id)
        return
    state["is_agent_speaking"] = True
    state["speaking_started_at"] = time.monotonic()
    try:
        await stream_pcm_to_asterisk(call_id, audio, writer, state, last_write)
    except (ConnectionError, OSError):
        # Caller/Asterisk hung up mid-playback - not an error worth a
        # traceback, just the call ending while audio was still queued.
        log.info("[%s] connection closed mid-playback", call_id)
    except Exception:
        log.exception("[%s] playback error", call_id)
    finally:
        state["is_agent_speaking"] = False


async def process_turns(call_id, transcript_queue, writer, hangup_event, state, last_write, history):
    """Speaks the pinned greeting first, then processes each completed
    caller turn from the queue sequentially - one call is a single line of
    conversation, no need to handle overlapping LLM turns."""
    history.append({"role": "user", "content": (
        f"Say, word for word: \"{AIAGENT_GREETING}\" - then, without pausing for a reply, "
        "immediately add ONE brief, engaging point about SASI college in a single short "
        "sentence. Keep the whole reply short overall - this is spoken audio, not text."
    )})
    await speak_turn(call_id, history, writer, state, last_write)

    while not hangup_event.is_set():
        try:
            text = await asyncio.wait_for(transcript_queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        if text is None:
            break
        log.info("[%s] 1002 said: %s", call_id, text)
        history.append({"role": "user", "content": text})
        await speak_turn(call_id, history, writer, state, last_write)


async def keepalive_writer(writer, hangup_event, last_write, state):
    """Fill idle gaps with silence so app_audiosocket's hardcoded 2s inactivity
    timeout doesn't fire while waiting on the Deepgram/OpenAI/ElevenLabs
    round trip. Suppressed while agent audio is actively streaming."""
    while not hangup_event.is_set():
        await asyncio.sleep(KEEPALIVE_CHECK_SECONDS)
        if hangup_event.is_set():
            break
        if state["is_agent_speaking"]:
            continue
        if time.monotonic() - last_write[0] >= KEEPALIVE_GAP_SECONDS:
            write_audiosocket_packet(writer, KIND_AUDIO, SILENCE_FRAME)
            try:
                await writer.drain()
            except (ConnectionError, OSError):
                break
            last_write[0] = time.monotonic()


async def handle_call(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    peer = writer.get_extra_info("peername")
    call_id = str(peer)

    kind, payload = await read_audiosocket_packet(reader)
    if kind == KIND_UUID:
        call_id = payload.hex()
        log.info("[%s] new call from Asterisk (%s)", call_id, peer)
    else:
        log.warning("[%s] expected UUID packet first, got kind=0x%02x", call_id, kind)

    missing = [name for name, val in (
        ("OPENAI_API_KEY", OPENAI_API_KEY),
        ("DEEPGRAM_API_KEY", DEEPGRAM_API_KEY),
        ("ELEVENLABS_API_KEY", ELEVENLABS_API_KEY),
        ("ELEVENLABS_VOICE_ID", ELEVENLABS_VOICE_ID),
    ) if not val]
    if missing:
        log.error("[%s] missing required env var(s) %s, hanging up", call_id, missing)
        write_audiosocket_packet(writer, KIND_HANGUP)
        await writer.drain()
        writer.close()
        return

    # Start filling the socket with silence immediately - before the STT
    # websocket handshake - so app_audiosocket's hardcoded 2s inactivity
    # timeout never gets a chance to fire during connection setup.
    hangup_event = asyncio.Event()
    last_write = [time.monotonic()]
    state = {"is_agent_speaking": False, "interrupted": False, "speaking_started_at": 0.0}
    keepalive_task = asyncio.create_task(keepalive_writer(writer, hangup_event, last_write, state))

    stt_ws = None
    try:
        stt_ws = await deepgram_stt_session()
        log.info("[%s] Deepgram STT session ready (model=%s, language=%s)", call_id, DEEPGRAM_MODEL, DEEPGRAM_LANGUAGE)
        history = [{"role": "system", "content": AIAGENT_INSTRUCTIONS}]
        transcript_queue = asyncio.Queue()
        await asyncio.gather(
            pump_asterisk_to_stt(call_id, reader, stt_ws, hangup_event),
            pump_stt_transcripts(call_id, stt_ws, transcript_queue, state, hangup_event),
            process_turns(call_id, transcript_queue, writer, hangup_event, state, last_write, history),
        )
    finally:
        hangup_event.set()
        keepalive_task.cancel()
        try:
            await keepalive_task
        except asyncio.CancelledError:
            pass
        if stt_ws is not None:
            try:
                await stt_ws.send(json.dumps({"type": "CloseStream"}))
            except (websockets.exceptions.ConnectionClosed, ConnectionError, OSError):
                pass
            await stt_ws.close()
        try:
            write_audiosocket_packet(writer, KIND_HANGUP)
            await writer.drain()
        except (ConnectionError, OSError):
            pass
        writer.close()
        log.info("[%s] call ended", call_id)


async def main():
    server = await asyncio.start_server(handle_call, LISTEN_HOST, LISTEN_PORT)
    log.info("aiagent-bridge listening on %s:%d", LISTEN_HOST, LISTEN_PORT)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
