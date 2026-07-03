#!/usr/bin/env python3
"""
aiagent-bridge

TCP server speaking the Asterisk AudioSocket protocol on one side, and a
3-stage AI voice pipeline on the other: Deepgram (speech-to-text) -> an Agno
Agent (reasoning + tool-calling, backed by OpenAI) -> ElevenLabs
(text-to-speech).

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
from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.openai import OpenAIChat

import retriever

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s.%(msecs)03d %(levelname)s:%(name)s:%(message)s",
    datefmt="%H:%M:%S",
)
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
# socket. The Deepgram -> Agno -> ElevenLabs round trip can exceed that
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
# silence_duration_ms.
DEEPGRAM_ENDPOINTING_MS = int(os.environ.get("DEEPGRAM_ENDPOINTING_MS", "800"))
# Grace period after the agent starts speaking during which barge-in
# detection is suppressed. On a phone/softphone without a headset (no
# acoustic echo cancellation), the agent's own TTS audio leaks back into the
# caller's mic and Deepgram transcribes it, which the bridge would otherwise
# mistake for the caller interrupting - self-echo risk is highest right at
# TTS onset, so this ignores that initial window.
BARGE_IN_GRACE_SECONDS = float(os.environ.get("BARGE_IN_GRACE_SECONDS", "1.2"))
DEEPGRAM_URL = (
    f"wss://api.deepgram.com/v1/listen?language={DEEPGRAM_LANGUAGE}"
    f"&model={DEEPGRAM_MODEL}&sample_rate={ASTERISK_SAMPLE_RATE}"
    f"&encoding=linear16&channels=1&endpointing={DEEPGRAM_ENDPOINTING_MS}"
    f"&interim_results=true"
    # speech_final (endpointing-based) has been observed to sometimes never
    # fire on a real utterance - the caller's turn gets transcribed
    # correctly but is silently dropped since nothing closes it out.
    # UtteranceEnd is a separate, word-timing-based silence signal Deepgram
    # sends independently of endpointing, used below as a fallback so a
    # turn still completes even if speech_final never arrives.
    f"&utterance_end_ms=1000"
)

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID")
# eleven_v3 is the only ElevenLabs model with confirmed-good Telugu output -
# the low-latency conversational models (flash_v2_5, multilingual_v2) don't
# officially support Telugu and were confirmed during development to
# mispronounce it. v3 also isn't available over ElevenLabs' streaming-input
# websocket (confirmed: connection rejected with HTTP 403) - but its
# streaming REST endpoint works fine and gets time-to-first-byte under a
# second even for v3 (measured ~0.6s vs ~6s total generation time), so this
# bridge streams the REST response instead of waiting for the whole clip.
ELEVENLABS_TTS_MODEL = os.environ.get("ELEVENLABS_TTS_MODEL", "eleven_v3")
# Hybrid model selection: eleven_v3's slowness is only actually needed for
# replies that contain Telugu script - a reply that comes out purely in
# English (e.g. mirroring a caller who spoke English) has no reason to pay
# that latency tax, since the fast model handles English perfectly well.
# Checked per-turn against the actual reply text, not guessed in advance.
ELEVENLABS_TTS_MODEL_FAST = os.environ.get("ELEVENLABS_TTS_MODEL_FAST", "eleven_flash_v2_5")
TELUGU_SCRIPT_RE = re.compile(r"[ఀ-౿]")
ELEVENLABS_HOST = "api.elevenlabs.io"
ELEVENLABS_TTS_STREAM_PATH_TMPL = "/v1/text-to-speech/{voice_id}/stream?output_format=pcm_8000"

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


def search_college_info(query: str) -> str:
    """Search SASI college's official website content (courses, departments,
    admissions, fees, facilities, placements, faculty, vision/mission,
    contact info, etc). Always call this before answering any factual
    question about SASI.

    Args:
        query: What to search for, e.g. "B.Tech CSE fee structure"
    """
    log.info("tool call search_college_info(%r)", query)
    results = retriever.search(query)
    log.info("retrieved %d matching chunk(s)", len(results))
    if not results:
        return json.dumps({
            "results": [],
            "note": "No matching information found on the SASI website for this query.",
        })
    return json.dumps({"results": results})


# One shared Agent handles every call; conversation continuity per call
# comes from passing session_id=call_id into run() - Agno keeps each
# session's history in the in-memory db, scoped separately per session_id,
# so concurrent calls never see each other's conversation.
aiagent = Agent(
    model=OpenAIChat(id=OPENAI_CHAT_MODEL, api_key=OPENAI_API_KEY, max_tokens=70),
    db=InMemoryDb(),
    tools=[search_college_info],
    instructions=AIAGENT_INSTRUCTIONS,
    add_history_to_context=True,
    num_history_runs=10,
    markdown=False,
)


def _agno_run_sync(call_id, text):
    result = aiagent.run(text, session_id=call_id)
    return (result.content or "").strip()


async def run_llm_turn(call_id, text):
    """Runs one turn through the Agno agent (including any tool calls it
    makes internally) in a background thread, since Agent.run is a
    blocking call - keeps the event loop free for the keepalive writer and
    other concurrent calls."""
    return await asyncio.to_thread(_agno_run_sync, call_id, text)


async def read_audiosocket_packet(reader: asyncio.StreamReader):
    header = await reader.readexactly(3)
    kind, length = header[0], struct.unpack(">H", header[1:3])[0]
    payload = await reader.readexactly(length) if length else b""
    return kind, payload


def write_audiosocket_packet(writer: asyncio.StreamWriter, kind: int, payload: bytes = b""):
    writer.write(bytes([kind]) + struct.pack(">H", len(payload)) + payload)


def _elevenlabs_tts_stream_sync(text, chunk_queue):
    """Runs in a background thread. POSTs to ElevenLabs' streaming REST
    endpoint and pushes raw pcm_8000 chunks onto chunk_queue as they arrive
    over the network, rather than waiting for the full clip. Even
    eleven_v3 - meaningfully slower than the low-latency models to fully
    finish a clip - starts returning bytes in well under a second on this
    endpoint (measured: ~0.6s time-to-first-byte vs ~6s total), so
    streaming gets the caller real audio almost immediately without
    trading away eleven_v3's Telugu quality. Terminates chunk_queue with
    None on success, or an Exception instance on failure."""
    model = ELEVENLABS_TTS_MODEL if TELUGU_SCRIPT_RE.search(text) else ELEVENLABS_TTS_MODEL_FAST
    log.info("TTS model selected: %s (streaming)", model)
    path = ELEVENLABS_TTS_STREAM_PATH_TMPL.format(voice_id=ELEVENLABS_VOICE_ID)
    body = json.dumps({"text": text, "model_id": model}).encode("utf-8")
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    conn = http.client.HTTPSConnection(ELEVENLABS_HOST, timeout=60)
    try:
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        if resp.status >= 400:
            data = resp.read()
            raise RuntimeError(f"{ELEVENLABS_HOST}{path} returned HTTP {resp.status}: {data[:500]!r}")
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            chunk_queue.put(chunk)
    except Exception as exc:
        chunk_queue.put(exc)
        return
    finally:
        conn.close()
    chunk_queue.put(None)


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
    """Listen for Deepgram Results/UtteranceEnd events. Any non-empty
    transcript while the agent is speaking is treated as a barge-in.
    is_final chunks accumulate into pending_text until the turn closes -
    either via speech_final (Deepgram's endpointing signal) or, as a
    fallback, an UtteranceEnd event. speech_final was observed on real
    calls to sometimes never fire on a genuine utterance (correctly
    transcribed speech silently dropped, no reply ever sent) - UtteranceEnd
    is a separate, more reliable silence signal that catches those cases."""
    pending_text = ""
    try:
        async for message in stt_ws:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "UtteranceEnd":
                if pending_text:
                    log.info("[%s] timing: UtteranceEnd at %.3f (fallback end-of-turn, speech_final never fired)",
                              call_id, time.monotonic())
                    await transcript_queue.put(pending_text)
                    pending_text = ""
                continue

            if msg_type != "Results":
                continue
            alt = data["channel"]["alternatives"][0]
            transcript = alt["transcript"].strip()
            if not transcript:
                continue
            log.info(
                "[%s] STT transcript (is_final=%s, speech_final=%s, conf=%.2f): %r",
                call_id, data.get("is_final"), data.get("speech_final"),
                alt.get("confidence", 0.0), transcript,
            )
            if (
                state["is_agent_speaking"]
                and not state["interrupted"]
                and time.monotonic() - state["speaking_started_at"] >= BARGE_IN_GRACE_SECONDS
            ):
                log.info("[%s] caller interrupted - barge-in detected", call_id)
                state["interrupted"] = True
            if data.get("is_final"):
                pending_text = (pending_text + " " + transcript).strip()
            if data.get("speech_final"):
                log.info("[%s] timing: speech_final at %.3f (caller stopped talking, starting reply pipeline)",
                          call_id, time.monotonic())
                if pending_text:
                    await transcript_queue.put(pending_text)
                    pending_text = ""
    except websockets.exceptions.ConnectionClosed:
        log.info("[%s] Deepgram STT connection closed", call_id)
    finally:
        hangup_event.set()
        await transcript_queue.put(None)  # unblocks process_turns on hangup


async def stream_tts_to_asterisk(call_id, text, writer, state, last_write, turn_start):
    """Starts the ElevenLabs streaming fetch in a background thread and
    frames pcm_8000 chunks into 20ms AudioSocket packets as they arrive,
    paced to roughly real-time playback so a barge-in (state['interrupted'])
    can actually cut off the remaining audio instead of it all being
    written to the socket instantly. Audio starts playing on ElevenLabs'
    time-to-first-byte rather than its total generation time - the
    keepalive covers the gap up to the first chunk, is_agent_speaking only
    flips True once real audio is actually in hand."""
    chunk_queue = queue.Queue()
    thread = threading.Thread(
        target=_elevenlabs_tts_stream_sync, args=(text, chunk_queue), daemon=True
    )
    thread.start()

    pending = b""
    bytes_out = 0
    first_chunk_at = None
    while True:
        try:
            item = await asyncio.wait_for(asyncio.to_thread(chunk_queue.get), timeout=10)
        except asyncio.TimeoutError:
            log.warning("[%s] TTS stream stalled - no data for 10s", call_id)
            break
        if item is None:
            break
        if isinstance(item, Exception):
            log.error("[%s] TTS streaming error: %r", call_id, item)
            break
        if first_chunk_at is None:
            first_chunk_at = time.monotonic()
            log.info("[%s] timing: first TTS audio chunk after %.2fs", call_id, first_chunk_at - turn_start)
            state["is_agent_speaking"] = True
            state["speaking_started_at"] = first_chunk_at
        pending += item
        while len(pending) >= OUTBOUND_FRAME_BYTES:
            if state["interrupted"]:
                log.info("[%s] TTS playback interrupted by barge-in, stopping early", call_id)
                pending = b""
                break
            frame, pending = pending[:OUTBOUND_FRAME_BYTES], pending[OUTBOUND_FRAME_BYTES:]
            write_audiosocket_packet(writer, KIND_AUDIO, frame)
            bytes_out += len(frame)
            await writer.drain()
            last_write[0] = time.monotonic()
            await asyncio.sleep(OUTBOUND_FRAME_MS / 1000)
        if state["interrupted"]:
            break
    if pending and not state["interrupted"]:
        frame = pending.ljust(OUTBOUND_FRAME_BYTES, b"\x00")
        write_audiosocket_packet(writer, KIND_AUDIO, frame)
        bytes_out += len(frame)
        await writer.drain()
        last_write[0] = time.monotonic()
    log.info("[%s] wrote %d bytes of agent audio to Asterisk (streamed)", call_id, bytes_out)


async def speak_turn(call_id, text, writer, state, last_write):
    """Run one Agno turn to completion and speak the result, honoring
    barge-in throughout."""
    state["interrupted"] = False
    turn_start = time.monotonic()
    reply = await run_llm_turn(call_id, text)
    llm_done = time.monotonic()
    log.info("[%s] agent said: %s", call_id, reply)
    log.info("[%s] timing: LLM turn took %.2fs", call_id, llm_done - turn_start)
    if not reply:
        return
    # is_agent_speaking suppresses the keepalive's silence filler, so it
    # must stay False until the streaming TTS fetch actually hands over its
    # first chunk - otherwise nothing fills the AudioSocket while we wait,
    # and Asterisk's own hardcoded 2s inactivity timeout kills the call
    # before any audio is ever sent. stream_tts_to_asterisk flips it once
    # real audio is in hand.
    try:
        await stream_tts_to_asterisk(call_id, reply, writer, state, last_write, llm_done)
    except (ConnectionError, OSError):
        # Caller/Asterisk hung up mid-playback - not an error worth a
        # traceback, just the call ending while audio was still queued.
        log.info("[%s] connection closed mid-playback", call_id)
    except Exception:
        log.exception("[%s] playback error", call_id)
    finally:
        state["is_agent_speaking"] = False


async def process_turns(call_id, transcript_queue, writer, hangup_event, state, last_write):
    """Speaks the pinned greeting first, then processes each completed
    caller turn from the queue sequentially - one call is a single line of
    conversation, no need to handle overlapping LLM turns."""
    greeting_instruction = (
        f"Say, word for word: \"{AIAGENT_GREETING}\" - then, without pausing for a reply, "
        "immediately add ONE brief, engaging point about SASI college in a single short "
        "sentence. Keep the whole reply short overall - this is spoken audio, not text."
    )
    await speak_turn(call_id, greeting_instruction, writer, state, last_write)

    while not hangup_event.is_set():
        try:
            text = await asyncio.wait_for(transcript_queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        if text is None:
            break
        log.info("[%s] 1002 said: %s", call_id, text)
        await speak_turn(call_id, text, writer, state, last_write)


async def keepalive_writer(call_id, writer, hangup_event, last_write, state):
    """Fill idle gaps with silence so app_audiosocket's hardcoded 2s inactivity
    timeout doesn't fire while waiting on the Deepgram/Agno/ElevenLabs
    round trip. Suppressed while agent audio is actively streaming."""
    log.info("[%s] keepalive writer started", call_id)
    while not hangup_event.is_set():
        await asyncio.sleep(KEEPALIVE_CHECK_SECONDS)
        if hangup_event.is_set():
            break
        if state["is_agent_speaking"]:
            continue
        if time.monotonic() - last_write[0] >= KEEPALIVE_GAP_SECONDS:
            log.info("[%s] keepalive: filling idle gap with silence", call_id)
            write_audiosocket_packet(writer, KIND_AUDIO, SILENCE_FRAME)
            try:
                await writer.drain()
            except (ConnectionError, OSError) as exc:
                log.warning("[%s] keepalive: drain failed, stopping: %r", call_id, exc)
                break
            last_write[0] = time.monotonic()
    log.info("[%s] keepalive writer exiting", call_id)


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
    keepalive_task = asyncio.create_task(keepalive_writer(call_id, writer, hangup_event, last_write, state))

    stt_ws = None
    try:
        stt_ws = await deepgram_stt_session()
        log.info("[%s] Deepgram STT session ready (model=%s, language=%s)", call_id, DEEPGRAM_MODEL, DEEPGRAM_LANGUAGE)
        transcript_queue = asyncio.Queue()
        await asyncio.gather(
            pump_asterisk_to_stt(call_id, reader, stt_ws, hangup_event),
            pump_stt_transcripts(call_id, stt_ws, transcript_queue, state, hangup_event),
            process_turns(call_id, transcript_queue, writer, hangup_event, state, last_write),
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
