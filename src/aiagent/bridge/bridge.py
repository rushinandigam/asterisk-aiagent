#!/usr/bin/env python3
"""
aiagent-bridge

TCP server speaking the Asterisk AudioSocket protocol on one side, and the
OpenAI Realtime API (speech-to-speech) over a WebSocket on the other.

Asterisk's AudioSocket app() sends/receives raw signed-linear 16-bit PCM,
mono, at the channel's negotiated rate. extensions_local.conf answers the
call with the ulaw codec, so that's 8kHz here. The OpenAI Realtime API
speaks 24kHz PCM16, so audio is resampled both ways with stdlib audioop.

AudioSocket wire format (one TCP connection per call):
    1 byte kind | 2 bytes length (big-endian) | <length> bytes payload
Kinds used here: 0x01 UUID (sent once by Asterisk), 0x10 audio (slin),
0x00 hangup/terminate.
"""
import asyncio
import audioop
import base64
import json
import logging
import os
import struct
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

ASTERISK_SAMPLE_RATE = 8000
OPENAI_SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2  # 16-bit PCM
OUTBOUND_FRAME_MS = 20
OUTBOUND_FRAME_BYTES = ASTERISK_SAMPLE_RATE * SAMPLE_WIDTH * OUTBOUND_FRAME_MS // 1000
SILENCE_FRAME = b"\x00" * OUTBOUND_FRAME_BYTES

# app_audiosocket.so hangs up after a hardcoded 2000ms of no activity on the
# socket. OpenAI's server-side VAD/response latency can exceed that during
# normal "thinking" pauses, so a keepalive writer fills gaps with silence.
KEEPALIVE_GAP_SECONDS = 0.5
KEEPALIVE_CHECK_SECONDS = 0.2

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime")
OPENAI_VOICE = os.environ.get("OPENAI_REALTIME_VOICE", "alloy")
OPENAI_INSTRUCTIONS = (os.environ.get("AIAGENT_INSTRUCTIONS") or (
    "You are a warm, energetic SASI (the college at sasi.ac.in) admissions campaign caller - not "
    "a passive helpdesk. This call may be inbound (someone dialed in) or outbound (you are calling "
    "a prospective student/parent); either way, YOU drive the conversation. Never open with or fall "
    "back to a generic question like 'how can I help you' or sit back waiting to be asked something - "
    "instead, proactively and enthusiastically promote SASI: lead with genuinely engaging, specific "
    "things about the college (courses, placements, faculty, facilities, achievements, campus life) "
    "and keep building on them, the way a proud staff member running an admissions drive would. "
    "After answering any question, don't just stop - continue the pitch by elaborating on another "
    "relevant good thing about SASI that follows naturally from the context. "
    "For ANY factual claim about SASI - courses, departments, admissions, fees, facilities, "
    "placements, faculty, vision/mission, contact details, history, anything about the college - "
    "you MUST call the search_college_info tool first and base what you say strictly on what it "
    "returns. Never invent or guess facts about SASI. If the tool returns nothing relevant, say "
    "you don't have that specific detail right now, offer to connect them with the admissions "
    "department, and keep the conversation going with another strength of SASI. If the caller asks "
    "about anything NOT related to SASI college (general knowledge, other organizations, personal "
    "topics, etc.), politely explain you can only talk about SASI College, and steer the "
    "conversation right back into promoting the institution."
)) + (
    " Speak slowly and clearly, at a measured pace, enunciating each word. "
    "Never rush your answers - pause briefly between sentences. "
    "Default to Telugu: greet and start the conversation in Telugu. If the caller speaks "
    "to you in a different language (English, Hindi, etc.), switch to and continue the "
    "rest of the conversation in that language instead."
)
OPENAI_GREETING = os.environ.get("AIAGENT_GREETING") or "SASI కళాశాలకు స్వాగతం!"
OPENAI_SPEED = float(os.environ.get("OPENAI_REALTIME_SPEED") or 0.85)
OPENAI_REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={OPENAI_MODEL}"


async def read_audiosocket_packet(reader: asyncio.StreamReader):
    header = await reader.readexactly(3)
    kind, length = header[0], struct.unpack(">H", header[1:3])[0]
    payload = await reader.readexactly(length) if length else b""
    return kind, payload


def write_audiosocket_packet(writer: asyncio.StreamWriter, kind: int, payload: bytes = b""):
    writer.write(bytes([kind]) + struct.pack(">H", len(payload)) + payload)


async def openai_session(call_id: str):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    ws = await websockets.connect(OPENAI_REALTIME_URL, additional_headers=headers, max_size=None)
    await ws.send(json.dumps({
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": OPENAI_INSTRUCTIONS,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": OPENAI_SAMPLE_RATE},
                    # Default threshold (0.5) and silence_duration_ms (500) are tuned for a
                    # close-talking mic; over a phone line, the agent's own voice leaking back
                    # into the caller's mic (no headset/AEC on softphones) trips server_vad's
                    # default sensitivity, which reads as the caller barging in mid-sentence
                    # and cancels the response - sounding like answers cutting each other off.
                    # Less sensitive threshold + longer confirm window + far_field noise
                    # reduction cuts down on that false triggering.
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.7,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 700,
                        "create_response": True,
                        "interrupt_response": True,
                    },
                    "noise_reduction": {"type": "far_field"},
                    "transcription": {"model": "whisper-1"},
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": OPENAI_SAMPLE_RATE},
                    "voice": OPENAI_VOICE,
                    "speed": OPENAI_SPEED,
                },
            },
            "tools": [
                {
                    "type": "function",
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
            ],
            "tool_choice": "auto",
        },
    }))
    # Wait for the server to confirm the session config before triggering a
    # response - sending response.create right after session.update (without
    # waiting for session.updated) intermittently causes OpenAI to return a
    # generic server_error, since the session hasn't finished applying yet.
    async for message in ws:
        event = json.loads(message)
        etype = event.get("type")
        if etype == "session.updated":
            break
        if etype == "error":
            log.error("[%s] OpenAI Realtime error during session setup: %s", call_id, event)
        else:
            log.info("[%s] event: %s", call_id, etype)
    # Open immediately rather than waiting for the caller to speak first - also
    # keeps audio flowing back to Asterisk well within its 2s timeout. Only the
    # opening line itself is pinned (so every call starts the same way); the
    # agent then keeps going on its own into the campaign pitch per
    # OPENAI_INSTRUCTIONS, rather than stopping to ask how it can help.
    await ws.send(json.dumps({
        "type": "response.create",
        "response": {"instructions": (
            f"Start by warmly saying, word for word: \"{OPENAI_GREETING}\" - then, without "
            "pausing for a reply or asking how you can help, immediately continue by "
            "enthusiastically highlighting two or three genuinely engaging things about SASI "
            "college (e.g. courses, placements, faculty, facilities) to draw the caller in, the "
            "way a proud admissions campaign caller would. Call search_college_info first if you "
            "need specific facts to mention."
        )},
    }))
    log.info("[%s] OpenAI Realtime session ready (model=%s)", call_id, OPENAI_MODEL)
    return ws


async def pump_asterisk_to_openai(call_id, reader, ws, hangup_event):
    """Read SLIN frames from Asterisk, upsample 8k->24k, forward to OpenAI."""
    resample_state = None
    packets_in = 0
    bytes_in = 0
    try:
        while True:
            kind, payload = await read_audiosocket_packet(reader)
            if kind == KIND_HANGUP:
                log.info("[%s] caller hung up", call_id)
                break
            if kind != KIND_AUDIO or not payload:
                log.info("[%s] non-audio AudioSocket packet kind=0x%02x len=%d", call_id, kind, len(payload))
                continue
            packets_in += 1
            bytes_in += len(payload)
            if packets_in % 100 == 0:
                log.info("[%s] received %d audio packets (%d bytes) from Asterisk so far", call_id, packets_in, bytes_in)
            audio24k, resample_state = audioop.ratecv(
                payload, SAMPLE_WIDTH, 1, ASTERISK_SAMPLE_RATE, OPENAI_SAMPLE_RATE, resample_state
            )
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio24k).decode("ascii"),
            }))
    except asyncio.IncompleteReadError:
        log.info("[%s] AudioSocket connection closed by Asterisk", call_id)
    finally:
        log.info("[%s] total received from Asterisk: %d packets, %d bytes", call_id, packets_in, bytes_in)
        hangup_event.set()


async def pump_openai_to_asterisk(call_id, ws, writer, hangup_event, last_write, responding):
    """Read audio deltas from OpenAI, downsample 24k->8k, frame, send to Asterisk."""
    resample_state = None
    pending = b""
    bytes_out = 0
    active_response_id = None
    cancelled_response_ids = set()
    try:
        async for message in ws:
            if hangup_event.is_set():
                break
            event = json.loads(message)
            etype = event.get("type")
            if etype == "response.created":
                # Stop the keepalive's silence injection for the duration of
                # this response - OpenAI delivers audio in bursts with gaps
                # that can exceed our keepalive threshold, and injecting
                # silence mid-burst made the agent sound choppy/garbled.
                active_response_id = event.get("response", {}).get("id")
                responding[0] = True
            elif etype == "input_audio_buffer.speech_started":
                # Caller started talking over the agent (barge-in). Cancel
                # the in-flight response - otherwise its audio keeps
                # streaming in parallel with the next response and the two
                # overlap into "multiple voices" answering at once.
                if responding[0] and active_response_id:
                    log.info("[%s] caller interrupted - cancelling response %s", call_id, active_response_id)
                    cancelled_response_ids.add(active_response_id)
                    await ws.send(json.dumps({"type": "response.cancel"}))
                    responding[0] = False
                    pending = b""
            elif etype in ("response.audio.delta", "response.output_audio.delta"):
                if event.get("response_id") in cancelled_response_ids:
                    continue
                delta = base64.b64decode(event["delta"])
                audio8k, resample_state = audioop.ratecv(
                    delta, SAMPLE_WIDTH, 1, OPENAI_SAMPLE_RATE, ASTERISK_SAMPLE_RATE, resample_state
                )
                pending += audio8k
                while len(pending) >= OUTBOUND_FRAME_BYTES:
                    frame, pending = pending[:OUTBOUND_FRAME_BYTES], pending[OUTBOUND_FRAME_BYTES:]
                    write_audiosocket_packet(writer, KIND_AUDIO, frame)
                    bytes_out += len(frame)
                await writer.drain()
                last_write[0] = time.monotonic()
            elif etype in ("response.audio.done", "response.output_audio.done"):
                log.info("[%s] wrote %d bytes of agent audio to Asterisk", call_id, bytes_out)
                bytes_out = 0
            elif etype == "response.done":
                cancelled_response_ids.discard(active_response_id)
                responding[0] = False
                last_write[0] = time.monotonic()
            elif etype == "conversation.item.input_audio_transcription.completed":
                log.info("[%s] 1002 said: %s", call_id, event.get("transcript", "").strip())
            elif etype == "conversation.item.input_audio_transcription.failed":
                log.warning("[%s] transcription failed: %s", call_id, event.get("error"))
            elif etype in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
                log.info("[%s] agent said: %s", call_id, event.get("transcript", "").strip())
            elif etype == "response.function_call_arguments.done":
                tool_call_id = event.get("call_id")
                try:
                    args = json.loads(event.get("arguments") or "{}")
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
                await ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": json.dumps(output),
                    },
                }))
                await ws.send(json.dumps({"type": "response.create"}))
            elif etype == "error":
                log.error("[%s] OpenAI Realtime error: %s", call_id, event)
            else:
                log.info("[%s] event: %s", call_id, etype)
    except websockets.exceptions.ConnectionClosed:
        log.info("[%s] OpenAI Realtime connection closed", call_id)
    finally:
        hangup_event.set()


async def keepalive_writer(writer, hangup_event, last_write, responding):
    """Fill idle gaps with silence so app_audiosocket's hardcoded 2s inactivity
    timeout doesn't fire while waiting on OpenAI. Suppressed while a response
    is actively streaming - OpenAI's own delivery gaps between audio bursts
    can exceed our threshold, and injecting silence mid-burst sounds garbled."""
    while not hangup_event.is_set():
        await asyncio.sleep(KEEPALIVE_CHECK_SECONDS)
        if hangup_event.is_set():
            break
        if responding[0]:
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

    if not OPENAI_API_KEY:
        log.error("[%s] OPENAI_API_KEY is not set, hanging up", call_id)
        write_audiosocket_packet(writer, KIND_HANGUP)
        await writer.drain()
        writer.close()
        return

    # Start filling the socket with silence immediately - before the
    # (possibly slow) OpenAI handshake - so app_audiosocket's hardcoded 2s
    # inactivity timeout never gets a chance to fire during connection setup.
    hangup_event = asyncio.Event()
    last_write = [time.monotonic()]
    responding = [False]
    keepalive_task = asyncio.create_task(keepalive_writer(writer, hangup_event, last_write, responding))
    ws = None
    try:
        ws = await openai_session(call_id)
        await asyncio.gather(
            pump_asterisk_to_openai(call_id, reader, ws, hangup_event),
            pump_openai_to_asterisk(call_id, ws, writer, hangup_event, last_write, responding),
        )
    finally:
        hangup_event.set()
        keepalive_task.cancel()
        try:
            await keepalive_task
        except asyncio.CancelledError:
            pass
        if ws is not None:
            await ws.close()
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
