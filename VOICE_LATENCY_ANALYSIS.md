# Voice Channel Latency Analysis

Test notes from live phone calls against the streaming voice example
(`voice_streaming.py`), using ad-hoc latency logging added to `VoiceChannel`
plus Twilio ConversationRelay/Voice Insights event timelines pulled via the
REST API. Numbers are single-sample measurements from real calls, not
load-test averages — treat trends as directional, not statistically
rigorous.

## What was instrumented

Added `time.monotonic()`-based timing logs to
[`src/tac/channels/voice/channel.py`](src/tac/channels/voice/channel.py), all
emitted at `INFO` level prefixed `Latency:`:

| Log line | Measures | Location |
|---|---|---|
| `setup to first prompt (includes user speech + STT)` | Time from WebSocket `setup` message to the first `prompt` message — dominated by the caller's own speaking time plus Twilio's STT | `handle_websocket` |
| `conversation orchestrator poll/init` | Time spent polling CO for the ConversationRelay-created conversation (first turn only) | `_initialize_conversation` |
| `memory retrieval` | `_retrieve_memory_if_enabled` duration | `_handle_prompt` |
| `on_message_ready callback` | Time in the app's `on_message_ready` callback (LLM call) | `_handle_prompt` |
| `time to first token` | Streaming responses only — time from send start to first chunk written to the websocket | `send_response` |
| `streaming response complete` / `response sent` | Total time to finish writing the response to the websocket | `send_response` |
| `total prompt handling` | End-to-end time from prompt received to response fully sent | `_handle_prompt` |

These cover everything **inside our own code**. They do not cover Twilio-side
STT/TTS internals or PSTN audio transport — for that we pulled
`client.insights.v1.calls(call_sid).events.list()` (ConversationRelay
`start_of_agent_speech` / `end_of_customer_speech` / `first_token_received` /
etc.), which required enabling Voice Insights Advanced Features on the
account first (initially 404'd until enabled).

## Test calls

### 1. `memory_mode` default (`"never"`)

Three calls, `CA3af7c09...`, `CA02ebb0...`, `CA856249...`:

| Stage | Call A | Call B | Call C |
|---|---|---|---|
| setup → first prompt (speech + STT) | 5711.6 ms | — | — |
| CO poll/init (first turn) | 1049.4 ms | — | — |
| Memory retrieval | 0.2 ms (disabled) | 0.2 ms | 0.2 ms |
| Time to first token | 1234.4 ms | — | — |
| Streaming complete | 1402.5 ms | — | — |
| Total prompt handling | 1403.1 ms | — | — |

Call B and C were cross-checked against Twilio's ConversationRelay event
timeline for a ground-truth, caller-perceived breakdown:

**Call B (`CA02ebb0...`)** — good network (0 packet loss, `ashburn/us1`):

```
end_of_customer_speech  → first_token_received     1996 ms
end_of_customer_speech  → start_of_agent_speech     2204 ms
```

**Call C (`CA856249...`)** — slightly worse network (1 packet lost, jitter
0.143 avg, `umatilla/us2`):

```
end_of_customer_speech  → first_token_received     2273 ms
end_of_customer_speech  → start_of_agent_speech     2463 ms
end_of_customer_speech  → end_of_agent_speech       5685 ms
```

**Key finding**: the user reported a ~5s perceived delay on call C, which did
not match the ~2.2–2.5s "wait before hearing anything" we could see from our
own logs or the `start_of_agent_speech` event. The discrepancy resolved once
we measured all the way to `end_of_agent_speech`: **5685 ms**, i.e. the user
was judging latency as "time until the whole reply finishes playing," not
just "time until the reply starts." That total splits into ~2.5s of actual
wait plus ~3.2s of TTS playback (reply length).

### 2. `memory_mode="once"` enabled

Changed `voice_streaming.py` to
`VoiceChannel(tac, config=VoiceChannelConfig(memory_mode="once"))`. Three
calls before/after a code change (see Optimization below):

| Stage | Before fix (`CA38c6d9...`) | After fix #1 (`CA689dd6...`) | After fix #2 (`CA2b666d...`) |
|---|---|---|---|
| Memory retrieval | 2713.4 ms | 2228.9 ms | 2546.0 ms |
| Time to first token | 926.8 ms | 1519.6 ms | 917.9 ms |
| Total prompt handling | 3842.4 ms | 4000.2 ms | 3812.7 ms |

Also confirmed `"once"` mode's cache invalidation works as designed —
`Invalidated cached memory on INACTIVE status` fires ~60-90s after the call
ends, once Conversation Orchestrator marks the conversation `INACTIVE`.

## Root causes identified

1. **STT (setup → first prompt), ~5.7–6.2s**: entirely on Twilio's side
   (caller's actual speaking time + ConversationRelay STT). Not something TAC
   code affects. This is the majority of wall-clock time in every call but
   isn't "latency" in the optimizable sense — most of it is the caller
   talking.
2. **LLM time-to-first-token, ~0.9–2.3s**: the primary optimizable latency on
   the "wait" side of the experience once memory/STT are accounted for.
3. **Memory retrieval, ~2.2–2.7s when enabled**: found that
   `TAC.retrieve_memory()` was calling `get_profile()` then
   `retrieve_memory()` **sequentially**, even though `retrieve_memory()` only
   needs `profile_id` (already known) and not the `get_profile()` result. See
   Optimization below.
4. **User-perceived latency ≠ time-to-first-token**: users judge the delay as
   "time until the reply is fully spoken," not "time until it starts." A
   reply that's slow to *start* but short feels faster than one that starts
   quickly but talks for 3+ seconds.

## Optimization implemented: parallelize `get_profile` + `retrieve_memory`

[`src/tac/core/tac.py`](src/tac/core/tac.py) — `TAC.retrieve_memory()` now
fires `get_profile()` and `retrieve_memory()` concurrently via
`asyncio.gather()` instead of awaiting them one after another, since the
memory recall call only needs the already-known `profile_id`, not the
profile response itself.

**Result**: memory retrieval dropped from 2713ms → 2229ms → 2546ms across two
follow-up calls — a modest, noisy improvement (not the ~50% reduction a clean
parallelization would suggest). This points to a deeper bottleneck.

## Root cause candidate, not yet fixed: no HTTP connection reuse

[`src/tac/context/base.py:98-105`](src/tac/context/base.py:98) —
`BaseAPIClient._get_client()` creates a **brand-new `httpx.AsyncClient` for
every single API call**, used once and torn down immediately
(`async with self._get_client() as client:`). This means every
`get_profile()` / `retrieve_memory()` / `lookup_profile()` call pays a fresh
TCP connection + TLS handshake with no keep-alive reuse, which likely
explains why parallelizing the two calls didn't roughly halve latency — both
calls independently pay the same fixed per-request connection cost, so
running them concurrently only removes the *serial* penalty, not the
per-request overhead itself.

The existing code comment ("avoid event loop issues") suggests this was a
deliberate tradeoff, likely to avoid sharing a client across event loops
(e.g. test runners spinning up a new loop per test). Fixing this safely would
mean lazily creating and caching one `httpx.AsyncClient` per
`(client instance, running event loop)`, invalidating/recreating it if the
loop changes — bigger blast radius than the `asyncio.gather` change since
`BaseAPIClient` is shared by `ConversationClient`, `KnowledgeClient`, and
`MemoryClient`. **Deferred per user request — do this next.**

## Other optimization ideas discussed (not implemented)

- **Prefetch memory during the STT window**: since `setup → first prompt` is
  already ~5.7–6.2s of otherwise-idle time (the caller is just talking),
  kicking off memory retrieval speculatively as soon as the call/conversation
  is known (rather than waiting for the first `prompt` event) could fully
  hide the ~2.2–2.7s memory cost behind time that's already being spent.
- **Smaller/faster LLM model** for shorter time-to-first-token.
- **Trim `conversation_history`** in `voice_streaming.py` to the last N turns
  — currently unbounded, so context (and prefill time) grows every turn.
- **Bypass Agents SDK overhead**: `Runner.run_streamed(agent, ...)` carries
  full agent-loop framework cost (tool dispatch, guardrails) even when the
  agent has no tools; a raw `client.responses.create(stream=True)` call might
  shave off framework overhead.
- **"Filler word" trick**: play a short pre-recorded "hmm, let me think…" the
  instant the prompt is processed, before the LLM has produced anything —
  masks perceived latency without changing actual LLM speed.
- **Try alternate TTS providers** — `TwiMLOptions.tts_provider` /
  `LanguageConfig.tts_provider` in
  [`src/tac/models/voice.py:150,223`](src/tac/models/voice.py:150) can be
  swapged per call; different providers may have different synthesis
  latency.
- **Reduce memory query scope** (`observations_limit` / `summaries_limit` /
  `communications_limit` in `TACConfig.memory_config`) for a smaller/faster
  recall query.

## Open items for next session

1. Implement connection reuse in `BaseAPIClient` (highest expected impact on
   memory latency; needs care around event-loop safety).
2. Re-test memory retrieval latency after the connection-reuse fix to see
   how much of the ~2.2–2.7s was TLS/TCP handshake overhead vs. actual
   Memory API processing time.
3. Consider prefetching memory during the STT window for `"once"` mode.
