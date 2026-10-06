# Deploying TAC at scale

TAC runs as N replicas behind an ordinary load balancer and **needs no shared
datastore** — no Redis, no database, not even as an option.

## The short version

| | Messaging (SMS, RCS, WhatsApp, Chat) | Voice |
|---|---|---|
| Where state lives | nowhere — derived per request | in the process holding the call's WebSocket |
| Load balancer | any replica, no stickiness | any replica for the WebSocket; see [instance affinity](#instance-affinity-for-voice) for the rest |
| Shutdown | nothing to drain | call `aclose()` — [drain](#draining-on-shutdown) |
| Extra API calls vs. a single instance | none by default | none |

## Messaging is stateless

A messaging channel keeps no conversation state between webhooks. Each request
builds its own `ConversationSession` from three things it has anyway: the
webhook payload, your channel configuration, and one `list_participants` call
the self-message check and reconciliation already needed. Nothing a session
holds is unrecoverable, so there is nothing to route stickily and nothing to
leak.

### The API-call budget

Per inbound message, in steady state:

| `memory_mode` | Calls | Which |
|---|---|---|
| `"never"` (default) | 2 | `GET /Participants`, `POST /Actions` |
| `"always"` | 4 | the above, plus the profile-trait fetch and `/Recall` |
| `"always"` with `fetch_profile_traits=False` | 3 | drops the trait fetch |

TAC's own outbound echo costs **zero** — the author address matches the
configured agent address, so it's discarded before any call is made.

`/Recall` never returns traits, so the trait fetch is a genuinely separate
call. If your prompts don't use `build_profile_prompt()`, turn it off:

```python
TACConfig(
    ...,
    memory_config=TwilioMemoryConfig(fetch_profile_traits=False),
)
```

The customer's `profile_id` is read straight off the participant list — no
profile lookup, and more reliable than resolving it from the address (which
guesses the identifier type and misses on CHAT and RCS).

### Reply with the session, not the id

```python
async def on_message(message: str, session: ConversationSession, memory) -> None:
    reply = await my_llm(message)
    await channel.send_response(session, reply)  # no extra API call
```

`send_response` still accepts a bare conversation id, but with no session to
consult it spends a `list_participants` call rebuilding one.

### Handlers must be idempotent

TAC deduplicates webhook retries with Twilio's `i-twilio-idempotency-token`,
in a bounded in-process cache. That catches a retry landing on the **same**
replica — the common case, and free. It cannot catch one landing elsewhere.

!!! warning "Contract"
    `on_message_ready` may be called twice for the same message in a
    multi-instance deployment. The blast radius is one extra LLM call and
    possibly one duplicate outbound message.

Tolerating that in the handler is cheaper than a distributed lock. If you
genuinely can't, route by `conversation_id` at the load balancer.

### Metadata across turns

Metadata you pass to `initiate_outbound_conversation` is stored on the
conversation in Conversation Orchestrator (keys of letters, digits, `.`, `_`
or `-`; string values up to 512 characters; at most 8 keys including TAC's
`direction`), so a reply can find it on any replica:

```python
async def on_message(text, session, memory):
    metadata = await session.conversation_metadata()
    appointment_id = metadata.get("appointment_id")
```

That costs one Conversation Orchestrator request on a replica that hasn't
seen the conversation, and nothing otherwise. `session.metadata` still
carries values between turns, including ones you write during a turn, but
only on the replica that handled the conversation before. Treat it as a
per-replica scratchpad. Entries that don't fit Conversation Orchestrator's
limits stay only in `session.metadata`, with a warning.

## Voice is pinned to one process, deliberately

A live WebSocket ties a call to the process that accepted it for the call's
whole lifetime, so that state is *correct* where it is. What matters is that it
is always cleaned up and that the call's other traffic can find it.

### Teardown is guaranteed

However the WebSocket closes, TAC frees the session, the socket registry, the
session manager entry, and any provider call state, then fires
`on_call_ended`.

### The three end-of-call hooks

They are not interchangeable:

| Hook | Fires | Instance | Carries |
|---|---|---|---|
| `VoiceChannel.on_call_ended` | WebSocket teardown, always | the one holding the call | the live session — transcript, `call_sid`, your `metadata` |
| `TAC.on_conversation_ended` | when the *conversation* closes: Conversation Orchestrator's CLOSED webhook in orchestrated mode, at teardown in relay-only and Media Streams | any instance | the session, rebuilt from Conversation Orchestrator if the call ended elsewhere |
| `VoiceChannel.on_call_status` | Twilio's Calls-API `status_callback`, only if registered before the call was placed | any instance | a `CallStatusEvent` — no session |

`on_call_ended` is the only place late-call in-memory state is still reachable:
a rebuild from Conversation Orchestrator restores identity — conversation id,
`call_sid`, profile, both participants — but not a transcript.

If Conversation Orchestrator closes a conversation while its call is still
live — a closed timeout during a long hold, or your code closing it —
`on_conversation_ended` fires at that moment with a snapshot of the live
session, and the call keeps running. `on_call_ended` still fires when the
call hangs up. CO starts a new conversation for the call's later traffic,
but the live session keeps the closed conversation's id; the new
conversation's own `on_conversation_ended` fires from a rebuilt session
when it closes. Until the call ends, `resolve_conversation_session_by_call_sid`
returns the old id on the replica holding the call and the new one
elsewhere, so key cross-replica correlation on `call_sid`.

### Instance affinity for voice

A call's out-of-band webhooks — status, AMD, recording, and the
ConversationRelay `<Connect action>` callback — carry only a `CallSid` and
arrive independently of the WebSocket. Pointed at a load balancer they land on
an arbitrary replica, where `get_conversation_session_by_call_sid` finds
nothing. TAC mints those URLs, so it can make them instance-specific:

```python
TACConfig(
    ...,
    voice_public_domain="voice.example.com",           # the load balancer
    instance_public_domain=os.environ["POD_ADDRESS"],  # this pod, directly
)
```

The WebSocket URL, the action URL, and every call-event callback then point at
this process, so a call's webhooks come back to the replica holding it.

This needs each replica to be reachable **from Twilio, over the public
internet**, at its own hostname with a valid TLS certificate (streams are
`wss://`). On Kubernetes that means per-pod Ingress hostnames with a wildcard
certificate. A headless Service isn't enough, because its DNS only resolves
inside the cluster. Many managed platforms can't give instances their own
public address at all, for example Cloud Run, App Runner, Azure Container
Apps, Heroku, or ECS behind an ALB.

Without it, leave the setting unset (the load balancer domain remains the
default) and either route by `CallSid` at the balancer or look the call up
with `VoiceChannel.resolve_conversation_session_by_call_sid`. For a
ConversationRelay call in orchestrated mode it falls back to Conversation
Orchestrator and returns an identity-only session — enough to correlate an
event with its conversation and customer, though not to reply on the call,
whose WebSocket is on another replica.

### What needs instance affinity

Everything TAC itself needs works without `instance_public_domain`: lifecycle
callbacks, analytics, and CallSid lookups. A few things do need the request to
reach the replica holding the call:

- **Per-call Media Streams session configs.** This is **required** when you
  run OpenAI Realtime or GPT-Live on more than one replica and use
  `on_inbound_call_session_config` or a per-call `session_config` on
  `initiate_outbound_conversation`. The config is computed on the replica that
  serves the TwiML webhook or places the call, and kept there in memory; only
  a token travels with the call. If the stream connects to another replica,
  the call runs with `default_session_config` (or fails if none is set), and
  TAC logs a warning naming `instance_public_domain`. With only
  `default_session_config`, nothing here needs affinity.
- **Acting on a live call from outside it.** Speaking with `send_response`,
  or reading the live transcript, only works on the replica holding the call.
  Do it from the call's own callbacks (`on_message_ready`, tools), which always
  run there. Or set `instance_public_domain` so call events (status, AMD,
  recording) land there too.
- **`memory_mode="once"` refresh during a call.** An INACTIVE webhook that
  reaches another replica doesn't refresh a live call's cached memory, so the
  call keeps the memory it fetched at the start.

### Draining on shutdown

Without a drain, a scale-in or redeploy drops live calls with no callback at
all: the socket dies with the process and nothing fires.

`TACFastAPIServer` wires this up for you — it registers
`VoiceChannel.aclose()` on the app's shutdown event, with the grace period from
`TACServerConfig.shutdown_grace_period` (30s by default).

```python
server = TACFastAPIServer(tac=tac, voice_channel=voice_channel)
server.start()  # aclose() runs on shutdown
```

Two cases need you to do it yourself:

- **You passed a FastAPI app with its own `lifespan=`.** Starlette ignores
  event handlers on such an app, so `await server.aclose()` in your lifespan's
  shutdown half.
- **You're not using `TACFastAPIServer`.** Call
  `await voice_channel.aclose()` from your framework's shutdown hook.

Draining refuses new WebSocket connections, waits up to the grace period for
calls to end naturally, then force-releases whatever remains so the hooks fire.
Fail your readiness probe *before* it runs so the balancer stops routing here,
and keep the grace period below your orchestrator's termination grace period or
the process is killed mid-drain.

## What TAC does not do

- **No shared state store.** Deliberately — it would put a distributed-systems
  dependency into an SDK whose appeal is dropping into an existing app, and
  there is almost nothing left worth caching.
- **No live session migration.** A call belongs to one process until it ends.
- **`memory_mode="once"` on messaging.** It caches a recall on a long-lived
  session, which messaging doesn't have. It's deprecated on messaging channels
  and removed in 3.0: until then it logs a deprecation warning and runs as
  `"always"`, which recalls with each message's text, so memory stays on with
  better relevance at one recall per message. Set `"always"` to silence the
  warning. `"once"` is unchanged on Voice.
