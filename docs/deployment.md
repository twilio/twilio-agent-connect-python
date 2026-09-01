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

This needs per-pod addressability — on Kubernetes, a headless Service plus
`POD_IP`/`POD_NAME`. Without it, leave the setting unset (the load balancer
domain remains the default) and route by `CallSid` at the balancer.

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
  session, which messaging doesn't have. Statelessly it would cost the same as
  `"always"` with worse relevance, so messaging channels raise at construction
  rather than degrade silently. Use `"always"`.
