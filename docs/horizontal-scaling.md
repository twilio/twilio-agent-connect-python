# Horizontal scaling

TAC runs as N replicas behind an ordinary load balancer and **needs no shared
datastore** — no Redis, no database, not even as an option.

## The short version

| | Messaging (SMS, RCS, WhatsApp, Chat) | Voice |
|---|---|---|
| Where state lives | nowhere — derived per request | in the process holding the call's WebSocket |
| Load balancer | any replica, no stickiness | any replica for the WebSocket; see [instance affinity](#instance-affinity-for-voice) for the rest |
| Shutdown | nothing to drain | live calls end on SIGTERM; [let them finish](#shutting-down-without-dropping-calls) at the platform |
| Extra API calls vs. a single instance | none by default | none |

## Messaging is stateless

A messaging channel keeps no conversation state between webhooks. Each request
builds its own `ConversationSession` from three things it has anyway: the
webhook payload, your channel configuration, and one `list_participants` call
the self-message check and reconciliation already needed. Nothing a session
holds is unrecoverable, so there is nothing to route stickily and nothing to
leak. If that `list_participants` call fails, the message can't be answered: it
is dropped and `on_error` fires with `dropped_inbound`.

A session lasts one turn: from the webhook until `on_message_ready` returns.
`session.metadata` doesn't carry over to the next message, on this replica or
any other. To keep state between turns, store it yourself, keyed by
`session.conversation_id`.

### The API-call budget

Per inbound message that gets a reply, in steady state. A turn with no reply
makes no `POST /Actions`, so it costs one call fewer.

| `memory_mode` | Calls | Which |
|---|---|---|
| `"never"` (default) | 2 | `GET /Participants`, `POST /Actions` |
| `"always"` | 4 | the above, plus the profile-trait fetch and `/Recall` |
| `"always"` with `fetch_profile_traits=False` | 3 | drops the trait fetch |

TAC's own outbound echo costs **zero** — the author address is one of the
channel's configured senders, so it's discarded before any call is made. With
several senders per channel, each reply goes out from the one the customer
messaged: the webhook's recipient, or, when the webhook doesn't name one, the
sender the conversation's participants hold.

`/Recall` never returns traits, so the trait fetch is a genuinely separate
call. If your prompts don't use `build_profile_prompt()`, turn it off:

```python
TACConfig(
    ...,
    memory_config=TwilioMemoryConfig(fetch_profile_traits=False),
)
```

The customer's `profile_id` is read straight off the participant list once
Conversation Orchestrator has resolved it, so there's no profile lookup.
That's more reliable than resolving it from the address, which guesses the
identifier type and misses on CHAT and RCS. Until a `profile_id` is set, a
memory fetch falls back to that address lookup, at one extra call.

### Reply with the session, not the id

```python
async def on_message(message: str, session: ConversationSession, memory) -> None:
    reply = await my_llm(message)
    await channel.send_response(session, reply)  # no extra API call
```

`send_response` still accepts a bare conversation id, but with no session to
consult it spends a `list_participants` call rebuilding one. On Chat, the
rebuilt session takes the required `channelId` from the customer's or agent's
CHAT participant address; when neither carries one, the send raises, so pass
the session there.

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

### Outbound metadata across turns

Metadata you pass to `initiate_outbound_conversation` is stored on the
conversation in Conversation Orchestrator (keys of up to 128 letters, digits,
`.`, `_` or `-`; string values up to 512 characters; at most 8 keys), so a reply
can find it on any replica:

```python
async def on_message(text, session, memory):
    metadata = await session.conversation_metadata()
    appointment_id = metadata.get("appointment_id")
```

That costs one Conversation Orchestrator request per turn that calls
`conversation_metadata()`, kept for the rest of that turn. Entries that don't
fit Conversation Orchestrator's limits are only on the session
`initiate_outbound_conversation` returns, with a warning.

### When a conversation closes

`on_conversation_ended` fires on whichever replica receives Conversation
Orchestrator's CLOSED webhook, for every closed conversation in your
conversation configuration that has a participant on that channel. That
includes conversations this process never handled, for example after a
restart. Don't assume an earlier `on_message_ready` ran on the same replica.

The session is built from the conversation's participants: the same
`profile_id`, customer and agent as during the conversation. Read the
conversation's metadata with `await session.conversation_metadata()`, which
uses the webhook's own copy when it carries one; `session.metadata` holds
nothing from earlier turns. It carries no profile traits (`session.profile`) or
cached memory, and its `started_at` is when it was rebuilt.

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
| `VoiceChannel.on_call_ended` | WebSocket teardown, always | the one holding the call | the live session — `call_sid`, your `metadata`, and on Media Streams the transcript |
| `TAC.on_conversation_ended` | when the *conversation* closes: Conversation Orchestrator's CLOSED webhook in orchestrated mode (or `end_call()`), at teardown in relay-only and Media Streams | orchestrated: any instance; otherwise the one holding the call | on the instance that held the call, its session as it ended; elsewhere, one rebuilt from Conversation Orchestrator |
| `VoiceChannel.on_call_status` | Twilio's Calls-API `status_callback`, only if registered before the call was placed | any instance | a `CallStatusEvent` — no session |

`on_call_ended` is the one hook guaranteed to see the live session: it fires
on the instance holding the call, every time. In orchestrated mode, that
instance keeps the ended session (for up to a day, bounded) so a CLOSED
reaching it hands `on_conversation_ended` the same session, `metadata`
included. A CLOSED reaching any other replica gets a session rebuilt from
Conversation Orchestrator, which restores identity — conversation id,
`call_sid`, profile, both participants — but not your `metadata`. With more
than one replica, read call state in `on_call_ended`.

If Conversation Orchestrator closes a conversation while its call is still
live — a non-null voice closed timeout expiring during a long silent hold,
or your code closing it — `on_conversation_ended` fires at that moment and
the call keeps running. On the replica holding the call it fires from a
snapshot of the live session; on any other replica, from a session rebuilt
from Conversation Orchestrator (identity only). `on_call_ended` still fires
when the call hangs up.

A closed conversation takes no new communications, so the call's later
traffic starts a new conversation in Conversation Orchestrator. The live
session keeps the closed conversation's id; the new conversation's own
`on_conversation_ended` fires from a rebuilt session when it closes. Until
the call ends, `resolve_conversation_session_by_call_sid` returns the closed
conversation's id on the replica holding the call. Elsewhere it returns the
new conversation once one exists (an active conversation wins, then the
newest), and the closed one before that. Key cross-replica correlation on
`call_sid`, not the conversation id.

### Instance affinity for voice

A call's out-of-band webhooks — status, AMD, recording, and the
ConversationRelay `<Connect action>` callback — carry only a `CallSid` and
arrive independently of the WebSocket. Pointed at a load balancer they land on
an arbitrary replica, where `get_conversation_session_by_call_sid` finds
nothing. TAC mints the URLs it controls, so it can make them instance-specific:

```python
TACConfig(
    ...,
    voice_public_domain="voice.example.com",           # the load balancer
    instance_public_domain=os.environ["POD_ADDRESS"],  # this pod, directly
)
```

`TACConfig.from_env()` reads it from `TWILIO_INSTANCE_PUBLIC_DOMAIN`.

The WebSocket URL and the action URL then point at this process, and so does
every status, AMD and recording callback on calls TAC places. On inbound
calls those callbacks come from the phone number's own configuration, which
TAC doesn't set, so they still reach the load balancer: route them by
`CallSid` or look the call up as below.

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
- **Acting on a live call from outside it.** Speaking with `send_response`
  (ConversationRelay; Media Streams has no text `send_response`), or reading
  the live transcript (Media Streams), only works on the replica holding the
  call.
  Do it from the call's own callbacks (`on_message_ready`, tools), which always
  run there. Or set `instance_public_domain` so call events (status, AMD,
  recording) on calls TAC places land there too.
- **`memory_mode="once"` refresh during a call.** An INACTIVE webhook that
  reaches another replica doesn't refresh a live call's cached memory, so the
  call keeps the memory it fetched at the start.

### Shutting down without dropping calls

Keeping live calls through a scale-in or redeploy is a platform job: TAC can't
keep a call alive once its process is told to stop. On SIGTERM, uvicorn (which
`TACFastAPIServer` runs on) closes every open WebSocket at once with code 1012,
and TAC's [teardown](#teardown-is-guaranteed) fires `on_call_ended` for each
call. Nothing leaks, but the calls are cut.

To let calls finish instead, don't send SIGTERM while they're running:

1. **Stop new calls reaching the replica.** Fail its readiness probe, or
   deregister it from the load balancer, first.
2. **Wait before stopping the process.** Delay SIGTERM long enough for the
   calls you want to protect to end, for example with a `preStop` hook.
3. **Give the platform room for that wait.** The orchestrator's termination
   grace period must cover the delay plus the time `on_call_ended` handlers
   need, or the process is killed mid-call.

On Kubernetes:

```yaml
spec:
  terminationGracePeriodSeconds: 330   # the preStop wait, plus time for teardown
  containers:
    - name: tac
      readinessProbe: { ... }          # fail this when draining
      lifecycle:
        preStop:
          exec:
            command: ["sleep", "300"]  # the longest call you want to keep
```

On a server other than uvicorn, check that it closes WebSocket connections on
shutdown. A process that exits with sockets still open runs no teardown, so no
`on_call_ended` fires for those calls.

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
