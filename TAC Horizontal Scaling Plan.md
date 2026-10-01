---
tags: [tac, architecture, scaling, python-sdk]
project: "[[TAC - Twilio Agent Connect]]"
---

# TAC Horizontal Scaling Plan

**Goal.** Make the TAC Python SDK safe to run as N replicas behind a load balancer with no sticky sessions for messaging, and with bounded, provably-cleaned-up state for voice.

Two different problems, two different answers:

- **Messaging (SMS, RCS, WhatsApp, Chat)** — webhook-driven, request/response. Any instance can serve any webhook. Fix = **derive the session per request**, hold nothing between webhooks.
- **Voice (ConversationRelay, Media Streams)** — a live WebSocket pins the call to one process. State is *correctly* instance-local. Fix = **guarantee teardown on disconnect** and **route the call's out-of-band webhooks back to the instance that owns the socket**.

**Non-goals.** Any Redis/database dependency — mandatory *or* optional (see §5); migrating live sessions between instances; changing the meaning of existing callbacks.

---

## 1. State inventory

Every piece of mutable instance state in `src/tac/`, and whether it survives horizontal scaling.

| State                   | Location                             | Holds                                     | Verdict                                                 |
| ----------------------- | ------------------------------------ | ----------------------------------------- | ------------------------------------------------------- |
| `_conversations`        | `channels/base.py:62`                | `dict[conv_id, ConversationSession]`      | **Unsafe (messaging)** / correct-but-must-drain (voice) |
| `_processed_tokens`     | `channels/base.py:65`                | webhook idempotency tokens, LRU 10k       | **Unsafe** — retry on another instance re-processes     |
| `_websockets`           | `channels/websocket_manager.py:37`   | live socket objects                       | Correct — inherently local                              |
| `_sessions`             | `session/thread_safe.py:30`          | `SessionState` + in-flight `asyncio.Task` | Correct — inherently local                              |
| `_calls`                | `.../openai_realtime/provider.py:88` | Twilio + OpenAI socket pair               | Correct — inherently local                              |
| `_call_session_configs` | `.../openai_realtime/provider.py:91` | per-call OpenAI session config            | **Unsafe + leaks** — see §4.3                           |
| TAC callbacks           | `core/tac.py:109-127`                | function refs                             | Stateless, fine                                         |

### Concrete defects this surfaces

1. **`CONVERSATION_UPDATED` is silently dropped on the wrong instance.** `_is_event_for_this_channel` returns `False` when the conversation isn't in the local dict (`channels/base.py:158-164`), and `_handle_conversation_updated` bails again at `channels/messaging.py:402-404`. With N instances the CLOSED webhook has a 1/N chance of reaching the instance holding the session, so `on_conversation_ended` usually never fires and the session leaks for the process lifetime.

2. **Voice orchestrated mode never tears down on hangup.** `_cleanup_connection` deliberately skips `_end_conversation` when the orchestrator is enabled (`.../conversation_relay/provider.py:848-852`), waiting for CO's CLOSED webhook — which lands on a random instance. The socket closes, the `ConversationSession` stays. This is the single largest voice leak.

3. **`_call_session_configs` is a cross-request dict with no eviction.** Written during the TwiML HTTP webhook (`provider.py:158`), read when the WebSocket connects (`provider.py:378`) — two separate requests that can hit different instances. Entries for calls that are never answered or never stream are never popped. Same applies to the outbound `session_config_token` entry (`provider.py:244`).

4. **`get_conversation_session_by_call_sid` scans local state only** (`voice/channel.py:389`). Status / AMD / recording / `<Connect action>` callbacks carry only a `CallSid` and arrive as an independent traffic class, so on a multi-instance deployment they resolve to `None`.

5. **Idempotency is per-process.** Twilio's retry of a webhook can be double-processed by a second instance.

---

## 2. Messaging: derive the session per request

The insight is that for messaging, `ConversationSession` holds **nothing that isn't already in the webhook payload or retrievable from Conversation Orchestrator**:

| Field | Source |
|---|---|
| `conversation_id` | webhook |
| `author_info.address` / `.participant_id` | `COMMUNICATION_CREATED` payload |
| `metadata["channel_id"]` | `Communication.channel_id` |
| `ai_agent_info.address` | config (`phone_number` / `whatsapp_number` / `rcs_sender_id` / `agent_address`) — no participant id needed, see §2.2 |
| `profile_id` | customer participant — only when `memory_mode != "never"` |
| `cached_memory` | not reconstructible — which is why `"once"` is removed from messaging (§2.7) |
| `pending_handoff_data` | voice only |

So the session becomes a **per-request value object**, constructed at the top of `process_webhook` and discarded when it returns. `MessagingChannel` stops touching `self._conversations` entirely.

### 2.1 Session construction — zero API calls

```python
def _resolve_session(
    self, conv_id: str, communication: Communication
) -> ConversationSession:
    session = ConversationSession(
        conversation_id=conv_id,
        channel=self.get_channel_name(),
    )
    session.author_info = AuthorInfo(
        address=communication.author.address,
        participant_id=communication.author.participant_id,
    )
    session.ai_agent_info = AuthorInfo(address=self.get_agent_address(session).address)
    if communication.channel_id:
        session.metadata["channel_id"] = communication.channel_id
    return session
```

Synchronous, no network. Everything comes from the webhook payload plus config:

- `author.participant_id` is a **required** field on `CommunicationParticipant` (`models/conversation.py:273`), so the customer side is free.
- The agent address is pure config on every channel — `phone_number` (SMS/voice), `whatsapp_number`, `rcs_sender_id`, `agent_address` (chat).
- `channel_id` (chat's required `channelSettings.channelId`) comes off the `Communication`.

### 2.2 Why no `GET /Participants` — the Actions API resolves by address

`createAction` (`POST /v2/Conversations/{cid}/Actions`) supports **three** resolution modes for both `payload.from` and each `payload.to[]`:

1. `participantId` + `channel` — resolves the address from the participant's registered addresses
2. `participantId` alone — works when the participant has exactly one address
3. **`address` + `channel` — explicit address**

TAC's own model already documents this (`models/conversation.py:430-455`: "Either `participant_id` or `address` must be supplied"), but the send path only ever exercises mode 1, which is what forces the participant lookup.

Switching `from` to mode 3 and `to` to the `participant_id` already in the webhook makes **sending require no lookup at all**. `list_participants` is not a send dependency — it serves three other purposes today:

| Consumer | Where | Frequency **today** | Needed when |
|---|---|---|---|
| `_is_own_message` fallback | `messaging.py:130` | **every inbound message** | author address ≠ configured agent address — i.e. every real customer message |
| `_reconcile_participants` | `messaging.py:533-662` | first message per conversation | always — but free after the first message, see §2.3 |
| `profile_id` for memory | `messaging.py:369-372` | first message per conversation | `memory_mode != "never"`, and address→profile lookup is unreliable (see §2.7) |

The first row is the one that matters and it is easy to miss. `_is_own_message` runs unconditionally at `messaging.py:311`; its "fast path" only short-circuits when the author *is* the agent (`is_default_agent_address`). For an actual inbound customer message the address never matches, so it falls through to `list_participants` **on every single message, today, already**.

Two consequences:

1. **Statelessness adds no participant lookup to the default path** — the call is already per-message. The first draft's "one extra GET per message" was wrong in both directions: it isn't extra, and it's already being paid.
2. **There's a redundancy to remove.** On a conversation's first message TAC calls `list_participants` twice — once in `_is_own_message`, once in `_reconcile_participants`. Fetch once at the top of `_handle_communication_created` and thread the list through both. Worth doing regardless of scaling.

**Net cost of stateless messaging with `memory_mode="never"` (the default): zero extra API calls.** With memory enabled it is not zero — see §2.7.

**Open question to settle empirically before P1.** The spec calls mode 3 "explicit address" and is silent on whether CO creates a participant when none owns that address. The top-level `POST /v2/Communications` endpoint explicitly does auto-create by `(address, channel)`, but `createAction` makes no such promise. Test both: (a) does delivery succeed, and (b) does the conversation end up with a correctly-typed `AI_AGENT` participant afterward? (b) matters even if (a) passes — CO's conversation summary and memory write depend on participant typing. If (b) fails, §2.3's reconcile pass stops being optional for v1-bridge accounts.

### 2.3 Reconciliation stays on, unconditionally — it is self-limiting

An earlier draft proposed hiding `_reconcile_participants` behind a `reconcile_participants: bool = False` flag on the grounds that it is "a v1-bridge-only repair pass". **Withdrawn on two counts.**

**First, the v1-bridge-only claim is unverified.** It comes from the method's own docstring and the commit that added it (`76d87a9`, PR #19, "handle v1-bridge UNKNOWN"). Nothing in this repo establishes what CO's *v2-native* capture assigns as a participant type on an inbound message — plausibly `UNKNOWN` for TAC's own number too, since capture has no way to know the address belongs to an AI agent. There is weak evidence pointing that way: the ConversationRelay path reads participants created by CO capture and simply tolerates finding no agent (`conversation_relay/provider.py:294`, `if agent_participant:`). That tolerance would hide the exact same gap on voice, because voice replies over the WebSocket and never needs the agent's participant id. So an untyped agent participant could be universal and invisible today. **Do not gate behavior on this until it is measured against a v2-native account** — it's listed with the §2.2 open question.

**Second, and decisively: it doesn't matter, because reconcile costs nothing in steady state.** The claim that per-message reconcile is expensive was simply wrong:

- **Zero extra reads.** It reuses the `list_participants` response already fetched for `_is_own_message` (§2.2).
- **Zero writes after the first message.** The happy-path row — agent correctly typed, customer correctly typed — returns immediately with no `PUT` and no profile work (`messaging.py:594-636`; asserted by `test_agent_plus_customer_no_puts`). Once the first message has repaired the types, every subsequent message takes that row.

So stateless per-message reconcile costs exactly what today's once-per-conversation reconcile costs: the repair writes happen once, on the first message, and never again. The only difference is that the *decision* to skip is re-derived from CO each time instead of cached on a session — which is the whole point of being stateless.

**Keep it unconditional. No flag.** Fewer knobs, no way to misconfigure it, and correct whichever way the v1-bridge question resolves.

### 2.4 `get_agent_address` signature change

`ChatChannel.get_agent_address` reads `self._conversations` to recover `channel_id` (`channels/chat.py:66-73`). Change the abstract method from `get_agent_address(conversation_id: str)` to `get_agent_address(session: ConversationSession)`. Breaking for anyone subclassing `MessagingChannel` directly; accept it with a changelog note rather than carrying a shim, since the old signature cannot be made correct.

### 2.5 `send_response` — address mode, no lookup

`send_response` currently reads the session out of `self._conversations` (`messaging.py:209`) purely to recover two participant ids for the Actions payload (`messaging.py:217-223`). With §2.2's mode-3 resolution it needs neither:

```python
SendMessageActionPayload(
    from_=ActionParticipantRef(          # mode 3 — config address
        channel=channel_name,
        address=self.get_agent_address(session).address,
    ),
    to=[ActionParticipantRef(            # mode 1 — id straight off the webhook
        channel=channel_name,
        participant_id=session.author_info.participant_id,
    )],
    content=ActionTextContent(text=response),
    channel_settings=channel_settings,
)
```

Signature accepts either form so existing callers keep working:

```python
async def send_response(
    self,
    conversation: str | ConversationSession,
    response: str,
    role: str | None = None,
) -> None:
```

- Passed a `ConversationSession` (what `on_message_ready` already receives) — zero API calls.
- Passed a `str` — no session, so no `author_info`. Send with `from` = config address (mode 3) and `to` = the **customer address**, which the caller must now supply, or fall back to one `list_participants` to find the sole `CUSTOMER`. Document the session form as the preferred path; the bare-string form is the compatibility shim, not the design.

That last case is the only place a lookup survives in the default configuration, and only for callers who discard the session they were handed.

`initiate_outbound_conversation` already returns `InitiateConversationResult(conversation_id, session)`, so the outbound path is already shaped for callers to hold and hand back the session object.

### 2.6 `CONVERSATION_UPDATED` handling

Delete the `_conversations` branch from `_is_event_for_this_channel` (`base.py:158-164`) — it is a state-dependent filter and cannot survive.

Replace with a stateless path in `_handle_conversation_updated`:

- **Fast exit:** if `configuration_id` doesn't match config (already checked, `messaging.py:399`), or the status is `CLOSED` and **no `on_conversation_ended` callback is registered**, return immediately. This means the common case costs nothing.
- **`CLOSED` with a callback registered:** `list_participants(conv_id)`, confirm a participant addresses this channel (this is what replaces the local-tracking filter, and it correctly prevents a CHAT close from firing on the SMS channel), rebuild the session, invoke `trigger_conversation_ended`. Fires exactly once, on whichever instance Twilio picked — strictly better than today's "usually zero times".
- **`INACTIVE`:** existed only to invalidate the `"once"` cache. With `"once"` gone from messaging (§2.7), this branch is dead code on messaging channels — delete it. Voice keeps its own INACTIVE handling (`voice/channel.py:463-472`), which is still correct there.

### 2.7 Memory modes — where statelessness actually costs something

`memory_mode` defaults to `"never"` (`base.py:38`, `messaging.py:52`), so most deployments are unaffected. For the other two it is not free, and the cost is in a place the draft didn't look: `retrieve_memory` memoizes **two** things on the session, not one (`core/tac.py:173-214`).

- `session.profile_id` — resolved once, then reused (`tac.py:189`)
- `session.profile` — the trait fetch, guarded by `if profile_id and not profile` (`tac.py:200`)

Both memoizations disappear with a per-request session. But only one of them costs anything, because `profile_id` can be taken straight off the customer participant in the `list_participants` result §2.2 shows we already fetch — no `lookup_profile` needed. Steady-state calls per inbound message, `memory_mode="always"`:

| | Today | Stateless |
|---|---|---|
| profile id resolution | cached on session | free — off the participant list |
| profile traits (`get_profile`) | cached on session | 1 call |
| `/Recall` | 1 call | 1 call |
| **total** | **~1** | **2** |

So `memory_mode="always"` costs **one** extra call per message, entirely in the memory path — and §5 shows how to remove even that for apps that don't use profile traits.

**`memory_mode="always"` is otherwise already stateless.** It touches no cache and takes no lock (`base.py:303-321`) — it reads `profile_id` / `author_info` / `conversation_id` off the session and calls out. Nothing about it needs a change beyond the cost above.

#### Drop `"once"` from messaging; keep it on voice

`memory_mode="once"` is the only mode with genuinely non-reconstructible state — `cached_memory` plus an `asyncio.Lock` (`models/session.py:71-80`), invalidated on the INACTIVE webhook. A per-request session cannot hold any of it.

What it degrades into is the problem. Statelessly, `"once"` issues the same **3 calls per message** as `"always"`, but primes with `query=None` and `conversation_id=None` (`base.py:338`) — deliberately, so Memory skips the server-side query-inference step (`tac.py:153-158`). So the degraded mode is *cheaper per recall but blind to the turn's topic*: identical call count to `"always"`, strictly worse relevance, and the INACTIVE invalidation becomes a no-op. It is a mode whose entire value is the cache, running without a cache. The name stops being true — nothing happens "once".

**Remove `"once"` from messaging channels.** Narrow `MessagingChannelConfig.memory_mode` to `Literal["never", "always"]` and raise at construction with a message pointing at `"always"`. Fail loudly rather than silently serving worse memory — a silent degradation here is invisible in production and shows up as "the bot forgot things".

**Keep `"once"` on voice, unchanged.** It is the mode voice most needs and the one place it is genuinely correct:

- A voice session is pinned to an instance by its WebSocket for the whole call and lives in `_conversations` (`voice/channel.py:86` wires `memory_mode` through from provider config; `conversation_relay/provider.py:784` reads it per prompt). The cache has a real lifetime.
- Voice has a per-utterance latency budget in the hundreds of milliseconds. Three memory calls on every utterance is not viable; messaging is asynchronous and does not care.

`"once"` is effectively a voice feature that happens to be exposed on messaging. `MemoryMode` (`models/memory.py:6`) stays a three-value `Literal` for voice; only the messaging config narrows.

**Migration.** `tests/test_memory_mode_once.py` exercises `"once"` exclusively through `SMSChannel` — that file moves to voice or goes away. Voice's coverage (`test_voice_channel.py:367`, `:2873`) is unaffected. Changelog entry required; this is a breaking change for any messaging deployment using `"once"`, and `"always"` is the drop-in replacement.

#### `profile_id` is not always recoverable from the address

#### `profile_id` is not always recoverable from the address

`retrieve_memory` already falls back to `lookup_profile` by `author_info.address` when `profile_id` is unset (`tac.py:173-198`), which is why dropping the participant lookup is viable at all. But the fallback picks `id_type` by a bare substring test — `"email" if "@" in address else "phone"` (`tac.py:180`). That holds for SMS, WhatsApp and Voice. It does **not** hold for:

- **CHAT** — an opaque identity string gets looked up as a phone number and misses.
- **RCS** — an `rcs:`-prefixed sender likewise.

On a miss the code raises `ValueError` internally, which the outer handler catches and converts into the CO `list_communications` fallback (`tac.py:230-244`) — so memory silently degrades from profile recall to raw conversation history. Note `_resolve_customer_profile` doesn't cover this either; it bails for anything but SMS/VOICE (`messaging.py:676`).

**Therefore:** for CHAT and RCS with memory enabled, `profile_id` off the customer participant is the reliable source and the `list_participants` call has to stay. Gate it on `memory_mode != "never" and channel in ("CHAT", "RCS")` rather than making it unconditional — and since §2.2 shows the list is already being fetched for `_is_own_message`, threading that one result through costs nothing extra anyway.

### 2.8 Idempotency

No code change. `_is_duplicate_webhook`'s bounded `OrderedDict` (`base.py:117-135`) stays exactly as it is — it is already stateless-compatible (a pure cache, no correctness depends on it) and catches same-instance retries for free. The cross-instance gap is handled by documentation, not infrastructure — see §5.

---

## 3. Voice: guaranteed teardown

Voice state is legitimately local. The requirement is that **when the WebSocket closes, nothing is left behind** — and that a scale-in doesn't strand a call.

### 3.1 Separate *resource teardown* from *conversation ended*

These are two different events and the current code conflates them. Splitting them fixes the leak **without** moving `on_conversation_ended`.

Remove the `is_orchestrator_enabled()` guard at `.../conversation_relay/provider.py:848-852` so `_cleanup_connection` always frees local resources — the socket, the `SessionManager` entry, the `_conversations` entry, the model socket. But it fires a **new** hook rather than `on_conversation_ended`:

| Hook | Fires | Carries |
|---|---|---|
| `on_call_ended` (**new**) | WebSocket teardown — every provider, every mode, always | the live in-memory session: transcript, metadata, `call_sid` |
| `on_conversation_ended` (**unchanged**) | when the *conversation* ends | the session |

`on_conversation_ended` keeps today's meaning exactly:

- **Orchestrated mode** — fires on CO's `CONVERSATION_UPDATED`/CLOSED webhook, as now. Because the local session is gone by then, it rebuilds statelessly from CO exactly the way messaging does in §2.6. That makes it *more* reliable than today, where it only fires if the webhook happens to land on the instance holding the session.
- **Relay-only and Media Streams** — no CO conversation exists, so it fires at teardown. **This is already the behavior** (`conversation_relay/provider.py:848-852`, `openai_realtime/provider.py:657`), so nothing changes.

Net: no leak, no semantic change to the existing hook, and a new hook that is strictly more useful than what CLOSED-time reconstruction could ever provide — `on_call_ended` is the only place the **transcript** is still reachable. `OpenAIRealtimeProvider.get_transcript` already documents this cliff (`openai_realtime/provider.py:100-104`): the transcript lives on `session.metadata` and vanishes when the session is popped. Today the only way to catch it is `on_conversation_ended`; `on_call_ended` makes it a first-class hook.

**Naming — open.** `VoiceChannel` already has `on_call_status`, `on_amd`, `on_recording`, so `on_call_ended` fits the family, but it reads as redundant with `on_call_status`'s terminal `completed` event. They are genuinely different: `on_call_status` is a Twilio Calls-API webhook that requires `status_callback` to be registered, arrives out-of-band, can land on another instance, and carries no session. `on_call_ended` is local, synchronous with teardown, always fires, and carries the session. Worth documenting that distinction prominently whatever the name — `on_voice_session_ended` is the more accurate but uglier alternative.

### 3.2 The teardown invariant

After `handle_websocket` returns, for a given `conv_id`, all of these must hold:

```
conv_id not in channel._conversations
not ws_manager.has_websocket(conv_id)
not session_manager.has_session(conv_id)
conv_id not in provider._calls                 # media streams
conv_id not in provider._call_session_configs  # media streams
```

Ship this as a test helper, `assert_no_residual_state(channel, conv_id)`, and assert it in every voice test that opens a socket.

Both providers already run cleanup in `finally` and both already handle the awkward races (the CR provider adopts a late-completing `init_task`'s `conv_id` at `provider.py:429-433`; the media-streams provider cancels and awaits `model_reader` at `provider.py:340-347`). The gap is coverage, not structure — the invariant needs asserting under: normal hangup, abrupt disconnect mid-stream, disconnect before any prompt, `_initialize_conversation` raising, `_connect_model` raising, and task-level cancellation.

**Known accepted risk:** `SessionState.cancel_stream_task` gives up after a 5s timeout (`session/state.py:36-41`). A generator that swallows cancellation leaves the task running after the session is dropped. The session dict is cleaned either way; the task is not. Log it at error (already does) and document that `on_message_ready` implementations must be cancellation-safe.

### 3.3 Drain on shutdown

Add `VoiceChannel.aclose()` and wire a FastAPI lifespan shutdown in `TACFastAPIServer`:

1. Stop accepting new WebSocket connections (fail readiness first so the LB stops routing).
2. Wait, bounded by a configurable grace period, for active calls to end naturally.
3. Force-close whatever remains so `on_conversation_ended` fires for every session.

Without this, a Kubernetes scale-in drops live calls with no callback at all.

---

## 4. Voice: instance affinity

### 4.1 The problem

TAC mints the URLs Twilio posts back to — the WebSocket URL, `voice_action_path`, and the status/AMD/recording callbacks (`voice_call_event_path`). Today they all derive from a single `TACConfig.voice_public_domain`, i.e. the load balancer. So the WebSocket lands on instance A while the AMD callback for the same call lands on instance B, where `get_conversation_session_by_call_sid` finds nothing.

### 4.2 The fix

Because TAC generates those URLs, TAC can make them instance-specific. Add a resolver to `TACConfig`:

```python
instance_public_domain: str | None = None   # this pod's directly-routable host
```

When set, the TwiML builders use it for the WebSocket URL, the action URL, and every call-event callback URL. Every out-of-band webhook for a call then returns to the process holding that call's socket, and `get_conversation_session_by_call_sid` resolves.

The per-call override hook already exists — `host_twiml_options.websocket_url` (`voice/channel.py:195`) — so this generalizes an established mechanism to the remaining URLs rather than inventing one.

Deployment note for the docs: this needs per-pod addressability (a headless service + `POD_IP`/`POD_NAME` on Kubernetes). Where that isn't available, sticky-by-`CallSid` at the LB is the fallback, and the plain LB domain remains the default.

### 4.3 Media Streams `_call_session_configs`

Two fixes, both needed:

- **Correctness:** carry the inbound per-call session config through `<Stream>` custom parameters, exactly as the outbound path already carries `_tac_session_config_token` (`.../openai_realtime/provider.py:216-225`). The config then travels with the call instead of sitting in a dict on whichever instance served the TwiML request.
- **Leak:** give any remaining entries (the outbound token bridge) a TTL and a bounded size. A call that is never answered currently leaves its entry forever.

---

## 5. No shared store

**Decision: no shared store, no Redis, not even as an optional extra.** Earlier drafts proposed a pluggable `TACStateStore`. It is not needed, and adding it would put a distributed-systems dependency into an SDK whose whole appeal is that it drops into an existing app. Dropped from the plan.

Once `"once"` is gone from messaging (§2.7), only two things a store would have bought remain, and both have answers that need no infrastructure.

#### `profile_id` is free — take it off the participant list

This is what removes the need for a cache. CO already stores `profileId` on the customer participant, and §2.2 established that `list_participants` is **already being fetched on every inbound message** for `_is_own_message`. So `profile_id` comes out of a response we already have. No `lookup_profile` call, and it is more reliable than the address fallback (which misfires on CHAT and RCS — see §2.7).

Revised steady-state cost for `memory_mode="always"`, with no store anywhere:

| | Today | Stateless, no store |
|---|---|---|
| `profile_id` | cached on session | free — off the participant list |
| profile traits (`get_profile`) | cached on session | 1 call |
| `/Recall` | 1 call | 1 call |
| **total** | **~1** | **2** |

One extra call per message, not three. On an asynchronous messaging path that already awaits an LLM, that is not worth a Redis.

**And the last call is removable too.** `get_profile` exists only to populate `session.profile` for `build_profile_prompt()` (`models/session.py:84-131`). `/Recall` does **not** return traits — it returns observations, summaries and communications only (`models/memory.py:321-341`) — so it is a genuine second call, but a pointless one for any app that never calls `build_profile_prompt`. Gate it: `MemoryConfig.fetch_profile_traits: bool = True` (or skip automatically when no `trait_groups` are configured). Apps that don't use traits go back to **1 call per message — parity with today, fully stateless.**

#### Idempotency: keep the in-process LRU, document the bound

`_is_duplicate_webhook`'s bounded `OrderedDict` (`base.py:117-135`) stays exactly as it is. It catches the common case — a retry landing on the same instance — and costs nothing.

It cannot catch a retry that lands on a *different* instance. Rather than build infrastructure for that, state the contract plainly in the docs: **`on_message_ready` handlers must tolerate being called twice for the same message.** That is already true today for any multi-instance deployment, and it is the same guarantee every webhook consumer has to make. The blast radius of a duplicate is one extra LLM call and possibly one duplicate outbound message — real, but bounded, and cheaper to make idempotent at the handler than to prevent with a distributed lock.

If a deployment genuinely cannot tolerate it, the answer is sticky routing by `conversation_id` at the load balancer, not a store inside the SDK.

---

## 6. API call budget

The whole point of this section is that the plan is close to free. Counting every CO and Memory HTTP call TAC makes per **inbound messaging webhook**, before and after.

### Does reconciliation add calls per message? No.

It needs `list_participants`. That call is **already being made on every message today**, inside `_is_own_message` (`messaging.py:130`) — its fast path only short-circuits when the author *is* the agent, which never happens for a real customer message. Today TAC then calls `list_participants` a *second* time inside `_reconcile_participants` on a conversation's first message (`messaging.py:341-342`).

Fetch it once at the top of `_handle_communication_created` and share the result across `_is_own_message`, reconcile, and `profile_id` extraction. Reconcile's per-message read cost is then **zero**, and its write cost is zero after the first message (§2.3).

### Steady state — per inbound message, after the first

| Call | Today | After | Δ |
|---|---|---|---|
| `GET /Participants` (`_is_own_message`) | 1 | 1 — now shared with reconcile + `profile_id` | 0 |
| `GET /Participants` (reconcile) | 0 (cached on session) | 0 (shares the above) | 0 |
| Participant repair `PUT`/`POST` | 0 | 0 (happy-path row, §2.3) | 0 |
| `POST /Actions` (send) | 1 | 1 | 0 |
| **`memory_mode="never"` total** | **2** | **2** | **±0** |
| `lookup_profile` | 0 (cached) | 0 — free off the participant list (§5) | 0 |
| `get_profile` (traits) | 0 (cached) | 1 | **+1** |
| `/Recall` | 1 | 1 | 0 |
| **`memory_mode="always"` total** | **3** | **4** | **+1** |

**With the default `memory_mode="never"`: zero extra calls.** With `"always"`: exactly one, the profile-trait fetch — removable via `fetch_profile_traits=False` for apps that never call `build_profile_prompt()`, which takes it back to ±0 (§5).

### First message of a conversation — the plan is *cheaper*

| Call | Today | After |
|---|---|---|
| `GET /Participants` | **2** (`_is_own_message`, then reconcile) | **1** (shared) |
| repair `PUT`/`POST` | 0–2 | 0–2 (unchanged) |
| `POST /Actions` | 1 | 1 |

Deduplicating that double fetch is a **−1 call per conversation** win that has nothing to do with scaling and could ship on its own.

### Other events

| Event | Today | After |
|---|---|---|
| `CONVERSATION_UPDATED` / INACTIVE | 0 | 0 — branch deleted with `"once"` (§2.6) |
| `CONVERSATION_UPDATED` / CLOSED, no `on_conversation_ended` registered | 0 | 0 — fast exit (§2.6) |
| `CONVERSATION_UPDATED` / CLOSED, callback registered | 0 (usually never fires at all) | 1 `GET /Participants`, once per conversation — and it now actually fires |
| Any voice path (§3, §4) | — | 0 — teardown, hooks and affinity are all local |

### Summary

- Default configuration (`memory_mode="never"`): **no additional API calls**, and one fewer per conversation.
- `memory_mode="always"`: **+1 per message**, reducible to 0.
- Reconciliation, kept always-on: **+0 per message**.
- Voice: **+0**.
- One `GET /Participants` per *conversation* on CLOSED, and only when a callback wants it.

This is the reason §5 concludes no shared store is warranted: there is almost nothing left to cache.

---

## 7. Refactor prerequisites found in review

Four structural items the sections above imply but don't specify. Each is a place where a plausible reading of the plan produces a regression, so they are written out explicitly.

### 7.1 Move session ownership out of `BaseChannel`

§2 says "`MessagingChannel` stops touching `self._conversations`" — but `_conversations` (`base.py:62`), `_start_conversation` (`base.py:214`) and `_end_conversation` (`base.py:253`) all live on `BaseChannel`, and voice depends on all three across seven call sites (`voice/channel.py:345,462`, `conversation_relay/provider.py:204,281,367,852`, `openai_realtime/provider.py:362,657`).

Move all three down to `VoiceChannel`. `BaseChannel` keeps only what both still share: the dedup LRU and `_retrieve_memory_if_enabled`. Messaging constructs sessions inline (§2.1) and calls `tac.trigger_conversation_ended` directly. Without this the two channel families quietly keep sharing mutable state neither one should own.

### 7.2 Split `_end_conversation` into release vs. ended

`_end_conversation` currently does two things at once: pop the session, then fire `on_conversation_ended` (`base.py:263-266`). §3.1's hook split needs them separate:

- `_release_session(conv_id)` — pop, free resources, fire **`on_call_ended`**. Called from every WebSocket teardown path.
- `trigger_conversation_ended` — fired independently: on CO CLOSED in orchestrated mode, at release in relay-only and Media Streams (no CO conversation exists).

All seven call sites listed in §7.1 need re-pointing at the right one. Getting this wrong is how `on_conversation_ended` ends up firing twice, or at the wrong moment.

### 7.3 Voice CLOSED handling must be rebuilt statelessly too — this is a real trap

`VoiceChannel.process_webhook` bails out when there is no local session (`voice/channel.py:457-459`). Once §3.1 frees the session at WebSocket teardown, that condition is **always** true in orchestrated mode — so `on_conversation_ended` would fire **never** for voice.

§3.1 claims it "rebuilds statelessly the way messaging does in §2.6", but §2.6 is written against `MessagingChannel`. Voice needs its own equivalent: on CLOSED, if an `on_conversation_ended` callback is registered, rebuild a minimal session from CO and fire it. This is a distinct work item, not a consequence of the messaging change, and it is the single easiest way to ship a silent regression from this plan.

### 7.4 Preserve the reply recipient — `to` is not always the author

§2.5's snippet uses `session.author_info.participant_id` for `to`. That is correct **only because** reconcile overwrites it (`messaging.py:369-372`) with the participant it resolved as `CUSTOMER` — "the participant on this channel that isn't at our address" (`messaging.py:627-634`) — which is not necessarily the author of the inbound message. They diverge whenever a non-customer authored the message.

A natural-looking implementation that reads `communication.author.participant_id` directly would silently change who receives the reply. Keep the reconcile overwrite, and add a test that pins the recipient when author ≠ resolved customer.

(Chat is unaffected: `reconcile_customer_type = False` means reconcile returns no customer and the webhook author's id is used deliberately — `chat.py:41`.)

### 7.5 Minor

- `_initiate_messaging_conversation` calls `_start_conversation` (`messaging.py:487`) and pops on error (`messaging.py:519`). Both become plain construction / dead code.
- Deleting the `_conversations` branch from `_is_event_for_this_channel` (`base.py:158-164`) is safe for voice, which re-checks its own dict at `voice/channel.py:457`. Verified, not assumed.
- `_is_own_message`'s fast path means TAC's own outbound echo costs **zero** API calls (`is_default_agent_address` → immediate `True`), which is what makes the §6 budget hold. Mode-3 send keeps this true, since `from` is the same config address the fast path matches on.
- Pre-existing, unchanged, but adjacent: `AGENT_TYPES` excludes `HUMAN_AGENT` (`base.py:21`), so a human agent's message is not treated as TAC's own and the LLM will reply to it. The stateless rewrite touches this exact code — leave the behavior alone, but don't mistake it for a new bug.

---

## 8. Delivery phases

| Phase | Scope | Breaking? |
|---|---|---|
| **P0a** | Prerequisite refactors: move session ownership to `VoiceChannel` (§7.1), split `_end_conversation` (§7.2). Pure refactor, no behavior change, lands green on the existing suite. | No |
| **P0b** | Voice resource teardown on disconnect + new `on_call_ended` hook (§3.1), **stateless voice CLOSED rebuild (§7.3)**, teardown invariant + tests (§3.2), `_call_session_configs` TTL (§4.3), shutdown drain (§3.3) | No — `on_conversation_ended` keeps its meaning; `on_call_ended` is additive |
| **P1** | Messaging session derivation (§2.1), address-mode send (§2.2, §2.5), reconcile kept unconditional (§2.3), `CONVERSATION_UPDATED` rework (§2.6), single shared `list_participants` per message (§2.2), drop `"once"` from messaging (§2.7), `profile_id` off the participant list (§5) | `get_agent_address` signature (§2.4); messaging `memory_mode="once"` removed (§2.7) |
| **P2** | Instance affinity URLs (§4.2), Media Streams config via custom parameters (§4.3) | No |
| **P3** | Docs: replace the horizontal-scaling caveat in `CLAUDE.md` and the published reference with a deployment guide; TypeScript SDK parity | No |

P0 is independently valuable and ships first — it fixes a real leak on single-instance deployments too, and it is now non-breaking.

No store phase. §5 removed it.

---

## 9. Testing

- **Mode-3 send, against a live account (gates P1).** Send with `from` = `{address, channel}` on a conversation where no participant owns that address. Assert (a) delivery succeeds and (b) the conversation afterward contains a correctly-typed `AI_AGENT` participant at that address. Run it on both a v2-native and a v1-bridge conversation — see the open question in §2.2.
- **Mode-3 send on CHAT (gates P1).** Chat's `from` is an identity string, not a phone number, and `channelId` rides in `channelSettings` rather than the address. Confirm CO accepts address-mode resolution for `CHAT` before assuming §2.5 is channel-agnostic.
- **Reply recipient when author ≠ resolved customer (§7.4).** Inbound authored by a participant that isn't the reconciled `CUSTOMER`; assert the reply still goes to the customer.
- **Voice CLOSED after teardown (§7.3).** Free the session at WebSocket disconnect, then deliver CO CLOSED; assert `on_conversation_ended` still fires. This is the regression §7.3 exists to prevent.
- **What does v2-native capture actually create? (settles §2.3.)** Send an inbound SMS to a TAC number on a v2-native account with no v1-bridge, then `GET /Participants` before TAC touches anything. Record the type assigned to TAC's own address. If it is `UNKNOWN`, the "v1-bridge only" framing in `_reconcile_participants`' docstring and PR #19 is wrong and the docstring should be corrected — reconcile is load-bearing for every account. Cheap to run, and it removes a standing unknown from the code's own documentation either way.
- **Two-instance simulation.** Construct two channel instances over the same config — no shared anything, which is the point. Deliver `COMMUNICATION_CREATED` to A and `CONVERSATION_UPDATED`/CLOSED to B; assert `on_conversation_ended` fires on B with a correctly rebuilt session, despite B having never seen the conversation.
- **Statelessness assertion.** After any messaging webhook completes, assert `channel._conversations` is empty. This is the regression guard that keeps messaging stateless as the code evolves.
- **Call-count regression.** Assert the exact HTTP call count per inbound message for each `memory_mode`, pinned to the §6 budget table (2 for `"never"`, 4 for `"always"`, 3 with `fetch_profile_traits=False`). This is what stops the budget silently drifting — it is the easiest thing in this plan to regress by accident.
- **Hook split (voice).** Disconnect the socket → `on_call_ended` fires with the transcript intact and all local state freed; `on_conversation_ended` has *not* fired yet in orchestrated mode. Then deliver CO CLOSED to a second instance → `on_conversation_ended` fires there.
- **Leak loop.** N connect/disconnect cycles across all the failure modes in §3.2; assert `assert_no_residual_state` after each and that `len(channel._conversations) == 0` at the end.
- **Drain.** Live socket + shutdown → `on_call_ended` fires, no residual state.

## 10. Docs to update

- `CLAUDE.md` — the "Horizontal scaling limitation" bullet becomes a description of the model, not a caveat.
- Published reference (`https://twilio.github.io/twilio-agent-connect-python/`) — a deployment guide covering the messaging stateless model, voice affinity, the drain requirement, and the explicit statement that TAC needs **no shared datastore**.
- `on_call_ended` vs `on_conversation_ended` vs `on_call_status` — a short table in the voice docs; three end-of-call-ish hooks is confusing without one.
- Handler idempotency contract — document that `on_message_ready` may be called twice for the same message in a multi-instance deployment (§5).
- Changelog — `get_agent_address` signature, messaging `memory_mode="once"` removal, new `on_call_ended`.
- `mkdocs.yml` `filters` — update for any newly public or newly internal methods.

---

## 11. Gaps found in review (2026-09-30)

Found reviewing this plan and its implementation against current `main`, after rebasing onto `5677eeb` (v2.5.0). **Documented only. None of these are fixed yet.** Line references are to the rebased branch.

The mode split from §3.1 matters here. Gaps 1, 4 and 7 apply **only to ConversationRelay in orchestrated mode**, the one combination where a CO `CLOSED` webhook ends the conversation and can land on another instance. In relay-only mode and in every Media Streams provider, teardown is the only end event.

### Correctness bugs

**G1. CLOSED during a live call removes the live session.**
`_handle_conversation_closed` (`voice/channel.py:475`) pops the session whether or not a WebSocket is still attached. If CO closes the conversation mid-call (a `statusTimeouts` expiry, or the app calling `update_conversation(..., "CLOSED")`), and the CLOSED webhook lands on the instance holding the call:

- `on_conversation_ended` fires with the live session;
- teardown later finds no session, so `_release_session` (`voice/channel.py:425`) returns early and **`on_call_ended` never fires**, which breaks its "always fires" contract;
- the call's remaining prompts hit "unknown conversation" and are dropped.

*Fix direction:* when a socket is still attached, CLOSED fires `on_conversation_ended` from a copy and leaves the session in place. Teardown stays the only thing that frees it.

**G2. Media Streams per-call `session_config` survives only with instance affinity.**
§4.3 says the config travels on the TwiML. The code sends only a **token** in `<Stream>` custom parameters; the config stays in an instance-local `ExpiringDict` (`media_streams/shared/openai_provider.py:74`). Sending a token is right, because each `<Parameter>` name+value must be under 500 chars. But without `instance_public_domain`, an inbound call whose WebSocket lands on another replica finds no entry and silently falls back to `default_session_config`. The same applies to outbound calls placed on one replica and streamed to another.

*Fix direction:* document the dependency on affinity, and/or rebuild the config when the stream connects (re-run `on_inbound_call_session_config` from the `start` message's CallSid + custom parameters) when the token misses locally.

### Coverage gaps

**G3. Telemetry from #127 is only partly carried over to the new lifecycle.** *(Introduced by the rebase, not in the original branch.)*

- **Voice:** "Conversation Started" moved to `VoiceChannel._start_conversation`, and "Conversation Ended" to `_release_session`. In orchestrated ConversationRelay mode, *Ended* now measures **call** duration at teardown, not time-to-CLOSED. That's a meaning change for the metric; previously it rarely fired at all with multiple instances.
- **Messaging emits neither "Conversation Started" nor "Conversation Ended" any more.** There's no session lifecycle to hang them on. *Ended* could be emitted on CLOSED, with duration from the webhook's `createdAt`/`updatedAt`. That needs care: the stateless CLOSED path fast-exits when no `on_conversation_ended` is registered, and the channel filter costs a `list_participants`. *Started* has no stateless equivalent for inbound conversations.
- The `TestMessagingCallSites` lifecycle tests moved to `TestVoiceSessionLifecycle` in `tests/test_analytics.py`.

**G4. No call-event fallback without per-pod addressability.**
Without `instance_public_domain`, status/AMD/recording events land on any replica, and `get_conversation_session_by_call_sid` (`voice/channel.py:577`) returns `None`. In orchestrated mode, `list_conversations(channel_id=call_sid)` plus the existing `_rebuild_session` could give an identity-only session on any replica.

**G5. App-initiated `send_response` to a live call on another instance.**
Affinity routes Twilio's callbacks, not the app's own calls (a background job, another handler). No Twilio API writes into a ConversationRelay socket, so this can't be fixed inside TAC.

*Action:* document it as a limitation in `docs/deployment.md`: send from inside the call's own callbacks, or route by `conversation_id` at the balancer.

**G6. GPT-Live keeps a now-redundant expiry mechanism.** *(Introduced by the rebase.)*
GPT-Live (#125) postdates this plan. The rebase moved the token handoff and `ExpiringDict` into the shared Media Streams base, so GPT-Live gets both. Its own `_pending_token_expiries` asyncio timers (`gpt_live/provider.py:87`) now duplicate the `ExpiringDict` TTL. That's harmless but redundant; remove it in a follow-up.

### Behavior changes not called out as breaking

**G7. Orchestrated voice `on_conversation_ended` now receives a rebuilt, identity-only session.**
`_rebuild_session` (`voice/channel.py:492`) restores only the ids, profile and participants. Apps that read `metadata`, the transcript or `pending_handoff_data` there must move to `on_call_ended`. The commit calls this non-breaking; it needs a changelog and migration note.

**G8. Messaging `session.metadata` no longer carries across turns, even on one instance.**
Anything an app writes in turn N is gone in turn N+1. The outbound `direction: outbound` metadata (`messaging.py:648`) lives only on the session that `initiate_outbound_conversation` returns. CO conversations have no attributes field to keep it in.

**G9. Messaging `memory_mode="once"` raises at construction.**
`messaging.py:109`. **Decided:** don't raise. Log a `DeprecationWarning` and run as `"always"`, which gives *better* relevance at +1 `/Recall` per message, so nothing silently gets worse. Remove it in the next major version. Voice `"once"` is unchanged. Update `CLAUDE.md`'s memory-modes bullet and §2.7 to match.

### Unverified assumptions (block the messaging half)

**V1. Address-mode Actions send** (§2.2, §9): `from` = `{address, channel}` with no participant id. Untested on a live account, for both SMS and CHAT. Does delivery succeed, and does the conversation end up with a correctly typed `AI_AGENT` participant?

**V2. v2-native participant typing** (§2.3, §9): does CO capture assign `UNKNOWN` to TAC's own address on a v2-native account? That decides whether reconciliation is really v1-bridge-only.

### Proposed sequencing

| Step | Scope | Breaking? |
|---|---|---|
| 1 | G1, G6, G2 (docs + rebuild-on-connect), G4 | No |
| 2 | G3 (voice semantics note + decide messaging lifecycle telemetry), G5 + G7 docs, changelog | No |
| 3 | Run V1/V2 on live accounts | — |
| 4 | Messaging half: G9 (deprecate instead of raise), G8 changelog/migration note | Yes (`get_agent_address`, `send_response` first param) |
