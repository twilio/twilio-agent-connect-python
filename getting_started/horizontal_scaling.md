# Run TAC on Multiple Replicas

This guide walks you through running a TAC application as several replicas behind a load balancer. TAC needs no shared datastore for this: no Redis and no database.

**Running a single instance?** You can skip this guide. Everything here only matters once a second replica can receive traffic.

## Step 1: Put the Replicas Behind a Load Balancer

Point Twilio's webhooks and your voice domain at an ordinary load balancer. No sticky sessions are needed:

- **Messaging (SMS, RCS, WhatsApp, Chat):** any replica can handle any webhook. Each message is handled on its own, from the webhook and one Conversation Orchestrator lookup.
- **Voice:** any replica can accept a call's WebSocket. The call then stays on that replica until it ends.

## Step 2: Make `on_message_ready` Idempotent

TAC drops duplicate webhook deliveries using Twilio's idempotency token, but it remembers them per replica. A retry that lands on a different replica is processed again.

> [!WARNING]
> With more than one replica, `on_message_ready` may run twice for the same message. At worst that's one extra LLM call and one duplicate reply.

Write handlers that tolerate this. If you can't, route webhooks by conversation id at the load balancer.

## Step 3: Keep State Between Turns Yourself

A messaging session lasts one turn: from the webhook until `on_message_ready` returns. Anything you put on `session.metadata` is gone by the next message, on every replica. To keep state across turns, store it in your own store, keyed by `session.conversation_id`.

Metadata you pass to `initiate_outbound_conversation` is the exception: TAC stores it on the conversation in Conversation Orchestrator, so any replica can read it back:

```python
async def on_message(message: str, session: ConversationSession, memory) -> str:
    metadata = await session.conversation_metadata()
    appointment_id = metadata.get("appointment_id")
    ...
```

Conversation Orchestrator allows at most 8 keys, each up to 128 letters, digits, `.`, `_` or `-`, with string values up to 512 characters. Entries outside those limits are only on the session `initiate_outbound_conversation` returns, and TAC logs a warning.

If TAC can't read a conversation's participants, it drops that message and calls your `on_error` handler with `dropped_inbound` set:

```python
def handle_error(error: Exception, context: dict) -> None:
    if context.get("dropped_inbound"):
        alert(f"Dropped inbound on {context.get('conversation_id')}: {error}")

tac.on_error(handle_error)
```

## Step 4: Read Call State in `on_call_ended`

`on_conversation_ended` fires on whichever replica receives Conversation Orchestrator's CLOSED webhook. That may not be the one that handled the conversation, so the session it gets is rebuilt from Conversation Orchestrator. It has the conversation id, profile and participants (and `call_sid` on voice), but none of your `session.metadata`.

For voice, use the hook that matches what you need:

| Hook | Fires | Gets |
|---|---|---|
| `VoiceChannel.on_call_ended` | when the call's WebSocket closes, on the replica that held the call, every time | the live session: `call_sid`, your `metadata`, and the transcript on Media Streams |
| `TAC.on_conversation_ended` | when the conversation closes, on any replica | a rebuilt session (identity only) |
| `VoiceChannel.on_call_status` | on Twilio's call status callback, on any replica | a `CallStatusEvent`, no session |

Save transcripts and per-call state in `on_call_ended`:

```python
async def on_call_ended(session: ConversationSession) -> None:
    await archive(session.call_sid, session.metadata.get("transcript", []))

voice_channel.on_call_ended(on_call_ended)
```

`end_call()` behaves like any hangup: `on_call_ended` fires right away, and `on_conversation_ended` fires once, when Conversation Orchestrator closes the conversation.

## Step 5: Give Each Voice Replica Its Own Hostname

Some of a call's webhooks, such as call status, answering machine detection, recording and the ConversationRelay `<Connect action>` callback, carry only a `CallSid`. Through the load balancer they reach a random replica, not the one holding the call. To send them to the right replica, set `TWILIO_INSTANCE_PUBLIC_DOMAIN` on each replica to that replica's own public hostname:

```bash
TWILIO_VOICE_PUBLIC_DOMAIN=voice.example.com          # the load balancer
TWILIO_INSTANCE_PUBLIC_DOMAIN=pod-1.voice.example.com  # this replica, directly
```

Or in code:

```python
TACConfig(
    ...,
    voice_public_domain="voice.example.com",
    instance_public_domain=os.environ["POD_ADDRESS"],
)
```

TAC then points the call's WebSocket and `<Connect action>` URLs at this replica, and the status, AMD and recording callbacks too on calls TAC places. Inbound calls take those callbacks from the phone number's own configuration, so they still go to the load balancer.

**When you need it:**

- **Required:** OpenAI Realtime or GPT-Live (Media Streams) with a per-call session config, from `on_inbound_call_session_config` or `session_config` on `initiate_outbound_conversation`. The config is held by the replica that served the call's TwiML or placed the call. If the stream connects elsewhere, the call falls back to `default_session_config` and TAC logs a warning. With only `default_session_config`, you don't need it.
- **Needed:** speaking on a live call (`send_response` on ConversationRelay) or reading its live transcript from outside the call, for example from a call status handler. Doing it from the call's own callbacks (`on_message_ready`, tools) works without it, because those always run on the right replica.
- **Not needed:** everything else, including lifecycle callbacks, analytics and `CallSid` lookups.

**What it takes:** each replica must be reachable from Twilio over the public internet, at its own hostname with a valid TLS certificate. On Kubernetes, that means per-pod Ingress hostnames with a wildcard certificate. A headless Service isn't enough, because its DNS only resolves inside the cluster. Many managed platforms can't give replicas their own public address, for example Cloud Run, App Runner, Azure Container Apps, Heroku, or ECS behind an ALB.

**Without it:** leave `TWILIO_INSTANCE_PUBLIC_DOMAIN` unset, and either route those webhooks by `CallSid` at the load balancer or look the call up with `VoiceChannel.resolve_conversation_session_by_call_sid()`. On orchestrated ConversationRelay calls it returns a session rebuilt from Conversation Orchestrator when the call is on another replica. That's enough to match an event to its conversation and customer, but not to speak on the call.

## Step 6: Shut Down Without Dropping Calls

On SIGTERM, uvicorn (which `TACFastAPIServer` runs on) closes every open WebSocket. TAC tears each call down and fires `on_call_ended`, so nothing leaks, but the calls are cut off. To let calls finish during a redeploy or scale-in, have your platform hold off SIGTERM:

1. **Stop new calls reaching the replica** by failing its readiness probe or removing it from the load balancer.
2. **Wait before stopping the process**, long enough for the calls you want to keep, for example with a `preStop` hook.
3. **Give the platform room for that wait.** The termination grace period must cover the wait plus the time your `on_call_ended` handlers need.

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

On a server other than uvicorn, check that it closes WebSocket connections on shutdown. If the process exits with sockets still open, TAC can't tear those calls down and `on_call_ended` doesn't fire.

## Optional: Cut Conversation Orchestrator Calls

- **Reply with the session, not the id.** `channel.send_response(session, reply)` costs no extra request. Passing a bare conversation id makes TAC look the conversation's participants up again. On Chat, that only works when the participants carry a `channelId`, so pass the session there.
- **Skip profile traits you don't use.** With `memory_mode="always"`, TAC fetches profile traits on every message. If your prompts don't use `build_profile_prompt()`, turn that off:

  ```python
  TACConfig(
      ...,
      memory_config=TwilioMemoryConfig(fetch_profile_traits=False),
  )
  ```

## Checklist

- [ ] Replicas behind a load balancer; no sticky sessions needed
- [ ] `on_message_ready` tolerates running twice for one message
- [ ] State between turns kept in your own store, keyed by `conversation_id`
- [ ] Call state saved in `on_call_ended`, not `on_conversation_ended`
- [ ] Voice: `TWILIO_INSTANCE_PUBLIC_DOMAIN` set per replica if you use per-call Media Streams session configs, or call webhooks routed by `CallSid` otherwise
- [ ] Voice: readiness probe, `preStop` wait and grace period set so redeploys don't cut calls
