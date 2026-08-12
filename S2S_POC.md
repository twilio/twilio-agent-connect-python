# TAC + Speech-to-Speech (S2S) POC

This document tracks the exploration of adding native speech-to-speech (S2S)
voice support to TAC, alongside the existing ConversationRelay-based
`VoiceChannel`. TAC's existing voice channel is cascaded (Twilio does
STT/TTS, TAC sees text turns); S2S providers instead listen to and produce
audio directly, with no intermediate transcription step.

Planned sections:

1. **OpenAI Realtime API via Twilio SIP Trunking** (this section) — Twilio
   forwards the call to OpenAI at the SIP level; audio never touches our
   server.
2. *(planned)* OpenAI Realtime API via Twilio Media Streams — audio bridged
   through our own server instead of a SIP hand-off.
3. *(planned)* TBD — a third S2S approach/provider.

---

## Section 1: OpenAI Realtime API + Twilio SIP Trunking

Implemented as `OpenAIRealtimeSipChannel` (`src/tac/channels/openai_realtime_sip/`).

### 1. Architecture

Twilio's Elastic SIP Trunking forwards the call directly to OpenAI at the SIP
level. Our server is never in the audio path — it only handles the
`realtime.call.incoming` webhook (to decide whether/how to accept the call)
and, optionally, a JSON-only "sideband" WebSocket for transcript capture and
tool calling.

At a component level:

```mermaid
flowchart LR
    Caller(["Caller"])
    Twilio["Twilio<br/>Elastic SIP Trunk"]
    OpenAI["OpenAI<br/>Realtime API"]
    TAC["TAC Server<br/>(webhook + control WS)"]

    Caller <-->|"phone call"| Twilio
    Twilio <-->|"SIP / SRTP audio"| OpenAI
    OpenAI -->|"webhook:<br/>realtime.call.incoming"| TAC
    TAC -->|"REST: /accept"| OpenAI
    TAC <-.->|"control WebSocket<br/>(JSON only, no audio)"| OpenAI

    linkStyle 1 stroke:#2563eb,stroke-width:2px
```

And the corresponding call flow over time:

```mermaid
sequenceDiagram
    participant Caller
    participant Twilio as Twilio<br/>(Elastic SIP Trunk)
    participant OpenAI as OpenAI<br/>Realtime API
    participant TAC as TAC Server<br/>(our webhook)

    Caller->>Twilio: Dials phone number
    Twilio->>OpenAI: SIP INVITE (Origination URI)
    OpenAI->>TAC: POST webhook (realtime.call.incoming)
    TAC->>TAC: on_call_incoming callback<br/>decides model/instructions/voice/tools
    TAC->>OpenAI: POST /accept
    OpenAI-->>Twilio: SIP 200 OK

    rect rgb(240, 240, 240)
    note over Caller, OpenAI: Audio flows directly over SIP/SRTP.<br/>Never touches the TAC server.
    Caller->>OpenAI: Speech (audio)
    OpenAI->>Caller: Speech (audio)
    end

    opt tools registered or transcription enabled
        TAC->>OpenAI: Open control WebSocket (call_id)
        loop for the life of the call
            OpenAI-->>TAC: transcript deltas
            OpenAI-->>TAC: function_call event
            TAC->>TAC: execute TACTool
            TAC->>OpenAI: function_call_output + response.create
        end
    end

    OpenAI--xTwilio: Call ends
    OpenAI--xTAC: Control WebSocket closes
    TAC->>TAC: on_conversation_ended(transcript)
```

Key point: **the audio path and the control path are two entirely separate
connections.** Media (SIP/SRTP) is negotiated and carried directly between
Twilio and OpenAI's infrastructure. The control WebSocket, when used, carries
only JSON events — no audio bytes ever cross it.

### 2. Setup

#### OpenAI side

1. `platform.openai.com/settings` → **Webhooks** → create a webhook:
   - Event type: `realtime.call.incoming`
   - URL: your public server + webhook path (default `/openai/incoming-call`)
   - Save the **webhook secret** — needed to verify incoming webhooks.
2. `platform.openai.com/settings` → **General** → note your **Project ID**
   (`proj_xxxxx`).

#### Twilio side

1. Buy a Voice-capable phone number.
2. **Elastic SIP Trunking** → create a trunk.
3. On the trunk, add an **Origination URI**:
   ```
   sip:proj_xxxxx@sip.api.openai.com;transport=tls
   ```
   (EU data residency: `sip-eu.api.openai.com`)
4. Enable **Secure Trunking** (`secure: true`) on the trunk.
5. Attach the phone number to the trunk.

#### Environment variables

```
TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_API_KEY / TWILIO_API_SECRET
TWILIO_PHONE_NUMBER      # TAC requires it even though this channel doesn't use it directly
OPENAI_API_KEY
OPENAI_WEBHOOK_SECRET
```

`TWILIO_CONVERSATION_CONFIGURATION_ID` should be **unset** — this channel
doesn't use Conversation Orchestrator or Memory.

### 3. Pros and cons

**Pros**

- Audio never touches our server — no binary frame handling, no
  audio-streaming infrastructure to build or run.
- Fits TAC's existing webhook → callback → decision lifecycle almost
  exactly, unlike a media bridge which needs a mostly-new lifecycle.
- True S2S: no cascaded STT → LLM → TTS latency chain.

**Cons**

- No access to the audio itself: no custom recording, no custom VAD, no
  injecting or mixing external audio.
- Debugging failures at the SIP layer is opaque: errors surface as a bare
  SIP response code (e.g. `400 Bad Request`) with no application-level
  detail, and there's no dashboard purpose-built for this integration on
  either side.
- Tightly coupled to OpenAI's specific Realtime-over-SIP feature and
  Twilio's Elastic SIP Trunking semantics; a Media Streams-based bridge
  (Section 2) is a more portable pattern if a second S2S provider is
  added later.
