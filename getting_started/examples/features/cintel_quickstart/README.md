# Conversation Intelligence Quickstart

Practice a real support call against an AI-powered customer while Twilio Conversation Intelligence scores your script adherence live.

## What This Demo Does

You play a **support agent** on a real PSTN call. The other side is a GPT-5.4-mini AI customer (a frustrated Owl Internet subscriber with a connectivity problem) that coaches you through a support script without breaking character. A browser dashboard shows:

- **Live Transcript** — agent and customer turns as they happen
- **Script Adherence** — checkpoints update in real time after each agent utterance
- **Post-call Summary** — generated automatically when the call ends

## Prerequisites

- Python 3.10+, [uv](https://docs.astral.sh/uv/getting-started/installation/), [ngrok](https://ngrok.com/)
- Twilio account with a phone number
- OpenAI API key

## Quick Start

### Step 1 — Start ngrok

```bash
ngrok http 3340
```

Note the domain (e.g. `abc123.ngrok-free.app`) — you'll enter it in the setup wizard.

### Step 2 — Run the Setup Wizard

```bash
uv run getting_started/examples/features/cintel_quickstart/setup_server.py
```

Open **http://localhost:8081** and complete the form. The wizard will:

1. Create (or reuse) a Memory Store
2. Create a Conversation Orchestrator configuration
3. Create an Intelligence Configuration with Script Adherence + Summary operators
4. Link the Intelligence Configuration to the Conversation Orchestrator
5. Configure your Twilio number's Voice webhook to point at your ngrok domain

Copy the env var output at the end.

### Step 3 — Configure the App

```bash
cp getting_started/examples/features/cintel_quickstart/.env.example \
   getting_started/examples/features/cintel_quickstart/.env
```

Fill in `.env` with your credentials and the values from the wizard. See [.env.example](.env.example) for all fields.

### Step 4 — Run the App

```bash
uv run getting_started/examples/features/cintel_quickstart/app.py
```

Open **http://localhost:3340**.

### Step 5 — Make a Call

Call your Twilio number and follow the script:

| Checkpoint | What to say |
|---|---|
| Greeting | "Hi, this is [your name] calling from Owl Internet" |
| Identity Verification | "Can I get your account number?" |
| Resolution Steps | "Let me trigger a remote modem reset" |
| Closing | "Thank you for choosing Owl Internet. Is there anything else?" |

Watch checkpoints update live. After the call ends, the summary appears automatically.

## Architecture

```
Phone call (PSTN)
  └── Twilio routes to https://<ngrok>/tac/twiml

FastAPI Server (app.py)
  ├── /                →  Dashboard (live transcript + script adherence)
  ├── /api/events      →  SSE stream to browser
  ├── /api/config      →  returns phone number for banner
  ├── /api/script      →  fetches script hints from Intelligence Config
  ├── /ci-webhook      →  receives CI operator results (script adherence + summary)
  └── /tac/*           →  TAC VoiceChannel (TwiML + WebSocket)

TAC VoiceChannel
  └── ConversationRelay ↔ WebSocket ↔ TAC ↔ handle_message_ready()
                                               └── OpenAI GPT-5.4-mini (AI customer)

Twilio Conversation Intelligence
  ├── Script-Adherence operator  →  fires per utterance → /ci-webhook → SSE checkpoint-update
  └── Summary operator           →  fires on call end   → /ci-webhook → SSE summary-update
```

## Environment Variables

| Variable | Description | Source |
|---|---|---|
| `TWILIO_ACCOUNT_SID` | Twilio Account SID | Twilio Console |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token | Twilio Console |
| `TWILIO_API_KEY` | API Key SID | Twilio Console |
| `TWILIO_API_SECRET` | API Secret | Twilio Console |
| `TWILIO_PHONE_NUMBER` | Your Twilio number (E.164) | Twilio Console |
| `TWILIO_CONVERSATION_CONFIGURATION_ID` | CO configuration ID | Setup wizard |
| `CONVERSATION_INTELLIGENCE_CONFIGURATION_ID` | CI configuration ID | Setup wizard |
| `TWILIO_VOICE_PUBLIC_DOMAIN` | ngrok domain, no `https://` | ngrok |
| `OPENAI_API_KEY` | OpenAI API key | OpenAI Console |

## Customisation

**Change the AI customer personality** — edit `SCRIPT_COACH_CUSTOMER_PROMPT` in `app.py`.

**Change the script checkpoints** — update the Script Adherence operator's `script` parameter in the Twilio Console (Intelligence → your config), then update the `data-category` and `data-key` attributes in `static/index.html` to match.

**Change the model** — update the `model=` argument in `handle_message_ready()` in `app.py`.

## Troubleshooting

**Checkpoints not updating**
1. Check Twilio Console → Debugger for failed webhook deliveries to `/ci-webhook`
2. Verify ngrok is running and `TWILIO_VOICE_PUBLIC_DOMAIN` matches the current domain
3. Check the Intelligence Configuration is linked to your CO config (Twilio Console → Intelligence)

**AI customer not responding**
1. Verify `OPENAI_API_KEY` is valid
2. Check server logs (`/tmp/cintel-app.log` if started with log redirect)

**ngrok domain changed**
1. Re-run the setup wizard with the new domain — it will re-configure the voice webhook
2. Update `TWILIO_VOICE_PUBLIC_DOMAIN` in `.env`
3. Restart the app
