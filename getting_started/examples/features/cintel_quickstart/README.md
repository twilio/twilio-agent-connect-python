# Conversation Intelligence Quickstart

Practice a real support call against an AI-powered customer while Twilio Conversation Intelligence scores your script adherence live.

## What This Demo Does

You play a **support agent** on a real PSTN call. The other side is a GPT-powered AI customer (a frustrated Owl Internet subscriber with a connectivity problem) that coaches you through a support script without breaking character. A browser dashboard shows:

- **Live Transcript** — agent and customer turns appear as they happen
- **Script Adherence** — checkpoints update in real time after each agent utterance
- **Post-call Summary** — generated automatically when the call ends

## Prerequisites

- Python 3.10+, [uv](https://docs.astral.sh/uv/getting-started/installation/), [ngrok](https://ngrok.com/)
- Twilio account with a phone number
- OpenAI API key

---

## Step-by-Step Setup

### Step 1 — Start ngrok

The app needs a public HTTPS URL so Twilio can reach it. Start ngrok tunneling to port 3340:

```bash
ngrok http 3340
```

Note the domain shown (e.g. `abc123.ngrok-free.app`) — **do not include `https://`**. You'll enter this in the setup wizard.

> If you have a paid ngrok account with a static domain, use `ngrok http --domain=your-domain.ngrok-free.app 3340`.

---

### Step 2 — Run the Setup Wizard

```bash
uv run getting_started/examples/features/cintel_quickstart/setup_server.py
```

Open **http://localhost:8081** in your browser. The wizard has four steps:

#### Step 1 — Credentials & ngrok Domain

You'll need the following from the [Twilio Console](https://console.twilio.com):

| Field | Where to find it |
|---|---|
| **API Key** | Console → API Keys & Tokens → Create API Key. Starts with `SK`. |
| **API Secret** | Shown once when you create the API Key — copy it immediately. |
| **Account SID** | Console → Dashboard. Starts with `AC`. |
| **Auth Token** | Console → Dashboard, next to Account SID. |
| **Twilio Phone Number** | Console → Phone Numbers → Manage → Active Numbers. E.164 format e.g. `+18005551234`. |
| **OpenAI API Key** | [platform.openai.com/api-keys](https://platform.openai.com/api-keys). Starts with `sk-`. |
| **ngrok Domain** | From Step 1 above. No `https://` prefix. |

#### Step 2 — Memory Store

The wizard creates a **Memory Store** — this is where Twilio stores conversation memory between calls. Give it a display name and description (e.g. `owl-support-memory` / `Memory for Owl Internet support calls`).

Click **Create Memory Store** and wait for it to provision (this can take ~30 seconds).

#### Step 3 — Conversation Orchestrator

This creates a **Conversation Orchestrator configuration** that manages the lifecycle of each call. Enter:

- **Display name** — URL-safe identifier, max 32 characters, e.g. `owl-support`
- **Description** — optional, e.g. `Owl Internet support demo`

Click **Create Configuration**. The wizard automatically sets up:
- VOICE channel with no capture rules (closes on hangup)
- Status callback pointing to your ngrok webhook URL

#### Step 4 — Intelligence & Voice Webhook

This step does two things:

1. **Creates an Intelligence Configuration** with two operators:
   - *Script Adherence* — evaluates each agent utterance against the support script
   - *Summary* — generates a call summary when the conversation ends

2. **Configures your Twilio phone number's Voice webhook** to point at `https://<ngrok>/twiml`

Click **Create Intelligence Config & Configure Webhook**.

#### Copy the .env Output

At the end of the wizard, a `.env` snippet is shown. Copy it — you'll need it in the next step.

---

### Step 3 — Configure the App

Copy the example env file:

```bash
cp getting_started/examples/features/cintel_quickstart/.env.example \
   getting_started/examples/features/cintel_quickstart/.env
```

Open `.env` and fill in all values. You can paste the snippet from the wizard directly — it contains the generated resource IDs. Add any missing values (credentials, phone number, ngrok domain) manually.

See [.env.example](.env.example) for the full list of required variables.

---

### Step 4 — Run the App

```bash
uv run getting_started/examples/features/cintel_quickstart/app.py
```

Open **http://localhost:3340**. You should see the dashboard with your phone number in the banner.

---

### Step 5 — Make a Call

Call your Twilio number. The AI customer will pick up. Follow this script as the support agent:

| # | Checkpoint | What to say |
|---|---|---|
| 1 | **Greeting** | "Hi, this is [your name] calling from Owl Internet" |
| 2 | **Identity Verification** | "Can I get your account number to pull up your information?" |
| 3 | **Resolution Steps** | "Let me trigger a remote modem reset for you" |
| 4 | **Closing** | "Thank you for choosing Owl Internet. Is there anything else I can help you with?" |

**What to expect:**
- Checkpoints turn green (✓) as you complete each one, or red (✗) if skipped
- The transcript shows both sides of the conversation in near real time
- After you hang up, the Summary panel appears with an AI-generated call summary

Use the **Reset** button in the header to clear the dashboard between calls.

---

## How It Works

1. You call your Twilio number — Twilio routes the call to `https://<ngrok>/twiml`
2. TAC's VoiceChannel connects via ConversationRelay WebSocket and starts a conversation in Conversation Orchestrator
3. Deepgram transcribes your speech; each utterance triggers `handle_message_ready()` in the app
4. The agent turn is immediately added to the dashboard transcript
5. The utterance is sent to OpenAI GPT which plays the role of the AI customer and responds
6. The customer response is sent back over the WebSocket so Twilio speaks it to you
7. In parallel, Twilio Conversation Intelligence evaluates the utterance against the script and POSTs the result to `/ci-webhook`
8. The dashboard polls `/api/transcript` every 200ms and updates checkpoints as CI results arrive
9. When you hang up, the Summary operator fires and POSTs a call summary to `/ci-webhook` — the summary panel appears on the dashboard

## Architecture

```
Phone call (PSTN)
  └── Twilio routes to https://<ngrok>/twiml

FastAPI Server (app.py)
  ├── /                 →  Dashboard (live transcript + script adherence)
  ├── /api/transcript   →  Polling endpoint (browser polls every 200ms)
  ├── /api/reset        →  Clears dashboard state
  ├── /api/config       →  Returns phone number for the banner
  ├── /api/script       →  Fetches script hints from Intelligence Config
  ├── /ci-webhook       →  Receives CI operator results (script adherence + summary)
  └── /twiml, /ws       →  TAC VoiceChannel (TwiML + WebSocket)

TAC VoiceChannel
  └── ConversationRelay ↔ WebSocket ↔ TAC ↔ handle_message_ready()
                                               └── OpenAI GPT (AI customer)

Twilio Conversation Intelligence
  ├── Script-Adherence operator  →  fires per utterance  → /ci-webhook → dashboard checkpoints
  └── Summary operator           →  fires on call end    → /ci-webhook → dashboard summary panel
```

---

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

---

## Customisation

**Change the AI customer personality** — edit `SCRIPT_COACH_CUSTOMER_PROMPT` in `app.py`.

**Change the script checkpoints** — update the Script Adherence operator's `script` parameter in the Twilio Console (Intelligence → your config), then update the `data-category` and `data-key` attributes in `static/index.html` to match.

**Change the model** — update the `model=` argument in `handle_message_ready()` in `app.py`.

---

## Troubleshooting

**Checkpoints not updating**
1. Check the server logs for `Received CINTEL webhook` — if absent, CI webhooks are not arriving
2. Verify your Twilio number's Voice webhook is set to `https://<ngrok>/twiml`
3. Check the Intelligence Configuration is linked to your CO config (Twilio Console → Intelligence)
4. Confirm ngrok is tunneling to port 3340 and `TWILIO_VOICE_PUBLIC_DOMAIN` matches the current domain

**Conversation using the wrong config**
Each call creates a conversation under whichever CO config was active when the call connected. If you recently changed `TWILIO_CONVERSATION_CONFIGURATION_ID`, restart the app. If an old `ACTIVE` conversation still exists for your number, close it via the Twilio Console or API before calling again.

**AI customer not responding**
1. Verify `OPENAI_API_KEY` is valid
2. Check server logs for OpenAI error messages

**ngrok domain changed**
1. Re-run the setup wizard with the new domain — it will re-configure the voice webhook and CI webhook URLs
2. Update `TWILIO_VOICE_PUBLIC_DOMAIN` in `.env`
3. Restart the app
