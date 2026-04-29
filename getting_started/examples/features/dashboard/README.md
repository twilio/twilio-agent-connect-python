# Demo UI

Observation dashboard for monitoring TAC conversations in real time.

> **Warning:** This dashboard has no authentication. It is intended for local
> development and demos only. Do not use in production.

## Quick Start

```bash
uv run python getting_started/examples/features/dashboard/app.py
```

Then open http://localhost:8000/dashboard

## Features

- **Active Sessions** — Live conversations with message viewer, profile memory, and agent context
- **History** — Browse closed conversations from Conversation Orchestrator
- **Agent Context** — Profile traits, observations, summaries from Conversation Memory
- **Events** — CI operator results (Sentiment, Summary) with Intelligence badges
- **Status indicators** — Green dot (ACTIVE), yellow dot (INACTIVE)
- **Resizable panels** — Drag handle between conversation and context panels
