"""
Conversation Intelligence Setup Wizard

Creates the v3 Intelligence Configuration with two rules:
  1. Script Adherence — fires on every caller utterance (COMMUNICATION trigger)
  2. Summary — fires when the call ends (CONVERSATION_END trigger)

Then links the configuration to your existing Conversation Orchestrator configuration
and prints the env vars you need to add to your .env file.

Usage:
    uv run getting_started/examples/features/cintel_quickstart/setup_ci.py

Prerequisites:
  - TWILIO_API_KEY, TWILIO_API_SECRET in env (or .env)
  - TWILIO_CONVERSATION_CONFIGURATION_ID in env (from the main setup wizard)
  - ngrok running and TWILIO_VOICE_PUBLIC_DOMAIN set (so webhook URL is known)
"""

import base64
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

INTELLIGENCE_BASE = "https://intelligence.twilio.com/v3"
CONVERSATIONS_BASE = "https://conversations.twilio.com/v2/ControlPlane"

# Twilio-authored operator IDs (stable across accounts)
SCRIPT_ADHERENCE_OPERATOR_ID = "intelligence_operator_01kf34tcyefpyb1t4m0nbd8rxg"
SUMMARY_OPERATOR_ID = "intelligence_operator_01kcv35pnkeysaf6z6cqtbpegn"


def _auth(api_key: str, api_secret: str) -> str:
    return "Basic " + base64.b64encode(f"{api_key}:{api_secret}".encode()).decode()


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: {name} is not set. Add it to your .env file.", file=sys.stderr)
        sys.exit(1)
    return val


def _prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{label}{suffix}: ").strip()
    return val or default


def main() -> None:
    print("=" * 60)
    print("  Conversation Intelligence Setup Wizard")
    print("=" * 60)
    print()

    api_key = _require_env("TWILIO_API_KEY")
    api_secret = _require_env("TWILIO_API_SECRET")
    conv_config_id = _require_env("TWILIO_CONVERSATION_CONFIGURATION_ID")

    ngrok_domain = os.environ.get("TWILIO_VOICE_PUBLIC_DOMAIN", "")
    if not ngrok_domain:
        ngrok_domain = _prompt("Enter your ngrok domain (e.g. abc123.ngrok.app)")
    if not ngrok_domain:
        print("ERROR: ngrok domain is required.", file=sys.stderr)
        sys.exit(1)

    webhook_url = f"https://{ngrok_domain}/ci-webhook"
    print(f"\nCI webhook URL: {webhook_url}")

    headers = {
        "Authorization": _auth(api_key, api_secret),
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=30) as client:
        # ── Step 1: Create Intelligence Configuration ────────────────────────
        print("\n[1/3] Creating Intelligence Configuration...")

        ci_payload = {
            "displayName": "Owl Internet Support",
            "description": "Script adherence + post-call summary for voice support",
            "rules": [
                {
                    "operators": [{"id": SCRIPT_ADHERENCE_OPERATOR_ID}],
                    "triggers": [{"on": "COMMUNICATION"}],
                    "actions": [{"type": "WEBHOOK", "method": "POST", "url": webhook_url}],
                },
                {
                    "operators": [{"id": SUMMARY_OPERATOR_ID}],
                    "triggers": [{"on": "CONVERSATION_END"}],
                    "actions": [{"type": "WEBHOOK", "method": "POST", "url": webhook_url}],
                },
            ],
        }

        resp = client.post(
            f"{INTELLIGENCE_BASE}/ControlPlane/Configurations",
            headers=headers,
            json=ci_payload,
        )

        if resp.status_code not in (200, 201):
            print(f"ERROR: Failed to create Intelligence Configuration ({resp.status_code}):")
            print(resp.text)
            sys.exit(1)

        ci_config = resp.json()
        ci_config_id = ci_config["id"]
        print(f"  ✓ Created: {ci_config_id}")

        # ── Step 2: Link to Conversation Orchestrator configuration ──────────
        print("\n[2/3] Linking to Conversation Orchestrator configuration...")

        # First GET current config to preserve existing fields
        get_resp = client.get(
            f"{CONVERSATIONS_BASE}/Configurations/{conv_config_id}",
            headers=headers,
        )

        if get_resp.status_code != 200:
            print(
                f"  WARNING: Could not fetch existing Conversation Configuration "
                f"({get_resp.status_code}). Skipping link step.\n"
                f"  You may need to manually add the Intelligence Configuration ID "
                f"to your Conversation Configuration."
            )
        else:
            existing = get_resp.json()
            existing_ci_ids: list[str] = existing.get("intelligenceConfigurationIds") or []

            if ci_config_id not in existing_ci_ids:
                new_ci_ids = existing_ci_ids + [ci_config_id]

                # PUT preserves all rules; we only update intelligenceConfigurationIds
                update_payload = {
                    "displayName": existing.get("displayName", ""),
                    "intelligenceConfigurationIds": new_ci_ids,
                }

                put_resp = client.put(
                    f"{CONVERSATIONS_BASE}/Configurations/{conv_config_id}",
                    headers=headers,
                    json=update_payload,
                )

                if put_resp.status_code in (200, 201):
                    print(f"  ✓ Linked {ci_config_id} → {conv_config_id}")
                else:
                    print(
                        f"  WARNING: Failed to link ({put_resp.status_code}): {put_resp.text}\n"
                        f'  Manually add intelligenceConfigurationIds: ["{ci_config_id}"] '
                        f"to your Conversation Configuration."
                    )
            else:
                print("  ✓ Already linked")

        # ── Step 3: Print env vars ────────────────────────────────────────────
        print("\n[3/3] Done! Add these to your .env file:\n")

        env_block = f"CONVERSATION_INTELLIGENCE_CONFIGURATION_ID={ci_config_id}\n"
        print(env_block)

        # Offer to write directly to .env
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        env_path = os.path.normpath(env_path)

        if os.path.exists(env_path):
            answer = input(f"Append to {env_path}? [y/N] ").strip().lower()
            if answer == "y":
                with open(env_path, "a") as f:
                    f.write("\n# Conversation Intelligence (cintel_quickstart)\n")
                    f.write(env_block)
                print(f"  ✓ Written to {env_path}")
        else:
            print(f"  (No .env found at {env_path} — add the lines manually)")

        print()
        print("Next step: run the quickstart app:")
        print("  uv run getting_started/examples/features/cintel_quickstart/app.py")
        print()


if __name__ == "__main__":
    main()
