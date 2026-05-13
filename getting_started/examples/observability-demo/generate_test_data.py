"""Generate test telemetry data to verify metrics and traces

This script simulates TAC operations to populate Grafana and Langfuse with test data.
"""

import asyncio
import os
import random

# Setup telemetry BEFORE importing TAC
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"

from tac.telemetry import TACTelemetry

# Initialize telemetry
telemetry = TACTelemetry()
telemetry.setup_meter(enable_otlp_exporter=True)
telemetry.setup_tracer(enable_otlp_exporter=True)

from tac.telemetry.metrics import MetricsClient
from tac.telemetry.tracer import TracerClient

print("🚀 Generating test telemetry data...")
print("📊 Metrics → Prometheus → Grafana (http://localhost:3000)")
print("🔍 Traces → Langfuse (http://localhost:3001)")
print()

metrics = MetricsClient()
tracer = TracerClient()

CHANNELS = ["sms", "whatsapp", "voice"]
CLIENT_TYPES = ["conversation", "memory", "knowledge"]


async def simulate_message_processing(channel: str, conversation_id: str):
    """Simulate a complete message processing lifecycle"""

    # Generate sample input/output
    user_input = f"Hello from {channel} user in {conversation_id}"
    llm_output = f"Hi! I received your message on {channel}."

    # Generate a profile ID for the user (in real usage, this comes from Conversation Memory)
    profile_id = f"profile_{hash(conversation_id) % 100:03d}"

    with tracer.start_span(
        "message.processing",
        attributes={
            "channel": channel,
            "conversation_id": conversation_id,
            "input": user_input,
            "output": llm_output,
            "enduser.id": profile_id,  # User identification for filtering in Langfuse
        },
    ):
        # 1. Message received
        metrics.message_received_count.add(1, attributes={"channel": channel})

        # 2. Memory retrieval
        with tracer.start_span(
            "memory.retrieve", attributes={"channel": channel, "conversation_id": conversation_id}
        ):
            memory_duration = random.uniform(0.05, 0.3)
            await asyncio.sleep(memory_duration)

            # Simulate memory API call
            metrics.api_request_count.add(1, attributes={"client_type": "memory", "method": "POST"})
            metrics.api_request_duration.record(
                memory_duration, attributes={"client_type": "memory", "method": "POST"}
            )

        # 3. User callback (conversation.ready)
        with tracer.start_span(
            "conversation.ready",
            attributes={"channel": channel, "conversation_id": conversation_id},
        ):
            # This is the critical path - includes LLM call
            callback_duration = random.uniform(0.2, 1.5)
            await asyncio.sleep(callback_duration)
            metrics.conversation_ready_duration.record(
                callback_duration, attributes={"channel": channel}
            )

        # 4. Send message
        if random.random() < 0.95:  # 95% success rate
            with tracer.start_span(
                "message.send", attributes={"channel": channel, "conversation_id": conversation_id}
            ):
                send_duration = random.uniform(0.02, 0.1)
                await asyncio.sleep(send_duration)
                metrics.message_sent_count.add(1, attributes={"channel": channel})
        else:
            # 5% error rate
            metrics.message_error_count.add(
                1, attributes={"channel": channel, "error_type": "http_5xx"}
            )


async def simulate_api_requests():
    """Simulate API requests to various services"""
    client_type = random.choice(CLIENT_TYPES)
    method = random.choice(["GET", "POST"])

    duration = random.uniform(0.05, 0.5)
    await asyncio.sleep(duration)

    metrics.api_request_count.add(1, attributes={"client_type": client_type, "method": method})
    metrics.api_request_duration.record(
        duration, attributes={"client_type": client_type, "method": method}
    )

    # Simulate occasional errors
    if random.random() < 0.05:  # 5% error rate
        error_type = random.choice(["http_4xx", "http_5xx", "timeout"])
        metrics.api_error_count.add(
            1, attributes={"client_type": client_type, "error_type": error_type}
        )


async def main():
    """Generate test data"""
    # Simulate some conversation starts/ends (metrics only, no spans)
    for _ in range(5):
        channel = random.choice(CHANNELS)
        # Conversation start
        metrics.conversation_start_duration.record(
            random.uniform(0.01, 0.05), attributes={"channel": channel}
        )
        metrics.conversation_active_count.add(1)

    tasks = []

    # Generate 30 messages across different channels
    for _ in range(30):
        channel = random.choice(CHANNELS)
        conversation_id = f"conv_{random.randint(1, 10):03d}"  # 10 different conversations
        tasks.append(simulate_message_processing(channel, conversation_id))

        # Add some API requests
        if random.random() < 0.3:
            tasks.append(simulate_api_requests())

    # Run all simulations
    print(f"Simulating {len(tasks)} operations...")
    await asyncio.gather(*tasks)

    # Simulate some conversation ends (metrics only)
    for _ in range(3):
        channel = random.choice(CHANNELS)
        metrics.conversation_end_duration.record(
            random.uniform(0.005, 0.02), attributes={"channel": channel, "reason": "completed"}
        )
        metrics.conversation_active_count.add(-1)

    # Flush telemetry data
    print("\n⏳ Flushing telemetry data...")
    telemetry.shutdown()
    await asyncio.sleep(2)

    print("\n✅ Done! View results:")
    print("   📊 Grafana: http://localhost:3000 (Dashboards → TAC Observability)")
    print("   🔍 Langfuse: http://localhost:3001 (Traces → message.processing)")
    print()
    print("Note: You'll need to create a Langfuse account and set these env vars:")
    print("   export LANGFUSE_PUBLIC_KEY='pk-lf-...'")
    print("   export LANGFUSE_SECRET_KEY='sk-lf-...'")


if __name__ == "__main__":
    asyncio.run(main())
