# Media Streams Providers

`VoiceChannel` delegates the real-time media transport to a pluggable
`VoiceProvider`. The default one, `ConversationRelayProvider`, is documented
under [Channels](channels.md).

The providers below bridge Twilio
[Media Streams](https://www.twilio.com/docs/voice/twiml/stream)
(`<Connect><Stream>`) to an OpenAI speech-to-speech API, relaying audio between
Twilio's WebSocket and the model's. Both need the optional `websockets`
dependency:

```bash
pip install "tac[server,gpt-live]"         # GPT-Live
pip install "tac[server,openai-realtime]"  # Realtime
```

Twilio's bidirectional `<Stream>` only ever carries 8kHz G.711 u-law audio, so
each provider exports the constant to put in your `session_config` — the two
APIs describe that same audio with different schemas, so the constants are
**not** interchangeable.

## OpenAI GPT-Live

::: tac.channels.voice.media_streams.gpt_live
    options:
      members:
        - GPTLiveProvider
        - GPTLiveProviderConfig
        - InitiateVoiceConversationOptionsGPTLive
        - TWILIO_AUDIO_FORMAT_FOR_GPT_LIVE

## OpenAI Realtime

::: tac.channels.voice.media_streams.openai_realtime
    options:
      members:
        - OpenAIRealtimeProvider
        - OpenAIRealtimeProviderConfig
        - InitiateVoiceConversationOptionsOpenAIRealtime
        - TWILIO_AUDIO_FORMAT_FOR_REALTIME

## Shared

Used by both providers.

::: tac.channels.voice.media_streams.twiml
    options:
      members:
        - generate_twiml

::: tac.models.voice
    options:
      members:
        - VoiceTwiMLOptionsMediaStreams

::: tac.models.stream
    options:
      members:
        - StreamStartMessage
