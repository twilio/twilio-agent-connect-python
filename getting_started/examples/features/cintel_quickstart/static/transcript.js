/**
 * Transcript Panel - Displays live conversation transcript
 */

const transcriptContainer = document.getElementById('transcript');
let interimTranscript = null;

fetch('/api/config')
  .then(r => r.json())
  .then(data => {
    const el = document.getElementById('phone-number');
    if (el) el.textContent = data.phone_number || 'Not configured';
  })
  .catch(() => {
    const el = document.getElementById('phone-number');
    if (el) el.textContent = 'Unavailable';
  });

// Fetch script hints from Intelligence Config and populate checkpoint hints
fetch('/api/script')
  .then(r => r.json())
  .then(hints => {
    Object.entries(hints).forEach(([category, example]) => {
      const checkpoint = document.querySelector(`[data-category="${category}"]`);
      if (checkpoint) {
        const hint = checkpoint.querySelector('.checkpoint-hint');
        if (hint) hint.textContent = example;
      }
    });
  })
  .catch(() => {});

document.addEventListener('sse-message', (event) => {
  const message = event.detail;
  if (message.type === 'transcript-update') {
    handleTranscriptUpdate(message.data);
  }
});

function handleTranscriptUpdate(data) {
  const { speaker, text, interim } = data;

  const empty = transcriptContainer.querySelector('.transcript-empty');
  if (empty) empty.remove();

  if (interim) {
    if (interimTranscript && interimTranscript.dataset.speaker === speaker) {
      interimTranscript.querySelector('.text').textContent = text;
    } else {
      if (interimTranscript) interimTranscript.remove();
      interimTranscript = createTranscriptElement(speaker, text, true);
      transcriptContainer.appendChild(interimTranscript);
    }
  } else {
    if (interimTranscript) {
      interimTranscript.remove();
      interimTranscript = null;
    }
    const finalElement = createTranscriptElement(speaker, text, false);
    transcriptContainer.appendChild(finalElement);
    transcriptContainer.scrollTop = transcriptContainer.scrollHeight;
  }
}

function createTranscriptElement(speaker, text, isInterim) {
  const div = document.createElement('div');
  div.className = `transcript-item ${speaker}`;
  div.dataset.speaker = speaker;
  if (isInterim) div.style.opacity = '0.6';

  const speakerLabel = document.createElement('div');
  speakerLabel.className = 'speaker-label';
  speakerLabel.textContent = speaker === 'agent' ? 'Agent' : 'Customer';

  const textEl = document.createElement('div');
  textEl.className = 'text';
  textEl.textContent = text;

  div.appendChild(speakerLabel);
  div.appendChild(textEl);
  return div;
}
