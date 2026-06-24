/**
 * Dashboard polling — fetches /api/transcript every 1.5s and updates UI.
 */

let lastTurnCount = 0;

fetch('/api/config')
  .then(r => r.json())
  .then(d => {
    const el = document.getElementById('phone-number');
    if (el) el.textContent = d.phone_number || 'Not configured';
  })
  .catch(() => {});

fetch('/api/script')
  .then(r => r.json())
  .then(hints => {
    Object.entries(hints).forEach(([category, example]) => {
      const cp = document.querySelector(`[data-category="${category}"]`);
      if (cp) {
        const hint = cp.querySelector('.checkpoint-hint');
        if (hint) hint.textContent = example;
      }
    });
  })
  .catch(() => {});

function poll() {
  fetch('/api/transcript')
    .then(r => r.json())
    .then(data => {
      updateTranscript(data.turns || []);
      updateCheckpoints(data.checkpoints || []);
      updateStatus(data.call_active, data.summary);
      if (data.summary) updateSummary(data.summary);
    })
    .catch(() => {})
    .finally(() => setTimeout(poll, 1500));
}

poll();

function updateTranscript(turns) {
  const container = document.getElementById('transcript');
  if (turns.length === lastTurnCount) return;

  // Remove empty state placeholder
  const empty = container.querySelector('.transcript-empty');
  if (empty) empty.remove();

  // Append only new turns
  for (let i = lastTurnCount; i < turns.length; i++) {
    const { speaker, text } = turns[i];
    const div = document.createElement('div');
    div.className = `transcript-item ${speaker}`;

    const label = document.createElement('div');
    label.className = 'speaker-label';
    label.textContent = speaker === 'agent' ? 'Agent' : 'Customer';

    const textEl = document.createElement('div');
    textEl.className = 'text';
    textEl.textContent = text;

    div.appendChild(label);
    div.appendChild(textEl);
    container.appendChild(div);
  }

  lastTurnCount = turns.length;
  container.scrollTop = container.scrollHeight;
}

function updateCheckpoints(checkpointList) {
  checkpointList.forEach(cp => {
    const { category, completed, skipped, criteria = [] } = cp;
    const el = document.querySelector(`[data-category="${category}"]`);
    if (!el) return;

    if (completed) {
      el.classList.add('completed');
      el.classList.remove('skipped');
      el.querySelector('.checkpoint-status').textContent = '✓';
    } else if (skipped) {
      el.classList.add('skipped');
      el.classList.remove('completed');
      el.querySelector('.checkpoint-status').textContent = '✕';
    }

    const criteriaList = el.querySelector('.criteria-list');
    if (criteriaList && criteria.length > 0) {
      criteria.forEach(c => {
        const cls = !c.evaluated ? 'pending' : c.met ? 'succeeded' : 'failed';
        const icon = !c.evaluated ? '' : c.met ? '✓' : '✗';
        const existing = criteriaList.querySelector(`[data-key="${c.key}"]`);
        if (existing) {
          existing.className = cls;
          existing.querySelector('.criteria-icon').textContent = icon;
        }
      });
    }
  });
}

function updateStatus(callActive, summary) {
  const dot = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  if (!dot || !text) return;

  if (summary) {
    dot.classList.remove('active');
    text.textContent = 'Call ended';
  } else if (callActive) {
    dot.classList.add('active');
    text.textContent = 'Call in progress';
  }
}

function updateSummary(text) {
  const panel = document.getElementById('summary-panel');
  const el = document.getElementById('summary-text');
  if (panel && el && !panel.style.display.includes('block')) {
    panel.style.display = 'block';
    el.textContent = text;
  }
}
