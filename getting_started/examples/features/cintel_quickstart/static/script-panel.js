/**
 * Script Panel - Handles checkpoint updates and summary display via SSE
 */

let eventSource;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;

function connectSSE() {
  eventSource = new EventSource('/api/events');

  eventSource.addEventListener('open', () => {
    console.log('[SSE] Connected to event stream');
    reconnectAttempts = 0;
  });

  eventSource.addEventListener('error', (error) => {
    console.error('[SSE] Connection error:', error);

    if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
      reconnectAttempts++;
      const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
      console.log(`[SSE] Reconnecting in ${delay}ms (attempt ${reconnectAttempts})`);

      setTimeout(() => {
        eventSource.close();
        connectSSE();
      }, delay);
    } else {
      console.error('[SSE] Max reconnection attempts reached');
      alert('Lost connection to server. Please refresh the page.');
    }
  });

  eventSource.addEventListener('message', (event) => {
    try {
      const message = JSON.parse(event.data);
      document.dispatchEvent(new CustomEvent('sse-message', { detail: message }));

      if (message.type === 'checkpoint-update') {
        handleCheckpointUpdate(message.data);
      } else if (message.type === 'summary-update') {
        handleSummaryUpdate(message.data);
      } else if (message.type === 'call-ended') {
        handleCallEnded();
      }
    } catch (error) {
      console.error('[SSE] Error parsing message:', error);
    }
  });
}

connectSSE();

function formatCriteriaKey(key) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function handleCheckpointUpdate(data) {
  const { category, completed, skipped, criteria = [] } = data;
  console.log(`[Checkpoint] ${category}: completed=${completed}, skipped=${skipped}`);

  const checkpoint = document.querySelector(`[data-category="${category}"]`);
  if (!checkpoint) {
    console.warn(`[Checkpoint] No element found for category: "${category}"`);
    return;
  }

  const statusEl = checkpoint.querySelector('.checkpoint-status');
  if (completed) {
    statusEl.textContent = '✓';
    checkpoint.classList.add('completed');
    checkpoint.classList.remove('skipped');
  } else if (skipped) {
    statusEl.textContent = '✕';
    checkpoint.classList.add('skipped');
    checkpoint.classList.remove('completed');
  }

  document.getElementById('status-dot').classList.add('active');
  document.getElementById('status-text').textContent = 'Call in progress';

  const criteriaList = checkpoint.querySelector('.criteria-list');
  if (criteriaList && criteria.length > 0) {
    criteria.forEach(c => {
      const cls = !c.evaluated ? 'pending' : c.met ? 'succeeded' : 'failed';
      const icon = !c.evaluated ? '·' : c.met ? '✓' : '✗';

      const existing = c.key ? criteriaList.querySelector(`[data-key="${c.key}"]`) : null;
      if (existing) {
        existing.className = cls;
        existing.querySelector('.criteria-icon').textContent = icon;
      } else {
        const li = document.createElement('li');
        li.className = cls;
        if (c.key) li.dataset.key = c.key;
        li.innerHTML = `<span class="criteria-icon">${icon}</span>${formatCriteriaKey(c.key)}`;
        criteriaList.appendChild(li);
      }
    });
  }
}

function handleCallEnded() {
  document.getElementById('status-dot').classList.remove('active');
  document.getElementById('status-text').textContent = 'Call ended';
}

function handleSummaryUpdate(data) {
  const { summary_text } = data;
  console.log('[Summary] Received:', summary_text?.slice(0, 60));

  handleCallEnded();
  document.getElementById('summary-panel').style.display = 'block';
  document.getElementById('summary-text').textContent = summary_text;
}
