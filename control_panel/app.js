/**
 * StreamSense Control Panel & Live Drift Visualizer
 * Real-time WebSocket consumer and Chart.js telemetry renderer.
 */

// State
let feedSocket = null;
let metricsSocket = null;
let isStreaming = false;
let messageCount = 0;
let windowCount = 0;

// Charts
let chartMagnitude = null;
let chartSimilarity = null;
let chartSentiment = null;

// Elements cache
const el = {
  statusPulse: document.getElementById('statusPulse'),
  activeAsin: document.getElementById('activeAsin'),
  statStreamedCount: document.getElementById('statStreamedCount'),
  statWindowsCount: document.getElementById('statWindowsCount'),
  statDriftScore: document.getElementById('statDriftScore'),
  feedLiveStatus: document.getElementById('feedLiveStatus'),
  feedCardsContainer: document.getElementById('feedCardsContainer'),
  feedEmptyState: document.getElementById('feedEmptyState'),

  // Drift Controls
  driftCurveSelect: document.getElementById('driftCurveSelect'),
  cycleLengthSlider: document.getElementById('cycleLengthSlider'),
  cycleLengthVal: document.getElementById('cycleLengthVal'),

  toggleAdjSwap: document.getElementById('toggleAdjSwap'),
  sliderAdjSwap: document.getElementById('sliderAdjSwap'),
  valAdjSwap: document.getElementById('valAdjSwap'),

  toggleClassSwap: document.getElementById('toggleClassSwap'),
  toggleClassShift: document.getElementById('toggleClassShift'),

  toggleFormality: document.getElementById('toggleFormality'),
  sliderFormality: document.getElementById('sliderFormality'),
  valFormality: document.getElementById('valFormality'),

  toggleNoise: document.getElementById('toggleNoise'),
  sliderNoise: document.getElementById('sliderNoise'),
  valNoise: document.getElementById('valNoise'),

  // Feed Controls
  speedSlider: document.getElementById('speedSlider'),
  speedVal: document.getElementById('speedVal'),
  windowCountInput: document.getElementById('windowCountInput'),
  windowTimeInput: document.getElementById('windowTimeInput'),

  btnStartFeed: document.getElementById('btnStartFeed'),
  btnPauseFeed: document.getElementById('btnPauseFeed'),
  btnResetFeed: document.getElementById('btnResetFeed'),

  // Window Summary Elements
  lblTriggerReason: document.getElementById('lblTriggerReason'),
  lblCosineSim: document.getElementById('lblCosineSim'),
  lblVocabOverlap: document.getElementById('lblVocabOverlap'),
  lblKLDiv: document.getElementById('lblKLDiv')
};

document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  bindControlEvents();
  fetchInitialStatus();
  connectWebSockets();
});

// ==========================================
// Chart.js Setup
// ==========================================
function initCharts() {
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 200 },
    plugins: {
      legend: {
        labels: { color: '#94a3b8', font: { size: 9, family: 'Outfit' }, boxWidth: 8 }
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255,255,255,0.03)' },
        ticks: { color: '#64748b', font: { size: 8, family: 'JetBrains Mono' }, maxTicksLimit: 6 }
      },
      y: {
        grid: { color: 'rgba(255,255,255,0.03)' },
        ticks: { color: '#64748b', font: { size: 8, family: 'JetBrains Mono' } }
      }
    }
  };

  // 1. Composite Drift Magnitude Chart
  const ctxMag = document.getElementById('chartMagnitude').getContext('2d');
  chartMagnitude = new Chart(ctxMag, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Net Drift %',
        data: [],
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245, 158, 11, 0.12)',
        borderWidth: 1.8,
        fill: true,
        tension: 0.35,
        pointRadius: 2
      }]
    },
    options: {
      ...commonOptions,
      scales: {
        ...commonOptions.scales,
        y: { ...commonOptions.scales.y, min: 0, max: 100 }
      }
    }
  });

  // 2. Similarity & Overlap Chart
  const ctxSim = document.getElementById('chartSimilarity').getContext('2d');
  chartSimilarity = new Chart(ctxSim, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Cosine Sim',
          data: [],
          borderColor: '#38bdf8',
          borderWidth: 1.8,
          tension: 0.3,
          pointRadius: 2
        },
        {
          label: 'Vocab Overlap',
          data: [],
          borderColor: '#a855f7',
          borderWidth: 1.5,
          borderDash: [3, 3],
          tension: 0.3,
          pointRadius: 2
        }
      ]
    },
    options: {
      ...commonOptions,
      scales: {
        ...commonOptions.scales,
        y: { ...commonOptions.scales.y, min: 0, max: 1.0 }
      }
    }
  });

  // 3. Sentiment Distribution & KL Chart
  const ctxSent = document.getElementById('chartSentiment').getContext('2d');
  chartSentiment = new Chart(ctxSent, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Pos %',
          data: [],
          borderColor: '#10b981',
          borderWidth: 1.5,
          tension: 0.3,
          pointRadius: 0
        },
        {
          label: 'Neg %',
          data: [],
          borderColor: '#f43f5e',
          borderWidth: 1.5,
          tension: 0.3,
          pointRadius: 0
        },
        {
          label: 'KL-Div',
          data: [],
          borderColor: '#e2e8f0',
          borderWidth: 1.8,
          borderDash: [2, 2],
          tension: 0.2,
          pointRadius: 2
        }
      ]
    },
    options: {
      ...commonOptions,
      scales: {
        ...commonOptions.scales,
        y: { ...commonOptions.scales.y, min: 0, max: 1.5 }
      }
    }
  });
}

// ==========================================
// WebSockets
// ==========================================
function connectWebSockets() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;

  // 1. Message Feed WebSocket
  feedSocket = new WebSocket(`${protocol}//${host}/ws/feed`);
  feedSocket.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      renderFeedCard(msg);
      messageCount++;
      el.statStreamedCount.textContent = messageCount.toLocaleString();
    } catch (e) {
      console.error("Error parsing feed message:", e);
    }
  };

  feedSocket.onclose = () => {
    setTimeout(connectWebSockets, 2000);
  };

  // 2. Metrics WebSocket
  metricsSocket = new WebSocket(`${protocol}//${host}/ws/metrics`);
  metricsSocket.onmessage = (event) => {
    try {
      const metrics = JSON.parse(event.data);
      updateMetricsDashboard(metrics);
      windowCount++;
      el.statWindowsCount.textContent = windowCount.toLocaleString();
    } catch (e) {
      console.error("Error parsing metrics:", e);
    }
  };
}

// ==========================================
// Render Live Comparison Feed Cards
// ==========================================
function renderFeedCard(item) {
  if (el.feedEmptyState) {
    el.feedEmptyState.style.display = 'none';
  }

  const card = document.createElement('div');
  card.className = `feed-pair-card ${item.is_drifted ? 'is-drifted' : ''}`;

  const origStars = '★'.repeat(Math.round(item.original_score)) + '☆'.repeat(5 - Math.round(item.original_score));
  const driftStars = '★'.repeat(Math.round(item.drifted_score)) + '☆'.repeat(5 - Math.round(item.drifted_score));
  const isScoreShifted = item.original_score !== item.drifted_score;

  // Active drift chips
  let chipsHtml = '';
  if (item.active_drifts && item.active_drifts.length > 0) {
    chipsHtml = item.active_drifts.map(d => {
      let chipClass = 'chip';
      if (d.includes('Adjective')) chipClass += ' semantic';
      else if (d.includes('Slang') || d.includes('Formality')) chipClass += ' slang';
      else if (d.includes('Noise')) chipClass += ' noise';
      return `<span class="${chipClass}">${escapeHtml(d)}</span>`;
    }).join('');
  }

  // Highlight word substitutions (Semantic antonyms & Slang shift)
  const highlightedDriftSummary = highlightSubstitutions(item.drifted_summary, item.modifications);
  const highlightedDriftText = highlightSubstitutions(item.drifted_text, item.modifications);

  card.innerHTML = `
    <!-- Left: Original -->
    <div class="card-side orig">
      <div class="card-meta">
        <span class="meta-user">${escapeHtml(item.profileName)}</span>
        <span class="meta-stars">${origStars} (${item.original_score}★)</span>
      </div>
      <div class="card-sum">${escapeHtml(item.original_summary)}</div>
      <div class="card-body">${escapeHtml(item.original_text)}</div>
    </div>

    <!-- Right: Drifted Stream -->
    <div class="card-side drift">
      <div class="card-meta">
        <span class="meta-user">Stream Msg #${item.step}</span>
        <span class="meta-stars ${isScoreShifted ? 'flipped' : ''}">
          ${driftStars} (${item.drifted_score}★) ${isScoreShifted ? '⚠️' : ''}
        </span>
      </div>
      <div class="card-sum">${highlightedDriftSummary}</div>
      <div class="card-body">${highlightedDriftText}</div>
      <div class="chips-row">${chipsHtml}</div>
    </div>
  `;

  el.feedCardsContainer.insertBefore(card, el.feedCardsContainer.firstChild);

  // Keep max 35 cards in DOM for performance
  while (el.feedCardsContainer.children.length > 35) {
    el.feedCardsContainer.removeChild(el.feedCardsContainer.lastChild);
  }
}

function highlightSubstitutions(text, modifications) {
  if (!text) return '';
  let safeText = escapeHtml(text);
  if (!modifications || modifications.length === 0) {
    return safeText;
  }

  // Filter semantic adjective swaps, slang shifts, and typo-corrupted words (len > 1)
  const validMods = modifications.filter(m => {
    const isTargetType = (m.type === 'adjective_swap' || m.type === 'slang_shift' || m.type === 'typo');
    const word = (m.corrupted_word || m.replaced_with || m.replacement || '').trim();
    return isTargetType && word.length > 1;
  });

  if (validMods.length === 0) {
    return safeText;
  }

  // Deduplicate and sort by length descending so longer phrases match first
  const seen = new Set();
  const sortedMods = [];
  for (const mod of validMods) {
    const word = (mod.corrupted_word || mod.replaced_with || mod.replacement || '').trim();
    const key = `${mod.type}:${word.toLowerCase()}`;
    if (!seen.has(key)) {
      seen.add(key);
      let cssClass = 'highlight-swap';
      if (mod.type === 'slang_shift') {
        cssClass = 'highlight-slang';
      } else if (mod.type === 'typo') {
        cssClass = 'highlight-typo';
      }

      sortedMods.push({
        word: word,
        type: mod.type,
        cssClass: cssClass
      });
    }
  }

  sortedMods.sort((a, b) => {
    if (b.word.length !== a.word.length) {
      return b.word.length - a.word.length;
    }
    const prio = { 'typo': 1, 'slang_shift': 2, 'adjective_swap': 3 };
    return (prio[a.type] || 9) - (prio[b.type] || 9);
  });

  for (const item of sortedMods) {
    const regex = createWordRegex(item.word);
    const parts = safeText.split(/(<[^>]+>)/g);
    for (let i = 0; i < parts.length; i++) {
      if (!parts[i].startsWith('<')) {
        parts[i] = parts[i].replace(regex, (match) => `<span class="${item.cssClass}">${match}</span>`);
      }
    }
    safeText = parts.join('');
  }

  return safeText;
}

function createWordRegex(phrase) {
  const safePhrase = escapeHtml(phrase);
  const escaped = escapeRegExp(safePhrase);
  const startsWithWord = /^\w/.test(phrase);
  const endsWithWord = /\w$/.test(phrase);

  const prefix = startsWithWord ? '\\b' : '(?:^|(?<=[\\s\\W]))';
  const suffix = endsWithWord ? '\\b' : '(?=[\\s\\W]|$)';
  return new RegExp(`${prefix}${escaped}${suffix}`, 'gi');
}

// ==========================================
// Update Analytics Dashboard Charts
// ==========================================
function updateMetricsDashboard(metrics) {
  const timestamp = metrics.timestamp || new Date().toLocaleTimeString();

  el.statDriftScore.textContent = `${metrics.drift_magnitude_pct}%`;
  el.lblTriggerReason.textContent = metrics.trigger_reason;
  el.lblCosineSim.textContent = metrics.cosine_similarity.toFixed(3);
  el.lblVocabOverlap.textContent = `${(metrics.vocab_overlap * 100).toFixed(1)}%`;
  el.lblKLDiv.textContent = metrics.sentiment_kl_divergence.toFixed(3);

  appendChartData(chartMagnitude, timestamp, [metrics.drift_magnitude_pct]);
  appendChartData(chartSimilarity, timestamp, [metrics.cosine_similarity, metrics.vocab_overlap]);

  const posPct = (metrics.drifted_sentiment_dist?.positive || 0);
  const negPct = (metrics.drifted_sentiment_dist?.negative || 0);
  appendChartData(chartSentiment, timestamp, [posPct, negPct, metrics.sentiment_kl_divergence]);
}

function appendChartData(chart, label, dataValues) {
  if (!chart) return;
  const maxPoints = 18;

  chart.data.labels.push(label);
  chart.data.datasets.forEach((ds, idx) => {
    ds.data.push(dataValues[idx]);
  });

  if (chart.data.labels.length > maxPoints) {
    chart.data.labels.shift();
    chart.data.datasets.forEach(ds => ds.data.shift());
  }

  chart.update('none');
}

// ==========================================
// Control Bindings
// ==========================================
function bindControlEvents() {
  el.sliderAdjSwap.addEventListener('input', () => {
    el.valAdjSwap.textContent = `${Math.round(el.sliderAdjSwap.value * 100)}%`;
    syncDriftConfig();
  });
  el.sliderFormality.addEventListener('input', () => {
    el.valFormality.textContent = `${Math.round(el.sliderFormality.value * 100)}%`;
    syncDriftConfig();
  });
  el.sliderNoise.addEventListener('input', () => {
    el.valNoise.textContent = `${Math.round(el.sliderNoise.value * 100)}%`;
    syncDriftConfig();
  });
  el.cycleLengthSlider.addEventListener('input', () => {
    el.cycleLengthVal.textContent = `${el.cycleLengthSlider.value} msgs`;
    syncDriftConfig();
  });
  el.speedSlider.addEventListener('input', () => {
    el.speedVal.textContent = `${parseFloat(el.speedSlider.value).toFixed(1)} msgs/s`;
    syncFeedConfig();
  });

  [
    el.toggleAdjSwap, el.toggleClassSwap, el.toggleClassShift,
    el.toggleFormality, el.toggleNoise, el.driftCurveSelect
  ].forEach(input => {
    input.addEventListener('change', () => {
      updateMethodBoxStyles();
      syncDriftConfig();
    });
  });

  [el.windowCountInput, el.windowTimeInput].forEach(inp => {
    inp.addEventListener('change', syncFeedConfig);
  });

  el.btnStartFeed.addEventListener('click', async () => {
    await fetch('/api/feed/start', { method: 'POST' });
    setStreamingUI(true);
  });

  el.btnPauseFeed.addEventListener('click', async () => {
    await fetch('/api/feed/stop', { method: 'POST' });
    setStreamingUI(false);
  });

  el.btnResetFeed.addEventListener('click', async () => {
    await fetch('/api/feed/reset', { method: 'POST' });
    messageCount = 0;
    windowCount = 0;
    el.statStreamedCount.textContent = '0';
    el.statWindowsCount.textContent = '0';
    el.statDriftScore.textContent = '0.0%';
    el.feedCardsContainer.innerHTML = `
      <div class="empty-feed-placeholder" id="feedEmptyState">
        <div class="empty-icon">📡</div>
        <h4>Stream Simulator is Idle</h4>
        <p>Click <strong>Start Stream</strong> in the control panel to begin streaming Amazon movie reviews through the drift engine.</p>
      </div>
    `;
    resetCharts();
    setStreamingUI(false);
  });
}

function updateMethodBoxStyles() {
  document.getElementById('cardAdjSwap').classList.toggle('active', el.toggleAdjSwap.checked);
  document.getElementById('cardClassSwap').classList.toggle('active', el.toggleClassSwap.checked);
  document.getElementById('cardClassShift').classList.toggle('active', el.toggleClassShift.checked);
  document.getElementById('cardFormality').classList.toggle('active', el.toggleFormality.checked);
  document.getElementById('cardNoise').classList.toggle('active', el.toggleNoise.checked);
}

function resetCharts() {
  [chartMagnitude, chartSimilarity, chartSentiment].forEach(ch => {
    if (ch) {
      ch.data.labels = [];
      ch.data.datasets.forEach(ds => ds.data = []);
      ch.update();
    }
  });
}

function setStreamingUI(streaming) {
  isStreaming = streaming;
  el.statusPulse.classList.toggle('active', streaming);
  el.feedLiveStatus.textContent = streaming ? 'LIVE STREAMING' : 'STREAM IDLE';
  el.feedLiveStatus.className = `feed-badge-status ${streaming ? 'streaming' : ''}`;
  el.btnStartFeed.disabled = streaming;
  el.btnPauseFeed.disabled = !streaming;
}

async function syncDriftConfig() {
  const payload = {
    enable_adjective_swap: el.toggleAdjSwap.checked,
    enable_class_swap: el.toggleClassSwap.checked,
    enable_class_shift: el.toggleClassShift.checked,
    enable_formality_shift: el.toggleFormality.checked,
    enable_noise_injection: el.toggleNoise.checked,
    adjective_swap_intensity: parseFloat(el.sliderAdjSwap.value),
    formality_intensity: parseFloat(el.sliderFormality.value),
    noise_intensity: parseFloat(el.sliderNoise.value),
    drift_curve: el.driftCurveSelect.value,
    drift_cycle_length: parseInt(el.cycleLengthSlider.value)
  };

  await fetch('/api/drift/configure', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
}

async function syncFeedConfig() {
  const payload = {
    messages_per_second: parseFloat(el.speedSlider.value),
    window_message_count: parseInt(el.windowCountInput.value),
    window_time_seconds: parseFloat(el.windowTimeInput.value)
  };

  await fetch('/api/feed/configure', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
}

async function fetchInitialStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    setStreamingUI(data.is_running);
    if (data.dataset_info?.target_asin) {
      el.activeAsin.textContent = data.dataset_info.target_asin;
    }
  } catch (e) {
    console.error("Could not fetch status:", e);
  }
}

function escapeHtml(text) {
  if (!text) return '';
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
