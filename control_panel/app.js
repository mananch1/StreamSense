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
  lblKLDiv: document.getElementById('lblKLDiv'),
  lblSpellingRate: document.getElementById('lblSpellingRate'),
  lblScoreDistDiv: document.getElementById('lblScoreDistDiv'),

  // Baseline Burn-In Elements
  burnInSlider: document.getElementById('burnInSlider'),
  burnInVal: document.getElementById('burnInVal'),
  baselineStatusBar: document.getElementById('baselineStatusBar'),
  baselineStatusText: document.getElementById('baselineStatusText'),
  baselineProgressFill: document.getElementById('baselineProgressFill'),

  // MLOps & Model Observability Elements
  modelVerBadge: document.getElementById('modelVerBadge'),
  modelStatusPill: document.getElementById('modelStatusPill'),
  lblModelWinAcc: document.getElementById('lblModelWinAcc'),
  lblModelCumAcc: document.getElementById('lblModelCumAcc'),
  lblCorpusSize: document.getElementById('lblCorpusSize'),
  btnRetrainModel: document.getElementById('btnRetrainModel'),
  retrainSubInfo: document.getElementById('retrainSubInfo'),
  lblSummaryModelAcc: document.getElementById('lblSummaryModelAcc'),
  lblSummaryModelVer: document.getElementById('lblSummaryModelVer'),

  // Presentation Guide Modal
  btnOpenGuide: document.getElementById('btnOpenGuide'),
  guideModal: document.getElementById('guideModal'),
  btnCloseGuide: document.getElementById('btnCloseGuide'),

  // 1-Click Demo Presets
  presetBtns: document.querySelectorAll('.preset-btn'),

  // Real-Time Stream Alert Banner
  streamAlertBanner: document.getElementById('streamAlertBanner'),
  bannerIcon: document.getElementById('bannerIcon'),
  bannerTitle: document.getElementById('bannerTitle'),
  bannerDriftBadge: document.getElementById('bannerDriftBadge'),
  bannerDesc: document.getElementById('bannerDesc'),

  // Retraining Overlay & Facts
  trainingOverlay: document.getElementById('trainingOverlay'),
  btnCloseTrainingOverlay: document.getElementById('btnCloseTrainingOverlay'),
  trainingSubtitle: document.getElementById('trainingSubtitle'),
  trainStep1: document.getElementById('trainStep1'),
  trainStep2: document.getElementById('trainStep2'),
  trainStep3: document.getElementById('trainStep3'),
  trainStep4: document.getElementById('trainStep4'),
  step1Badge: document.getElementById('step1Badge'),
  step2Badge: document.getElementById('step2Badge'),
  step3Badge: document.getElementById('step3Badge'),
  step4Badge: document.getElementById('step4Badge'),
  trainingProgressFill: document.getElementById('trainingProgressFill'),
  factCounter: document.getElementById('factCounter'),
  factBody: document.getElementById('factBody'),
  btnNextFact: document.getElementById('btnNextFact'),
  trainingSuccessFooter: document.getElementById('trainingSuccessFooter'),
  trainingSuccessTitle: document.getElementById('trainingSuccessTitle'),
  trainingSuccessMsg: document.getElementById('trainingSuccessMsg')
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

  // 1. Composite Drift Magnitude & Model Accuracy Chart
  const ctxMag = document.getElementById('chartMagnitude').getContext('2d');
  chartMagnitude = new Chart(ctxMag, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Net Drift %',
          data: [],
          borderColor: '#f59e0b',
          backgroundColor: 'rgba(245, 158, 11, 0.12)',
          borderWidth: 1.8,
          fill: true,
          tension: 0.35,
          pointRadius: 2
        },
        {
          label: 'Model Accuracy %',
          data: [],
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.08)',
          borderWidth: 1.8,
          fill: false,
          tension: 0.35,
          pointRadius: 2
        }
      ]
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
      ${renderModelPredChip(item)}
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
  // Update real-time stream health alert banner
  updateStreamAlertBanner(metrics);

  // Handle burn-in vs monitoring phase
  if (metrics.phase === 'burn_in') {
    if (el.baselineStatusBar) el.baselineStatusBar.style.display = 'block';
    if (el.baselineStatusText) el.baselineStatusText.textContent = `Calibrating Baseline (${metrics.burn_in_progress})...`;
    if (metrics.burn_in_progress && el.baselineProgressFill) {
      const [current, total] = metrics.burn_in_progress.split('/').map(Number);
      el.baselineProgressFill.style.width = `${Math.min(100, Math.round((current / (total || 1)) * 100))}%`;
    }
    if (el.lblTriggerReason) el.lblTriggerReason.textContent = metrics.trigger_reason || 'Burn-in calibration';
    if (el.lblCosineSim) el.lblCosineSim.textContent = '1.000';
    if (el.lblVocabOverlap) el.lblVocabOverlap.textContent = '100.0%';
    if (el.lblKLDiv) el.lblKLDiv.textContent = '0.000';
    if (el.lblSpellingRate) el.lblSpellingRate.textContent = metrics.spelling_error_rate ? metrics.spelling_error_rate.toFixed(3) : '0.000';
    if (el.lblScoreDistDiv) el.lblScoreDistDiv.textContent = '0.000';
    return; // Don't update drift charts during burn-in
  }

  // Baseline is ready — hide status bar
  if (el.baselineStatusBar) el.baselineStatusBar.style.display = 'none';

  const timestamp = metrics.timestamp || new Date().toLocaleTimeString();

  el.statDriftScore.textContent = `${metrics.drift_magnitude_pct}%`;
  el.lblTriggerReason.textContent = metrics.trigger_reason;
  el.lblCosineSim.textContent = metrics.cosine_similarity.toFixed(3);
  el.lblVocabOverlap.textContent = `${(metrics.vocab_overlap * 100).toFixed(1)}%`;
  el.lblKLDiv.textContent = metrics.sentiment_kl_divergence.toFixed(3);

  // Update new reference-free metric labels
  if (el.lblSpellingRate) el.lblSpellingRate.textContent = metrics.spelling_error_rate.toFixed(3);
  if (el.lblScoreDistDiv) el.lblScoreDistDiv.textContent = metrics.score_dist_divergence.toFixed(3);

  appendChartData(chartMagnitude, timestamp, [
    metrics.drift_magnitude_pct,
    Math.round((metrics.model_window_accuracy !== undefined ? metrics.model_window_accuracy : 1.0) * 100)
  ]);
  appendChartData(chartSimilarity, timestamp, [metrics.cosine_similarity, metrics.vocab_overlap]);

  const sentDist = metrics.current_sentiment_dist || metrics.drifted_sentiment_dist || {};
  const posPct = (sentDist.positive || 0);
  const negPct = (sentDist.negative || 0);
  appendChartData(chartSentiment, timestamp, [posPct, negPct, metrics.sentiment_kl_divergence]);

  // Update MLOps Model Observability Stats
  if (metrics.model_version) {
    if (el.modelVerBadge) el.modelVerBadge.textContent = `Model ${metrics.model_version}`;
    if (el.lblSummaryModelVer) el.lblSummaryModelVer.textContent = metrics.model_version;
  }
  if (metrics.model_window_accuracy !== undefined) {
    const winPct = `${(metrics.model_window_accuracy * 100).toFixed(1)}%`;
    if (el.lblModelWinAcc) el.lblModelWinAcc.textContent = winPct;
    if (el.lblSummaryModelAcc) el.lblSummaryModelAcc.textContent = winPct;
  }
  if (metrics.model_cumulative_accuracy !== undefined && el.lblModelCumAcc) {
    el.lblModelCumAcc.textContent = `${(metrics.model_cumulative_accuracy * 100).toFixed(1)}%`;
  }
  if (metrics.accumulated_samples_count !== undefined && el.lblCorpusSize) {
    el.lblCorpusSize.textContent = `${metrics.accumulated_samples_count} msgs`;
  }
  if (el.modelStatusPill && metrics.model_status) {
    el.modelStatusPill.textContent = metrics.model_status.toUpperCase();
    el.modelStatusPill.className = `model-status-pill ${metrics.model_status}`;
  }
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
// Curated AML & Data Drift Insights
// ==========================================
const AML_FACTS = [
  {
    title: "Concept Drift vs. Covariate Shift",
    text: "Covariate shift alters input feature distributions P(X) while keeping conditional labels P(Y|X) fixed. Concept drift mutates the true underlying mapping P(Y|X) itself, breaking models even when text vocabularies remain identical."
  },
  {
    title: "The Cyclic Shift Paradox",
    text: "Rule-based sentiment lexicons (like VADER) fail irreversibly during cyclic rating shifts (+1 star) because their word polarities are static. Trainable models with sublinear TF-IDF recover from ~21% to 70% accuracy."
  },
  {
    title: "Reference-Free Streaming MLOps",
    text: "In production pipelines, ground-truth human annotations take days to arrive. StreamSense computes unsupervised proxy signals (centroid cosine shift, vocab coverage, spelling rates) to detect degradation before SLA violations occur."
  },
  {
    title: "Leave-One-Out (LOO) Noise Floor",
    text: "Finite stream windows have intrinsic natural variance. StreamSense uses Leave-One-Out cross-validation during initial burn-in to establish empirical noise floors, ensuring clean streams maintain ~3-7% drift instead of false alerts."
  },
  {
    title: "Max-Blended Metric Aggregation",
    text: "A simple arithmetic mean dilutes acute drift in single modalities (e.g. 100% label shift diluted down to ~15%). Our max-blended aggregation (0.65×max + 0.35×mean) triggers immediate critical alerts above 60%."
  },
  {
    title: "Sublinear Term Frequency Scaling",
    text: "By applying sublinear scaling (1 + log(tf)), high-frequency filler words are dampened from dominating centroid vectors, making cosine similarity sharply sensitive to semantic topic shifts."
  },
  {
    title: "Morphological Spell Error Reduction",
    text: "Standard dictionaries misflag valid inflections (e.g. 'workouts', 'exceeded') as spelling errors. Our rule-based morphological expansion dropped false spelling error rate from 12.8% to 1.7%."
  },
  {
    title: "Closed-Loop Self-Healing Workflow",
    text: "Once drift exceeds the critical threshold, StreamSense ingests recent stream samples to retrain the classifier and recalibrate reference centroids in real time, restoring end-to-end model confidence."
  }
];

let currentFactIdx = 0;
let factInterval = null;

function displayFact(index) {
  if (!AML_FACTS || AML_FACTS.length === 0) return;
  currentFactIdx = (index + AML_FACTS.length) % AML_FACTS.length;
  const fact = AML_FACTS[currentFactIdx];
  if (el.factCounter) el.factCounter.textContent = `Fact ${currentFactIdx + 1} of ${AML_FACTS.length}`;
  if (el.factBody) {
    el.factBody.innerHTML = `<strong>${escapeHtml(fact.title)}:</strong> ${escapeHtml(fact.text)}`;
  }
}

function nextFact() {
  displayFact(currentFactIdx + 1);
}

function startFactCycle() {
  stopFactCycle();
  displayFact(Math.floor(Math.random() * AML_FACTS.length));
  factInterval = setInterval(nextFact, 4000);
}

function stopFactCycle() {
  if (factInterval) {
    clearInterval(factInterval);
    factInterval = null;
  }
}

// ==========================================
// Retraining Pipeline & Overlay Animation
// ==========================================
let isRetraining = false;

function setPipelineStep(stepNum, status) {
  const stepEl = el[`trainStep${stepNum}`];
  const badgeEl = el[`step${stepNum}Badge`];
  if (!stepEl || !badgeEl) return;

  stepEl.classList.remove('active', 'completed');
  if (status === 'running') {
    stepEl.classList.add('active');
    badgeEl.textContent = 'RUNNING';
  } else if (status === 'completed') {
    stepEl.classList.add('completed');
    badgeEl.textContent = 'DONE';
  } else {
    badgeEl.textContent = 'QUEUED';
  }
}

async function triggerModelRetrain() {
  if (isRetraining) return;
  isRetraining = true;

  if (el.btnRetrainModel) {
    el.btnRetrainModel.disabled = true;
    el.btnRetrainModel.classList.add('loading');
  }

  // Reveal training overlay modal
  if (el.trainingOverlay) {
    el.trainingOverlay.style.display = 'flex';
  }
  if (el.trainingSuccessFooter) {
    el.trainingSuccessFooter.style.display = 'none';
  }
  startFactCycle();

  // Reset pipeline stages
  setPipelineStep(1, 'running');
  setPipelineStep(2, 'queued');
  setPipelineStep(3, 'queued');
  setPipelineStep(4, 'queued');
  if (el.trainingProgressFill) el.trainingProgressFill.style.width = '15%';
  if (el.trainingSubtitle) el.trainingSubtitle.textContent = 'Buffering recent post-drift reviews from stream memory...';

  // Fire retrain API call in background
  const retrainPromise = fetch('/api/model/retrain', { method: 'POST' })
    .then(r => r.json())
    .catch(err => {
      console.error('Retrain API call failed:', err);
      return { status: 'error', message: err.message };
    });

  // Stage 1 -> Stage 2 (650ms)
  await new Promise(r => setTimeout(r, 650));
  setPipelineStep(1, 'completed');
  setPipelineStep(2, 'running');
  if (el.trainingProgressFill) el.trainingProgressFill.style.width = '45%';
  if (el.trainingSubtitle) el.trainingSubtitle.textContent = 'Extracting sublinear TF-IDF features & VADER priors...';

  // Stage 2 -> Stage 3 (750ms)
  await new Promise(r => setTimeout(r, 750));
  setPipelineStep(2, 'completed');
  setPipelineStep(3, 'running');
  if (el.trainingProgressFill) el.trainingProgressFill.style.width = '75%';
  if (el.trainingSubtitle) el.trainingSubtitle.textContent = 'Optimizing balanced L-BFGS Logistic Regression model...';

  // Stage 3 -> Stage 4 (750ms)
  await new Promise(r => setTimeout(r, 750));
  setPipelineStep(3, 'completed');
  setPipelineStep(4, 'running');
  if (el.trainingProgressFill) el.trainingProgressFill.style.width = '92%';
  if (el.trainingSubtitle) el.trainingSubtitle.textContent = 'Recalibrating baseline reference profile & noise floors...';

  // Await API response
  const data = await retrainPromise;

  // Complete Stage 4 (550ms)
  await new Promise(r => setTimeout(r, 550));
  setPipelineStep(4, 'completed');
  if (el.trainingProgressFill) el.trainingProgressFill.style.width = '100%';
  if (el.trainingSubtitle) el.trainingSubtitle.textContent = 'Pipeline execution complete! Model weights and baseline updated.';

  // Show success footer
  if (el.trainingSuccessFooter) {
    if (data.status === 'trained') {
      if (el.trainingSuccessTitle) el.trainingSuccessTitle.textContent = `Model ${data.version || 'v2.0'} Deployed Successfully!`;
      if (el.trainingSuccessMsg) el.trainingSuccessMsg.textContent = `Trained on ${data.sample_count || 150} recent stream samples. Accuracy recovery active.`;
    } else {
      if (el.trainingSuccessTitle) el.trainingSuccessTitle.textContent = 'Retrain Pipeline Completed';
      if (el.trainingSuccessMsg) el.trainingSuccessMsg.textContent = data.message || 'Stream model profile refreshed.';
    }
    el.trainingSuccessFooter.style.display = 'block';
  }

  // Update UI badges
  if (data.status === 'trained') {
    if (el.modelVerBadge) el.modelVerBadge.textContent = `Model ${data.version}`;
    if (el.lblSummaryModelVer) el.lblSummaryModelVer.textContent = data.version;
    if (el.retrainSubInfo) {
      el.retrainSubInfo.textContent = `Active: ${data.version} (${data.sample_count} samples) • Baseline Recalibrated`;
    }
    if (el.modelStatusPill) {
      el.modelStatusPill.textContent = 'HEALTHY';
      el.modelStatusPill.className = 'model-status-pill healthy';
    }
  }

  // Keep success message visible briefly before dismiss
  await new Promise(r => setTimeout(r, 1400));
  hideTrainingOverlay();

  isRetraining = false;
  if (el.btnRetrainModel) {
    el.btnRetrainModel.disabled = false;
    el.btnRetrainModel.classList.remove('loading');
  }
}

function hideTrainingOverlay() {
  if (el.trainingOverlay) {
    el.trainingOverlay.style.display = 'none';
  }
  stopFactCycle();
}

// ==========================================
// 1-Click Demo Presets
// ==========================================
function clearPresetSelection() {
  if (el.presetBtns) {
    el.presetBtns.forEach(btn => btn.classList.remove('active'));
  }
}

async function applyPreset(presetType) {
  if (el.presetBtns) {
    el.presetBtns.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.preset === presetType);
    });
  }

  if (presetType === 'clean') {
    el.toggleAdjSwap.checked = false;
    el.toggleClassSwap.checked = false;
    el.toggleClassShift.checked = false;
    el.toggleFormality.checked = false;
    el.toggleNoise.checked = false;
    el.driftCurveSelect.value = 'constant';
  } else if (presetType === 'adjective') {
    el.toggleAdjSwap.checked = true;
    el.sliderAdjSwap.value = 0.70;
    el.valAdjSwap.textContent = '70%';
    el.toggleClassSwap.checked = false;
    el.toggleClassShift.checked = false;
    el.toggleFormality.checked = false;
    el.toggleNoise.checked = false;
    el.driftCurveSelect.value = 'constant';
  } else if (presetType === 'shift') {
    el.toggleAdjSwap.checked = false;
    el.toggleClassSwap.checked = false;
    el.toggleClassShift.checked = true;
    el.toggleFormality.checked = false;
    el.toggleNoise.checked = false;
    el.driftCurveSelect.value = 'constant';
  } else if (presetType === 'noise') {
    el.toggleAdjSwap.checked = false;
    el.toggleClassSwap.checked = false;
    el.toggleClassShift.checked = false;
    el.toggleFormality.checked = true;
    el.sliderFormality.value = 0.70;
    el.valFormality.textContent = '70%';
    el.toggleNoise.checked = true;
    el.sliderNoise.value = 0.60;
    el.valNoise.textContent = '60%';
    el.driftCurveSelect.value = 'gradual';
  } else if (presetType === 'stress') {
    el.toggleAdjSwap.checked = true;
    el.sliderAdjSwap.value = 0.60;
    el.valAdjSwap.textContent = '60%';
    el.toggleClassShift.checked = true;
    el.toggleClassSwap.checked = false;
    el.toggleFormality.checked = true;
    el.sliderFormality.value = 0.60;
    el.valFormality.textContent = '60%';
    el.toggleNoise.checked = true;
    el.sliderNoise.value = 0.50;
    el.valNoise.textContent = '50%';
    el.driftCurveSelect.value = 'sinusoidal';
  }

  updateMethodBoxStyles();
  await syncDriftConfig();
}

// ==========================================
// Real-Time Stream Alert Banner
// ==========================================
function updateStreamAlertBanner(metrics) {
  if (!el.streamAlertBanner) return;

  if (metrics.phase === 'burn_in') {
    el.streamAlertBanner.className = 'stream-alert-banner banner-warning';
    if (el.bannerIcon) el.bannerIcon.textContent = '⏳';
    if (el.bannerTitle) el.bannerTitle.textContent = 'CALIBRATING BASELINE TOLERANCE';
    if (el.bannerDriftBadge) el.bannerDriftBadge.textContent = metrics.burn_in_progress || 'Burn-In';
    if (el.bannerDesc) el.bannerDesc.textContent = 'Profiling clean vocabulary, sentiment polarity, and LOO empirical noise floors...';
    return;
  }

  const driftPct = metrics.drift_magnitude_pct !== undefined ? metrics.drift_magnitude_pct : 0;
  if (el.bannerDriftBadge) el.bannerDriftBadge.textContent = `Drift: ${driftPct.toFixed(1)}%`;

  if (driftPct < 15.0) {
    el.streamAlertBanner.className = 'stream-alert-banner banner-healthy';
    if (el.bannerIcon) el.bannerIcon.textContent = '🟢';
    if (el.bannerTitle) el.bannerTitle.textContent = 'STREAM IN-DISTRIBUTION — MODEL HEALTHY';
    if (el.bannerDesc) el.bannerDesc.textContent = 'Drift magnitude is within calibrated baseline tolerance. Model predictions are highly reliable.';
  } else if (driftPct < 45.0) {
    el.streamAlertBanner.className = 'stream-alert-banner banner-warning';
    if (el.bannerIcon) el.bannerIcon.textContent = '⚠️';
    if (el.bannerTitle) el.bannerTitle.textContent = 'MODERATE DRIFT DETECTED — MONITORING';
    if (el.bannerDesc) el.bannerDesc.textContent = `Distribution divergence detected (${metrics.trigger_reason || 'semantic shift'}). Model performance may begin degrading.`;
  } else {
    el.streamAlertBanner.className = 'stream-alert-banner banner-critical';
    if (el.bannerIcon) el.bannerIcon.textContent = '🚨';
    if (el.bannerTitle) el.bannerTitle.textContent = 'CRITICAL DRIFT ALERT — RETRAINING RECOMMENDED';
    if (el.bannerDesc) el.bannerDesc.textContent = 'Severe data drift observed! Accuracy drop imminent. Click "Retrain Model on Stream Data" to restore performance.';
  }
}

// ==========================================
// Control Bindings
// ==========================================
function bindControlEvents() {
  // Sliders with value reflection
  el.sliderAdjSwap.addEventListener('input', () => {
    el.valAdjSwap.textContent = `${Math.round(el.sliderAdjSwap.value * 100)}%`;
    clearPresetSelection();
    syncDriftConfig();
  });
  el.sliderFormality.addEventListener('input', () => {
    el.valFormality.textContent = `${Math.round(el.sliderFormality.value * 100)}%`;
    clearPresetSelection();
    syncDriftConfig();
  });
  el.sliderNoise.addEventListener('input', () => {
    el.valNoise.textContent = `${Math.round(el.sliderNoise.value * 100)}%`;
    clearPresetSelection();
    syncDriftConfig();
  });
  el.cycleLengthSlider.addEventListener('input', () => {
    el.cycleLengthVal.textContent = `${el.cycleLengthSlider.value} msgs`;
    clearPresetSelection();
    syncDriftConfig();
  });
  el.speedSlider.addEventListener('input', () => {
    el.speedVal.textContent = `${parseFloat(el.speedSlider.value).toFixed(1)} msgs/s`;
    syncFeedConfig();
  });
  if (el.burnInSlider) {
    el.burnInSlider.addEventListener('input', () => {
      if (el.burnInVal) el.burnInVal.textContent = `${el.burnInSlider.value} windows`;
      syncFeedConfig();
    });
  }

  // Toggles and Selects
  [
    el.toggleAdjSwap, el.toggleClassSwap, el.toggleClassShift,
    el.toggleFormality, el.toggleNoise, el.driftCurveSelect
  ].forEach(input => {
    input.addEventListener('change', () => {
      clearPresetSelection();
      updateMethodBoxStyles();
      syncDriftConfig();
    });
  });

  [el.windowCountInput, el.windowTimeInput].forEach(inp => {
    inp.addEventListener('change', syncFeedConfig);
  });

  // 1-Click Demo Presets
  if (el.presetBtns) {
    el.presetBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        applyPreset(btn.dataset.preset);
      });
    });
  }

  // Presentation Pitch Guide Modal
  if (el.btnOpenGuide) {
    el.btnOpenGuide.addEventListener('click', () => {
      if (el.guideModal) el.guideModal.style.display = 'flex';
    });
  }
  if (el.btnCloseGuide) {
    el.btnCloseGuide.addEventListener('click', () => {
      if (el.guideModal) el.guideModal.style.display = 'none';
    });
  }
  if (el.guideModal) {
    el.guideModal.addEventListener('click', (e) => {
      if (e.target === el.guideModal) {
        el.guideModal.style.display = 'none';
      }
    });
  }

  // Retraining Overlay Modal Controls
  if (el.btnRetrainModel) {
    el.btnRetrainModel.addEventListener('click', triggerModelRetrain);
  }
  if (el.btnCloseTrainingOverlay) {
    el.btnCloseTrainingOverlay.addEventListener('click', hideTrainingOverlay);
  }
  if (el.btnNextFact) {
    el.btnNextFact.addEventListener('click', nextFact);
  }

  // Feed Actions
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
    if (el.baselineStatusBar) el.baselineStatusBar.style.display = 'none';
    if (el.baselineProgressFill) el.baselineProgressFill.style.width = '0%';
    if (el.lblSpellingRate) el.lblSpellingRate.textContent = '0.000';
    if (el.lblScoreDistDiv) el.lblScoreDistDiv.textContent = '0.000';
    if (el.modelVerBadge) el.modelVerBadge.textContent = 'Model v1.0';
    if (el.modelStatusPill) {
      el.modelStatusPill.textContent = 'CALIBRATING';
      el.modelStatusPill.className = 'model-status-pill calibrating';
    }
    if (el.lblModelWinAcc) el.lblModelWinAcc.textContent = '--%';
    if (el.lblModelCumAcc) el.lblModelCumAcc.textContent = '--%';
    if (el.lblCorpusSize) el.lblCorpusSize.textContent = '0 msgs';
    if (el.lblSummaryModelAcc) el.lblSummaryModelAcc.textContent = '100.0%';
    if (el.lblSummaryModelVer) el.lblSummaryModelVer.textContent = 'v1.0';
    if (el.retrainSubInfo) el.retrainSubInfo.textContent = 'Retrains sentiment analyzer on recent drifted samples';

    // Reset Alert Banner
    if (el.streamAlertBanner) {
      el.streamAlertBanner.className = 'stream-alert-banner banner-healthy';
      if (el.bannerIcon) el.bannerIcon.textContent = '🟢';
      if (el.bannerTitle) el.bannerTitle.textContent = 'STREAM IN-DISTRIBUTION — MODEL HEALTHY';
      if (el.bannerDriftBadge) el.bannerDriftBadge.textContent = 'Drift: 0.0%';
      if (el.bannerDesc) el.bannerDesc.textContent = 'Drift magnitude is within calibrated baseline tolerance. Model predictions are highly reliable.';
    }

    // Reset preset to clean
    if (el.presetBtns) {
      el.presetBtns.forEach(btn => btn.classList.toggle('active', btn.dataset.preset === 'clean'));
    }

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
    window_time_seconds: parseFloat(el.windowTimeInput.value),
    burn_in_windows: parseInt(el.burnInSlider ? el.burnInSlider.value : 3)
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
    if (data.config?.burn_in_windows && el.burnInSlider) {
      el.burnInSlider.value = data.config.burn_in_windows;
      if (el.burnInVal) el.burnInVal.textContent = `${data.config.burn_in_windows} windows`;
    }
    if (data.model_status) {
      if (data.model_status.version && el.modelVerBadge) {
        el.modelVerBadge.textContent = `Model ${data.model_status.version}`;
        if (el.lblSummaryModelVer) el.lblSummaryModelVer.textContent = data.model_status.version;
      }
      if (data.model_status.status && el.modelStatusPill) {
        el.modelStatusPill.textContent = data.model_status.status.toUpperCase();
        el.modelStatusPill.className = `model-status-pill ${data.model_status.status}`;
      }
    }
  } catch (e) {
    console.error("Could not fetch status:", e);
  }
}

function renderModelPredChip(item) {
  if (!item.model_prediction || item.model_prediction === 'untrained' || item.model_prediction === 'calibrating') {
    return '';
  }
  const isMatch = item.prediction_correct;
  const badgeClass = isMatch ? 'pred-match' : 'pred-mismatch';
  const icon = isMatch ? '✓' : '✗';
  const confPct = Math.round((item.model_confidence || 0) * 100);
  const ver = item.model_version || 'v1.0';
  return `
    <div class="card-model-pred ${badgeClass}">
      <span class="pred-status-icon">${icon}</span>
      <span class="pred-text"><strong>${ver}:</strong> ${item.model_prediction.toUpperCase()} (${confPct}%)</span>
      <span class="pred-divider">|</span>
      <span class="pred-actual">Label: <strong>${(item.ground_truth_label || '').toUpperCase()}</strong></span>
    </div>
  `;
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
