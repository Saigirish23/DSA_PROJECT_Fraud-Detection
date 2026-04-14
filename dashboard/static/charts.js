async function fetchJson(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

const chartRegistry = {};

function num(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function pickFeatureColumns(sample) {
  if (!sample) {
    return {
      degree: "total_degree",
      clustering: "clustering_coefficient",
      pagerank: "pagerank",
      betweenness: "betweenness_centrality",
      nodeId: "node_id",
    };
  }

  const keys = Object.keys(sample);
  return {
    degree: keys.includes("total_degree") ? "total_degree" : "degree",
    clustering: keys.includes("clustering_coefficient") ? "clustering_coefficient" : "clustering",
    pagerank: keys.includes("pagerank") ? "pagerank" : "page_rank",
    betweenness: keys.includes("betweenness_centrality") ? "betweenness_centrality" : "betweenness",
    nodeId: keys.includes("node_id") ? "node_id" : keys[0],
  };
}

function statusBadge(score) {
  if (score >= 0.55) return '<span class="badge fraud">FRAUD</span>';
  if (score >= 0.45) return '<span class="badge warn">SUSPICIOUS</span>';
  return '<span class="badge safe">NORMAL</span>';
}

function destroyChartIfExists(canvasId) {
  if (chartRegistry[canvasId]) {
    chartRegistry[canvasId].destroy();
    chartRegistry[canvasId] = null;
  }
}

function buildHistogram(values, bins = 20) {
  const clean = values.filter((v) => Number.isFinite(v));
  if (!clean.length) {
    return { labels: [], counts: [] };
  }

  const min = Math.min(...clean);
  const max = Math.max(...clean);

  // Handle constant-valued features gracefully.
  if (Math.abs(max - min) < 1e-12) {
    return {
      labels: [min.toFixed(4)],
      counts: [clean.length],
    };
  }

  const width = (max - min) / bins;
  const counts = Array.from({ length: bins }, () => 0);

  for (const v of clean) {
    const rawIdx = Math.floor((v - min) / width);
    const idx = Math.max(0, Math.min(bins - 1, rawIdx));
    counts[idx] += 1;
  }

  const labels = counts.map((_, i) => {
    const left = min + i * width;
    const right = left + width;
    return `${left.toFixed(4)}-${right.toFixed(4)}`;
  });

  return { labels, counts };
}

function renderFeatureChart(canvasId, values, color, title) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) {
    console.warn(`Canvas element #${canvasId} not found`);
    return;
  }

  destroyChartIfExists(canvasId);
  
  // Set canvas pixel dimensions for proper rendering
  const width = ctx.offsetWidth || ctx.parentElement?.offsetWidth || 300;
  const height = ctx.offsetHeight || 180;
  ctx.width = width;
  ctx.height = height;
  
  const hist = buildHistogram(values, 16);

  if (!hist.labels.length) {
    const fallbackCtx = ctx.getContext("2d");
    fallbackCtx.fillStyle = "#4a6080";
    fallbackCtx.font = "12px Space Mono";
    fallbackCtx.fillText("No feature data available", 16, 24);
    console.warn(`No histogram data for ${canvasId}`);
    return;
  }
  
  console.log(`Rendering ${canvasId}: ${values.length} values, ${hist.labels.length} bins`);

  chartRegistry[canvasId] = new Chart(ctx, {
    type: "bar",
    data: {
      labels: hist.labels,
      datasets: [{
        label: title,
        data: hist.counts,
        backgroundColor: color,
        borderColor: color,
        borderWidth: 1,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        title: { display: true, text: title, color: "#c8d8f0" },
      },
      scales: {
        x: {
          ticks: { color: "#4a6080", maxTicksLimit: 6 },
          grid: { color: "rgba(30,45,74,0.3)" },
          title: { display: true, text: "Value Range", color: "#4a6080" },
        },
        y: {
          ticks: { color: "#4a6080" },
          grid: { color: "rgba(30,45,74,0.3)" },
          title: { display: true, text: "Count", color: "#4a6080" },
        },
      },
    },
  });
}

function renderDonut(fraud, normal) {
  const canvas = document.getElementById("histCanvas");
  if (!canvas) return;

  destroyChartIfExists("histCanvas");
  
  // Set canvas pixel dimensions for proper rendering
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  
  const total = Math.max(fraud + normal, 1);
  chartRegistry["histCanvas"] = new Chart(canvas, {
    type: "doughnut",
    data: {
      labels: ["Fraud", "Normal"],
      datasets: [{
        data: [fraud, normal],
        backgroundColor: ["#ff3e5b", "#00a6ff"],
        borderColor: ["#ff3e5b", "#00a6ff"],
      }],
    },
    options: {
      plugins: {
        legend: { labels: { color: "#c8d8f0" } },
      },
    },
  });

  const fraudPct = ((fraud * 100) / total).toFixed(2);
  const normalPct = ((normal * 100) / total).toFixed(2);
  const stats = document.getElementById("donutStats");
  if (stats) {
    stats.textContent = `Fraud: ${fraud} (${fraudPct}%) | Normal: ${normal} (${normalPct}%)`;
  }
}

function renderModelTable(metrics) {
  const tbody = document.getElementById("modelTableBody");
  if (!tbody) return;
  tbody.innerHTML = "";

  if (!metrics || !metrics.length) {
    tbody.innerHTML = '<tr><td colspan="6">No model metrics available.</td></tr>';
    return;
  }

  let bestF1 = -1;
  metrics.forEach((m) => {
    bestF1 = Math.max(bestF1, num(m.f1));
  });

  metrics.forEach((m) => {
    const tr = document.createElement("tr");
    if (Math.abs(num(m.f1) - bestF1) < 1e-12) {
      tr.style.background = "rgba(0,245,160,0.12)";
    }
    tr.innerHTML = `
      <td>${m.Model ?? m.model ?? "-"}</td>
      <td>${num(m.accuracy).toFixed(4)}</td>
      <td>${num(m.precision).toFixed(4)}</td>
      <td>${num(m.recall).toFixed(4)}</td>
      <td>${num(m.f1).toFixed(4)}</td>
      <td>${num(m.roc_auc).toFixed(4)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderNodeTable(features, cols, labelsMap) {
  const tbody = document.getElementById("nodeTableBody");
  if (!tbody) return;
  tbody.innerHTML = "";

  if (!features || !features.length) {
    tbody.innerHTML = '<tr><td colspan="6">No node features available.</td></tr>';
    return;
  }

  const sorted = [...features].sort((a, b) => num(b[cols.degree]) - num(a[cols.degree])).slice(0, 25);
  sorted.forEach((n) => {
    const nodeId = String(n[cols.nodeId]);
    const label = labelsMap.get(nodeId) ?? 0;
    const score = label ? 0.7 : 0.2;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="font-family:'Space Mono';color:#fff">${nodeId}</td>
      <td>${num(n[cols.degree]).toFixed(0)}</td>
      <td>${num(n[cols.clustering]).toFixed(4)}</td>
      <td>${num(n[cols.pagerank]).toFixed(6)}</td>
      <td>${num(n[cols.betweenness]).toFixed(6)}</td>
      <td>${statusBadge(score)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderTrainingLoss(history) {
  const canvas = document.getElementById("trainingLossCanvas");
  const fallback = document.getElementById("trainingFallback");
  if (!canvas || !fallback) return;

  destroyChartIfExists("trainingLossCanvas");
  
  // Set canvas pixel dimensions for proper rendering
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;

  if (!history || !history.length) {
    fallback.textContent = "Training data not available.";
    return;
  }

  const first = history[0] || {};
  let lossCol = null;
  if (Object.prototype.hasOwnProperty.call(first, "train_loss")) lossCol = "train_loss";
  if (!lossCol && Object.prototype.hasOwnProperty.call(first, "loss")) lossCol = "loss";

  if (!lossCol) {
    fallback.textContent = "Training data not available.";
    return;
  }

  const losses = history.map((r) => num(r[lossCol]));
  const labels = losses.map((_, i) => i + 1);

  chartRegistry["trainingLossCanvas"] = new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Training Loss",
        data: losses,
        borderColor: "#00e5ff",
        backgroundColor: "rgba(0,229,255,0.15)",
        fill: true,
        tension: 0.25,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#c8d8f0" } } },
      scales: {
        x: { title: { display: true, text: "Epoch", color: "#4a6080" }, ticks: { color: "#4a6080" }, grid: { color: "rgba(30,45,74,0.3)" } },
        y: { title: { display: true, text: "Loss", color: "#4a6080" }, ticks: { color: "#4a6080" }, grid: { color: "rgba(30,45,74,0.3)" } },
      },
    },
  });

  fallback.textContent = "";
}

async function initDashboard() {
  console.log('Dashboard initialization started');
  const [metrics, features, labels, cppStatus, trainingHistory] = await Promise.all([
    fetchJson("/api/metrics"),
    fetchJson("/api/features"),
    fetchJson("/api/labels"),
    fetchJson("/api/cpp_status"),
    fetchJson("/api/training_history"),
  ]);

  console.log('API data loaded:', {
    metrics: metrics?.length || 0,
    features: features?.length || 0,
    labels,
    cppStatus,
    trainingHistory: trainingHistory?.length || 0
  });

  const featureRows = Array.isArray(features) ? features : [];
  const metricRows = Array.isArray(metrics) ? metrics : [];
  const historyRows = Array.isArray(trainingHistory) ? trainingHistory : [];

  const fraud = num(labels?.fraud);
  const normal = num(labels?.normal);
  const totalNodes = featureRows.length;
  const totalLabels = fraud + normal;
  const fraudRatio = totalLabels > 0 ? ((fraud * 100) / totalLabels).toFixed(2) : "0.00";

  const statusEl = document.getElementById("cppStatusText");
  if (statusEl) {
    statusEl.textContent = cppStatus?.available ? "C++ Backend: Active" : "C++ Backend: Not Compiled";
  }

  const totalNodesEl = document.getElementById("totalNodes");
  const fraudCountEl = document.getElementById("fraudCount");
  const ratioEl = document.getElementById("cycleCount");
  const bestF1El = document.getElementById("aucScore");
  if (totalNodesEl) totalNodesEl.textContent = String(totalNodes || normal + fraud || 0);
  if (fraudCountEl) fraudCountEl.textContent = String(fraud);
  if (ratioEl) ratioEl.textContent = `${fraudRatio}%`;

  let bestF1 = 0;
  metricRows.forEach((m) => { bestF1 = Math.max(bestF1, num(m.f1)); });
  if (bestF1El) bestF1El.textContent = bestF1 ? bestF1.toFixed(3) : "—";

  const alert = document.getElementById("alertText");
  if (alert) {
    if (metricRows.length) {
      const best = metricRows.reduce((acc, cur) => (num(cur.f1) > num(acc.f1) ? cur : acc), metricRows[0]);
      alert.innerHTML = `<strong>${fraud} suspicious nodes detected</strong> — Best model: ${best.Model ?? best.model ?? "N/A"} with F1 ${(num(best.f1)).toFixed(3)}`;
    } else {
      alert.innerHTML = "<strong>No metric files found</strong> — Run the Python pipeline to generate dashboard data.";
    }
  }

  renderDonut(fraud, normal);
  renderModelTable(metricRows);
  renderTrainingLoss(historyRows);

  if (!featureRows.length) {
    ["degreeChart", "clusteringChart", "pagerankChart", "betweennessChart"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) {
        const ctx = el.getContext("2d");
        ctx.fillStyle = "#4a6080";
        ctx.font = "12px Space Mono";
        ctx.fillText("No feature data available", 16, 24);
      }
    });
    renderNodeTable([], pickFeatureColumns(null), new Map());
    return;
  }

  const cols = pickFeatureColumns(featureRows[0]);
  const sample = featureRows.slice(0, 30);
  const labelsByNode = new Map();

  if (Array.isArray(features)) {
    const labelColCandidates = ["heuristic_label", "label", "is_fraud"];
    featureRows.forEach((r) => {
      const nodeId = String(r[cols.nodeId]);
      let l = null;
      for (const c of labelColCandidates) {
        if (Object.prototype.hasOwnProperty.call(r, c)) {
          l = num(r[c]);
          break;
        }
      }
      if (l !== null) labelsByNode.set(nodeId, l);
    });
  }

  renderFeatureChart("degreeChart", sample.map((r) => num(r[cols.degree])), "rgba(255,62,91,0.75)", "Degree Distribution");
  renderFeatureChart("clusteringChart", sample.map((r) => num(r[cols.clustering])), "rgba(255,176,32,0.75)", "Clustering Distribution");
  renderFeatureChart("pagerankChart", sample.map((r) => num(r[cols.pagerank])), "rgba(0,245,160,0.75)", "PageRank Distribution");
  renderFeatureChart("betweennessChart", sample.map((r) => num(r[cols.betweenness])), "rgba(0,229,255,0.75)", "Betweenness Distribution");

  renderNodeTable(featureRows, cols, labelsByNode);
}

document.querySelectorAll(".algo-chip").forEach((chip) => {
  chip.addEventListener("click", () => chip.classList.toggle("active"));
});

initDashboard();
