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

function pickFirstColumn(keys, candidates, fallback = null) {
  for (const name of candidates) {
    if (keys.includes(name)) return name;
  }
  return fallback;
}

function pickFeatureColumns(sample) {
  if (!sample) {
    return {
      degree: "degree",
      inDegree: "in_degree",
      outDegree: "out_degree",
      clustering: "clustering",
      pagerank: "pagerank",
      betweenness: "betweenness",
      recentTx: "recent_transaction_sum",
      nodeId: "node_id",
    };
  }

  const keys = Object.keys(sample);
  return {
    degree: pickFirstColumn(keys, ["degree", "total_degree"], "degree"),
    inDegree: pickFirstColumn(keys, ["in_degree"], null),
    outDegree: pickFirstColumn(keys, ["out_degree"], null),
    clustering: pickFirstColumn(keys, ["clustering", "clustering_coefficient"], "clustering"),
    pagerank: pickFirstColumn(keys, ["pagerank", "page_rank"], "pagerank"),
    betweenness: pickFirstColumn(keys, ["betweenness", "betweenness_centrality"], null),
    recentTx: pickFirstColumn(keys, ["recent_transaction_sum", "tx_count_window"], null),
    nodeId: pickFirstColumn(keys, ["node_id", "account_id"], keys[0] || "node_id"),
  };
}

function pickAuxFeature(cols) {
  if (cols.recentTx) {
    return {
      key: cols.recentTx,
      label: cols.recentTx === "recent_transaction_sum" ? "Recent Tx Sum" : "Tx Count Window",
      format: cols.recentTx === "recent_transaction_sum" ? "amount" : "count",
      color: "rgba(159,255,84,0.75)",
    };
  }

  if (cols.betweenness) {
    return {
      key: cols.betweenness,
      label: "Betweenness",
      format: "score",
      color: "rgba(0,229,255,0.75)",
    };
  }

  if (cols.outDegree) {
    return {
      key: cols.outDegree,
      label: "Out Degree",
      format: "count",
      color: "rgba(0,229,255,0.75)",
    };
  }

  return {
    key: cols.degree,
    label: "Degree",
    format: "count",
    color: "rgba(0,229,255,0.75)",
  };
}

function statusBadge(score) {
  if (score >= 0.55) return '<span class="badge fraud">FRAUD</span>';
  if (score >= 0.45) return '<span class="badge warn">SUSPICIOUS</span>';
  return '<span class="badge safe">NORMAL</span>';
}

function setNodeTableHeaders(auxFeature) {
  const auxHeader = document.getElementById("auxFeatureHeader");
  if (auxHeader) {
    auxHeader.textContent = auxFeature?.label || "Aux Feature";
  }

  const auxLegend = document.getElementById("auxLegendLabel");
  if (auxLegend) {
    auxLegend.textContent = auxFeature?.label || "Aux Feature";
  }
}

function formatAuxValue(value, fmt) {
  if (fmt === "count") return num(value).toFixed(0);
  if (fmt === "amount") return num(value).toFixed(2);
  return num(value).toFixed(6);
}

function computeNodeRiskScore(nodeId, labelsMap, predictionMap) {
  const label = labelsMap.get(nodeId);
  if (Number.isFinite(label) && label >= 0) {
    return label >= 0.5 ? 0.7 : 0.2;
  }

  const predictedProb = predictionMap.get(nodeId);
  if (Number.isFinite(predictedProb)) {
    return predictedProb;
  }

  return 0.2;
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
    return;
  }

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
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;

  const total = Math.max(fraud + normal, 1);
  chartRegistry.histCanvas = new Chart(canvas, {
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

function renderNodeTable(features, cols, labelsMap, predictionMap, auxFeature) {
  const tbody = document.getElementById("nodeTableBody");
  if (!tbody) return;
  tbody.innerHTML = "";

  if (!features || !features.length) {
    tbody.innerHTML = '<tr><td colspan="8">No node features available.</td></tr>';
    return;
  }

  const auxKey = auxFeature?.key || cols.degree;
  const sorted = [...features]
    .sort((a, b) => num(b[cols.degree]) - num(a[cols.degree]))
    .slice(0, 25);

  sorted.forEach((n) => {
    const nodeId = String(n[cols.nodeId]);
    const inDegreeVal = cols.inDegree ? num(n[cols.inDegree]) : 0;
    const outDegreeVal = cols.outDegree ? num(n[cols.outDegree]) : 0;
    const score = computeNodeRiskScore(nodeId, labelsMap, predictionMap);

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="font-family:'Space Mono';color:#fff">${nodeId}</td>
      <td>${num(n[cols.degree]).toFixed(0)}</td>
      <td>${inDegreeVal.toFixed(0)}</td>
      <td>${outDegreeVal.toFixed(0)}</td>
      <td>${num(n[cols.clustering]).toFixed(4)}</td>
      <td>${num(n[cols.pagerank]).toFixed(6)}</td>
      <td>${formatAuxValue(n[auxKey], auxFeature?.format || "score")}</td>
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

  chartRegistry.trainingLossCanvas = new Chart(canvas, {
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
        x: {
          title: { display: true, text: "Epoch", color: "#4a6080" },
          ticks: { color: "#4a6080" },
          grid: { color: "rgba(30,45,74,0.3)" },
        },
        y: {
          title: { display: true, text: "Loss", color: "#4a6080" },
          ticks: { color: "#4a6080" },
          grid: { color: "rgba(30,45,74,0.3)" },
        },
      },
    },
  });

  fallback.textContent = "";
}

function renderMetricsKPI(metrics) {
  if (!metrics || !metrics.length) {
    return;
  }

  const metricDict = {};
  metrics.forEach((m) => {
    if (m.metric && m.value) {
      metricDict[m.metric] = num(m.value);
    }
  });

  const kpis = [
    { label: "Accuracy", key: "accuracy", format: ".2%" },
    { label: "Precision", key: "precision", format: ".2%" },
    { label: "Recall", key: "recall", format: ".2%" },
    { label: "F1-Score", key: "f1", format: ".4f" },
    { label: "ROC-AUC", key: "roc_auc", format: ".4f" },
  ];

  const metricsContainer = document.getElementById("metricsContainer");
  if (!metricsContainer) return;

  metricsContainer.innerHTML = kpis.map((kpi) => {
    const value = metricDict[kpi.key] ?? 0;
    const formatted = (
      kpi.format === ".2%"
        ? `${(value * 100).toFixed(1)}%`
        : kpi.format === ".4f"
          ? value.toFixed(4)
          : value.toFixed(2)
    );

    return `
      <div class="card metric-kpi">
        <div class="card-label">${kpi.label}</div>
        <div class="card-value">${formatted}</div>
      </div>
    `;
  }).join("");
}

function renderTopRiskyNodes(predictions) {
  const tbody = document.getElementById("riskyNodesTableBody");
  if (!tbody) return;
  tbody.innerHTML = "";

  if (!predictions || !predictions.length) {
    tbody.innerHTML = '<tr><td colspan="4">No prediction data available.</td></tr>';
    return;
  }

  const sorted = [...predictions]
    .sort((a, b) => num(b.fraud_probability) - num(a.fraud_probability))
    .slice(0, 10);

  sorted.forEach((p, idx) => {
    const tr = document.createElement("tr");
    const fraudProb = num(p.fraud_probability);
    const badge = fraudProb >= 0.6
      ? '<span class="badge fraud">HIGH RISK</span>'
      : fraudProb >= 0.4
        ? '<span class="badge warn">MEDIUM</span>'
        : '<span class="badge safe">LOW</span>';

    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td style="font-family:'Space Mono';color:#fff">${String(p.node_id || "-")}</td>
      <td>${(fraudProb * 100).toFixed(2)}%</td>
      <td>${badge}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function initDashboard() {
  const [metrics, features, labels, cppStatus, trainingHistory, predictions, graphStats, featureMetadata] = await Promise.all([
    fetchJson("/api/metrics"),
    fetchJson("/api/features?limit=1200"),
    fetchJson("/api/labels"),
    fetchJson("/api/cpp_status"),
    fetchJson("/api/training_history"),
    fetchJson("/api/predictions"),
    fetchJson("/api/graph_stats"),
    fetchJson("/api/feature_metadata"),
  ]);

  const featureRows = Array.isArray(features) ? features : [];
  const metricRows = Array.isArray(metrics) ? metrics : [];
  const historyRows = Array.isArray(trainingHistory) ? trainingHistory : [];
  const predictionRows = Array.isArray(predictions) ? predictions : [];
  const metadata = featureMetadata && typeof featureMetadata === "object" ? featureMetadata : {};

  const fraud = num(labels?.fraud);
  const normal = num(labels?.normal);
  const totalNodes = graphStats?.nodes || featureRows.length;
  const totalLabels = fraud + normal;
  const fraudRatio = totalLabels > 0 ? ((fraud * 100) / totalLabels).toFixed(2) : "0.00";

  const statusEl = document.getElementById("cppStatusText");
  if (statusEl) {
    statusEl.textContent = cppStatus?.available ? "C++ Backend: Active" : "C++ Backend: Not Compiled";
  }

  const inferredDynamic = featureRows.length > 0 && Object.prototype.hasOwnProperty.call(featureRows[0], "recent_transaction_sum");
  const isDynamic = metadata.mode === "dynamic" || inferredDynamic;

  const modeEl = document.getElementById("pipelineModeText");
  if (modeEl) {
    modeEl.textContent = isDynamic ? "Pipeline: Dynamic" : "Pipeline: Static";
  }

  const titleEl = document.getElementById("suspiciousSectionTitle");
  if (titleEl) {
    titleEl.textContent = isDynamic
      ? "Top Suspicious Nodes — Dynamic + Structural Signals"
      : "Top Suspicious Nodes — Structural Signals";
  }

  const totalNodesEl = document.getElementById("totalNodes");
  const fraudCountEl = document.getElementById("fraudCount");
  const ratioEl = document.getElementById("cycleCount");
  const bestF1El = document.getElementById("aucScore");
  if (totalNodesEl) totalNodesEl.textContent = String(totalNodes || normal + fraud || 0);
  if (fraudCountEl) fraudCountEl.textContent = String(fraud);
  if (ratioEl) ratioEl.textContent = `${fraudRatio}%`;

  let bestF1 = 0;
  metricRows.forEach((m) => {
    bestF1 = Math.max(bestF1, num(m.f1));
  });
  if (bestF1El) bestF1El.textContent = bestF1 ? bestF1.toFixed(3) : "—";

  const alert = document.getElementById("alertText");
  if (alert) {
    const modeText = isDynamic ? "Dynamic pipeline active" : "Static pipeline active";
    if (metricRows.length) {
      const f1MetricRow = metricRows.find((m) => m.metric === "f1");
      const f1Score = f1MetricRow
        ? num(f1MetricRow.value).toFixed(3)
        : bestF1
          ? bestF1.toFixed(3)
          : "N/A";
      alert.innerHTML = `<strong>${modeText}</strong> — F1 Score: ${f1Score} | Risky nodes detected: ${predictionRows.length}`;
    } else {
      alert.innerHTML = `<strong>${modeText}</strong> — No metric files found. Run the Python pipeline to generate dashboard data.`;
    }
  }

  renderDonut(fraud, normal);
  renderModelTable(metricRows);
  renderTrainingLoss(historyRows);
  renderMetricsKPI(metricRows);
  renderTopRiskyNodes(predictionRows);

  if (!featureRows.length) {
    setNodeTableHeaders({ label: "Aux Feature" });
    ["degreeChart", "clusteringChart", "pagerankChart", "betweennessChart"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) {
        const ctx = el.getContext("2d");
        ctx.fillStyle = "#4a6080";
        ctx.font = "12px Space Mono";
        ctx.fillText("No feature data available", 16, 24);
      }
    });
    renderNodeTable([], pickFeatureColumns(null), new Map(), new Map(), { key: "degree", label: "Aux Feature", format: "count" });
    return;
  }

  const cols = pickFeatureColumns(featureRows[0]);
  const auxFeature = pickAuxFeature(cols);
  const sample = featureRows.slice(0, 30);
  const labelsByNode = new Map();
  const predictionByNode = new Map();

  setNodeTableHeaders(auxFeature);

  predictionRows.forEach((p) => {
    const nodeId = String(p.node_id ?? "");
    if (!nodeId) return;
    predictionByNode.set(nodeId, num(p.fraud_probability, NaN));
  });

  const labelColCandidates = ["heuristic_label", "label", "is_fraud"];
  featureRows.forEach((r) => {
    const nodeId = String(r[cols.nodeId]);
    let labelValue = null;
    for (const c of labelColCandidates) {
      if (Object.prototype.hasOwnProperty.call(r, c)) {
        labelValue = num(r[c], NaN);
        break;
      }
    }
    if (Number.isFinite(labelValue)) {
      labelsByNode.set(nodeId, labelValue);
    }
  });

  renderFeatureChart("degreeChart", sample.map((r) => num(r[cols.degree])), "rgba(255,62,91,0.75)", "Degree Distribution");
  renderFeatureChart("clusteringChart", sample.map((r) => num(r[cols.clustering])), "rgba(255,176,32,0.75)", "Clustering Distribution");
  renderFeatureChart("pagerankChart", sample.map((r) => num(r[cols.pagerank])), "rgba(0,245,160,0.75)", "PageRank Distribution");
  renderFeatureChart(
    "betweennessChart",
    sample.map((r) => num(r[auxFeature.key])),
    auxFeature.color,
    `${auxFeature.label} Distribution`
  );

  renderNodeTable(featureRows, cols, labelsByNode, predictionByNode, auxFeature);
}

document.querySelectorAll(".algo-chip").forEach((chip) => {
  chip.addEventListener("click", () => chip.classList.toggle("active"));
});

initDashboard();
