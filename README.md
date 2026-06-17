# 🚨 Fraud Detection in Transaction Graphs (Elliptic + Dynamic Features)

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python\&logoColor=white)
![C++](https://img.shields.io/badge/C++-00599C?logo=cplusplus\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch\&logoColor=white)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-GNN-orange)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Analytics-green)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas\&logoColor=white)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikitlearn\&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?logo=nvidia\&logoColor=white)

</p>

---

## 📖 Overview

This repository implements an end-to-end fraud detection pipeline that combines:

* 🕸️ Graph-based Heuristics
* 🤖 Graph Neural Networks (PyTorch Geometric)
* 🔗 Hybrid Fusion Strategies

The system supports the **Elliptic Bitcoin Dataset** and a **Dynamic Graph Feature Pipeline** for fraud detection in evolving transaction networks.

---

## ⚙️ Current Implementation

* ✅ Primary dataset path is **Elliptic** when raw Elliptic CSVs are available
* ✅ Dynamic pipeline enabled via:

```bash
python main.py --dynamic
```

* ✅ Training capped to a maximum of **100 epochs** in `src/train.py`
* ✅ Dashboard reads latest artifacts from:

  * `data/processed`
  * `outputs/results`

---

## 🏗️ Pipeline Overview

```text
1. Load dataset (Elliptic or canonical fallback)
2. Build directed transaction graph
3. Feature Engineering
      ├─ Static Path (python main.py)
      └─ Dynamic Path (python main.py --dynamic)
4. Heuristic Fraud Scoring + Pseudo Labels
5. PyTorch Geometric Data Preparation
6. Train / Validation / Test Split
7. GNN Training
8. Evaluation & Hybrid Comparisons
```

### Evaluated Approaches

* 📌 Heuristic Only
* 📌 GNN Only
* 📌 Hybrid Strategy A (Early Fusion)
* 📌 Hybrid Strategy B (Late Fusion)

---

## 🚀 Quick Start

### Requirements

* Python 3.10+
* CUDA-capable GPU (recommended)
* Linux / Windows / macOS

### Installation

```bash
pip install -r requirements.txt
```

### Run Static Pipeline

```bash
python main.py
```

### Run Dynamic Incremental Pipeline

```bash
python main.py --dynamic
```

---

## 📊 Dashboard

Launch the dashboard:

```bash
python dashboard/dashboard_server.py
```

Open:

```text
http://localhost:5000
```

---

## 📈 Current Result Snapshot

*(Generated from `outputs/results/final_metrics.csv`)*

| Model                             |     Accuracy |    Precision | Fraud Recall |           F1 |
| --------------------------------- | -----------: | -----------: | -----------: | -----------: |
| Heuristic Only                    |     0.771459 |     0.023810 |     0.075472 |     0.036199 |
| GNN Only                          |     0.780758 |     0.116230 |     0.432390 |     0.183211 |
| Hybrid (Strategy A, Early Fusion) | **0.796779** | **0.168837** |     0.275660 | **0.209412** |
| Hybrid (Strategy B, Late Fusion)  |     0.778970 |     0.115900 | **0.435535** |     0.183080 |

### 🏆 Best Results

| Metric            | Model             |
| ----------------- | ----------------- |
| Best Accuracy     | Hybrid Strategy A |
| Best Precision    | Hybrid Strategy A |
| Best F1 Score     | Hybrid Strategy A |
| Best Fraud Recall | Hybrid Strategy B |

---

## 📂 Repository Structure

```text
dsa_project/
│
├── config.py
├── main.py
├── README.md
├── requirements.txt
├── train_elliptic.py
│
├── cpp/
│   ├── Makefile
│   ├── graph_algorithms.cpp
│   ├── graph_algorithms.h
│   └── graph_runner.py
│
├── dashboard/
│   ├── dashboard.html
│   ├── dashboard_server.py
│   └── static/
│       ├── charts.js
│       └── style.css
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── models/
│   └── best_gcn.pt
│
├── outputs/
│   ├── plots/
│   └── results/
│
└── src/
    ├── data_loader.py
    ├── dynamic_graph.py
    ├── features.py
    ├── heuristics.py
    ├── gnn_model.py
    ├── train.py
    ├── evaluate.py
    ├── hybrid.py
    ├── bitcoin_loader.py
    ├── bitcoin_model.py
    ├── bitcoin_train.py
    ├── bitcoin_train_tuned.py
    ├── elliptic_loader.py
    └── hparam_sweep.py
```

---

## 📝 Notes

* `data/external/` is ignored for GitHub push safety due to large artifacts.
* Regeneratable artifacts are written to:

  * `data/processed/`
  * `outputs/results/`
* If the dashboard fails to start because port **5000** is already in use, terminate the existing process and restart the server.

