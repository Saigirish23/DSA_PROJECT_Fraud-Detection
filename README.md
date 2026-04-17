# Fraud Detection in Transaction Graphs (Elliptic + Dynamic Features)

This repository implements an end-to-end fraud detection pipeline that combines:

- graph-based heuristics,
- Graph Neural Networks (PyTorch Geometric),
- and hybrid fusion strategies.

It now supports the Elliptic Bitcoin dataset and a true incremental dynamic feature pipeline.

## What Is Current In This Repo

- Primary dataset path is Elliptic when raw Elliptic CSVs are available.
- Dynamic pipeline is enabled with:
  - `python main.py --dynamic`
- Training is capped to a maximum of 100 epochs in `src/train.py`.
- Dashboard reads the latest artifacts from `data/processed` and `outputs/results`.

## Pipeline Overview

1. Load dataset (Elliptic or canonical fallback) and build directed transaction graph.
2. Feature engineering:
     - static path (`python main.py`) or
     - incremental dynamic path (`python main.py --dynamic`).
3. Heuristic fraud scoring + pseudo labels.
4. PyG data preparation and train/val/test split.
5. GNN training.
6. Evaluation and hybrid comparisons:
     - Heuristic only
     - GNN only
     - Hybrid Strategy A (early fusion)
     - Hybrid Strategy B (late fusion)

## Quick Start

### Requirements

- Python 3.10+
- CUDA-capable GPU recommended (runs on CPU as well)

### Install

```bash
pip install -r requirements.txt
```

### Run

- Static features:

```bash
python main.py
```

- Dynamic incremental features:

```bash
python main.py --dynamic
```

### Launch Dashboard

```bash
python dashboard/dashboard_server.py
```

Then open: http://localhost:5000

## Current Result Snapshot (from outputs/results/final_metrics.csv)

| Model | Accuracy | Precision | Fraud Recall | F1 |
|---|---:|---:|---:|---:|
| Heuristic Only | 0.771459 | 0.023810 | 0.075472 | 0.036199 |
| GNN Only | 0.780758 | 0.116230 | 0.432390 | 0.183211 |
| Hybrid (Strategy A, early fusion) | 0.796779 | 0.168837 | 0.275660 | 0.209412 |
| Hybrid (Strategy B, late fusion) | 0.778970 | 0.115900 | 0.435535 | 0.183080 |

Best values from this run:

- Best F1: Hybrid Strategy A (0.209412)
- Best Fraud Recall: Hybrid Strategy B (0.435535)
- Best Accuracy: Hybrid Strategy A (0.796779)

## Repository Structure (Current)

```text
dsa_project/
|-- config.py
|-- main.py
|-- README.md
|-- requirements.txt
|-- train_elliptic.py
|
|-- cpp/
|   |-- Makefile
|   |-- graph_algorithms.cpp
|   |-- graph_algorithms.h
|   `-- graph_runner.py
|
|-- dashboard/
|   |-- dashboard.html
|   |-- dashboard_server.py
|   `-- static/
|       |-- charts.js
|       `-- style.css
|
|-- data/
|   |-- raw/
|   |   |-- account_ground_truth.csv
|   |   |-- transactions.csv
|   |   |-- elliptic_features.csv
|   |   |-- elliptic_edges.csv
|   |   |-- elliptic_labels.csv
|   |   `-- bitcoin/
|   |-- processed/
|   |   |-- node_features.csv
|   |   |-- node_features_full.csv
|   |   `-- labels.csv
|   `-- external/
|       `-- pyg_elliptic/
|
|-- models/
|   `-- best_gcn.pt
|
|-- outputs/
|   |-- plots/
|   `-- results/
|       |-- final_metrics.csv
|       |-- model_comparison.csv
|       |-- alpha_sweep_results.csv
|       |-- node_predictions.csv
|       `-- training_history.csv
|
`-- src/
      |-- data_loader.py
      |-- dynamic_graph.py
      |-- features.py
      |-- heuristics.py
      |-- gnn_model.py
      |-- train.py
      |-- evaluate.py
      |-- hybrid.py
      |-- bitcoin_loader.py
      |-- bitcoin_model.py
      |-- bitcoin_train.py
      |-- bitcoin_train_tuned.py
      |-- elliptic_loader.py
      `-- hparam_sweep.py
```

## Notes

- `data/external/` is ignored for GitHub push safety (large artifacts).
- Regeneratable artifacts are written to `data/processed/` and `outputs/results/`.
- If the dashboard fails to start because port 5000 is busy, stop the existing process and restart.
