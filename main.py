
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

import config
from src.data_loader import load_dataset, build_graph, set_seeds, build_pyg_data
from src.dynamic_graph import DynamicFraudGraph, CANONICAL_FEATURE_COLUMNS
from src.evaluate import run_full_evaluation
from src.features import compute_all_features
from src.heuristics import compute_fraud_scores, generate_heuristic_labels
from src.train import train_model

logger = config.setup_logging(__name__)


def _parse_args():
    parser = argparse.ArgumentParser(description="Fraud detection pipeline runner")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Use true incremental dynamic pipeline instead of static features",
    )
    parser.add_argument(
        "--bitcoin",
        action="store_true",
        help="Run dedicated Bitcoin/Elliptic training workflow",
    )
    parser.add_argument(
        "--dynamic-window-days",
        type=int,
        default=7,
        help="Sliding window size in days for dynamic graph updates",
    )
    parser.add_argument(
        "--dynamic-snapshot-stride",
        type=int,
        default=5000,
        help="Number of transactions between dynamic feature snapshots",
    )
    return parser.parse_args()


def _build_dynamic_snapshot_features(df, window_days=7, snapshot_stride=5000):
    """
    Phase 7: True incremental dynamic feature pipeline.

    This function processes transactions ONE BY ONE through the DynamicFraudGraph.
    NO global recomputation (no nx.pagerank, no nx.clustering) is ever called.

    Flow:
        1. Sort transactions by time
        2. For each transaction: incrementally update the dynamic graph
        3. Every `snapshot_stride` transactions: snapshot features (pure lookup)
        4. Aggregate snapshots into final feature matrix

    All features are maintained incrementally inside DynamicFraudGraph:
        - degree:       O(1) per update
        - clustering:   O(min(deg(u), deg(v))) per update — triangle counting
        - pagerank:     O(k × avg_deg) per update — local relaxation
        - recent_tx_sum: O(log n) per update — Fenwick tree
    """
    required_cols = {"sender_id", "receiver_id"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            "Dynamic pipeline requires sender_id and receiver_id columns. Found: {}".format(
                sorted(df.columns)
            )
        )

    work = df.copy()
    work["sender_id"] = work["sender_id"].astype(str)
    work["receiver_id"] = work["receiver_id"].astype(str)

    if "amount" not in work.columns:
        work["amount"] = 0.0
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce").fillna(0.0)

    if "timestamp" in work.columns:
        parsed_ts = pd.to_datetime(work["timestamp"], errors="coerce")
        parsed_ts = parsed_ts.fillna(pd.Timestamp("2023-01-01"))
        work["timestamp_int"] = parsed_ts.astype("int64") // 10**9
    else:
        work["timestamp_int"] = np.arange(len(work), dtype=np.int64)

    work = work.sort_values(["timestamp_int", "sender_id", "receiver_id"], kind="mergesort").reset_index(drop=True)

    window_seconds = max(1, int(window_days) * 24 * 60 * 60)
    stride = max(1, int(snapshot_stride))
    dg = DynamicFraudGraph(window_size=window_seconds)

    snapshots = []
    total = len(work)

    t_start = time.time()

    for i, row in enumerate(work.itertuples(index=False), start=1):
        # Phase 7: Each transaction is processed incrementally.
        # Inside add_transaction:
        #   - degree updated: O(1)
        #   - triangles/clustering updated: O(min(deg(u), deg(v)))
        #   - PageRank locally relaxed: O(k × avg_deg)
        #   - Fenwick tree updated: O(log n)
        #   - Expired edges removed: O(1) amortized
        dg.add_transaction(
            sender=getattr(row, "sender_id"),
            receiver=getattr(row, "receiver_id"),
            amount=float(getattr(row, "amount", 0.0)),
            timestamp=int(getattr(row, "timestamp_int", 0)),
        )

        if i % stride == 0 or i == total:
            # Snapshot: pure lookup, NO recomputation triggered
            snap = dg.get_all_features()
            if not snap.empty:
                snap["snapshot_index"] = len(snapshots)
                snapshots.append(snap)

    t_elapsed = time.time() - t_start
    logger.info("  Dynamic incremental processing: %.2f seconds for %d transactions", t_elapsed, total)

    if not snapshots:
        return pd.DataFrame(columns=["node_id"] + CANONICAL_FEATURE_COLUMNS)

    stacked = pd.concat(snapshots, ignore_index=True)
    stacked["node_id"] = stacked["node_id"].astype(str)
    agg = (
        stacked.groupby("node_id", as_index=False)[CANONICAL_FEATURE_COLUMNS]
        .mean()
        .reset_index(drop=True)
    )
    return agg


def main():
    args = _parse_args()

    if args.bitcoin:
        logger.info("Running Bitcoin/Elliptic workflow (--bitcoin)")
        from src.bitcoin_train import train_on_bitcoin

        train_on_bitcoin()
        return

    logger.info("=" * 60)
    logger.info("FRAUD DETECTION IN TRANSACTION GRAPHS (END-TO-END)")
    logger.info("=" * 60)

    config.ensure_dirs()
    set_seeds()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Running on device: %s", device)

    logger.info("\n>>> PHASE 2: Data Loading & Graph Construction")
    df, account_labels = load_dataset()
    G = build_graph(df)

    logger.info("\n>>> PHASE 3: Feature Engineering")
    if args.dynamic:
        logger.info("Using TRUE INCREMENTAL dynamic features (--dynamic)")
        logger.info("  No NetworkX recomputation — all features maintained incrementally")
        features_df = _build_dynamic_snapshot_features(
            df,
            window_days=args.dynamic_window_days,
            snapshot_stride=args.dynamic_snapshot_stride,
        )
        all_nodes_df = pd.DataFrame({"node_id": [str(n) for n in sorted(G.nodes())]})
        features_df = all_nodes_df.merge(features_df, on="node_id", how="left")
        for col in CANONICAL_FEATURE_COLUMNS:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce").fillna(0.0)
        features_df.to_csv(config.NODE_FEATURES_PATH, index=False)
        logger.info(
            "  Dynamic snapshots stacked into canonical features: %d nodes, %d columns",
            len(features_df),
            len(features_df.columns),
        )
    else:
        logger.info("Using static feature pipeline (default)")
        features_df = compute_all_features(G)

    logger.info("\n>>> PHASE 4: Rule-based Heuristics")
    scored_df = compute_fraud_scores(features_df)
    labels_df = generate_heuristic_labels(scored_df)

    logger.info("\n>>> PHASE 5: PyG Data Preparation")
    data, scaler, node_to_idx = build_pyg_data(G, features_df)

    logger.info("\n>>> PHASE 6: GNN Model Training")
    model, history = train_model(data, device)

    logger.info("\n>>> PHASE 7 & 8: Evaluation & Hybrid Model")
    comparison_df = run_full_evaluation(
        G, features_df, labels_df, data, model, history, account_labels, device, node_to_idx=node_to_idx
    )

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)
    logger.info("View outputs in:")
    logger.info("  - %s (Data & Labels)", config.PROCESSED_DATA_DIR)
    logger.info("  - %s (Plots & Visualizations)", config.PLOTS_DIR)
    logger.info("  - %s (Metrics & Results)", config.RESULTS_DIR)
    logger.info("  - %s (Trained Models)", config.MODELS_DIR)

    if history:
        history_series = {}
        max_len = 0
        for key, value in history.items():
            if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
                values = list(value)
                history_series[key] = values
                max_len = max(max_len, len(values))

        if history_series and max_len > 0:
            for key, values in history_series.items():
                if len(values) < max_len:
                    history_series[key] = values + [np.nan] * (max_len - len(values))

            pd.DataFrame(history_series).to_csv(
                os.path.join(config.RESULTS_DIR, "training_history.csv"), index=False
            )
            print("Training history saved for dashboard.")

    print("\nTo view results dashboard: python3 dashboard/dashboard_server.py")


if __name__ == "__main__":
    main()
