
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import config

logger = config.setup_logging(__name__)


def normalize_series(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)


def compute_fraud_scores(features_df):
    weights = config.HEURISTIC_WEIGHTS
    w1, w2, w3 = weights["w1"], weights["w2"], weights["w3"]

    logger.info("Computing heuristic fraud scores...")
    logger.info("  Weights: w1=%.2f (degree), w2=%.2f (1-clustering), w3=%.2f (pagerank)",
                w1, w2, w3)

    df = features_df.copy()

    degree_col = "degree" if "degree" in df.columns else "total_degree"
    clustering_col = "clustering" if "clustering" in df.columns else "clustering_coefficient"

    norm_degree = normalize_series(df[degree_col])
    norm_clustering = normalize_series(df[clustering_col])
    norm_pagerank = normalize_series(df["pagerank"])

    df["fraud_score"] = (
        w1 * norm_degree
        + w2 * (1.0 - norm_clustering)
        + w3 * norm_pagerank
    )

    df = df.sort_values("fraud_score", ascending=False).reset_index(drop=True)

    logger.info("  Fraud score range: [%.4f, %.4f]", df["fraud_score"].min(), df["fraud_score"].max())
    logger.info("  Mean fraud score: %.4f", df["fraud_score"].mean())

    return df


def generate_heuristic_labels(scored_df, threshold=None):
    if threshold is None:
        threshold = config.HEURISTIC_FRAUD_THRESHOLD

    logger.info("Generating heuristic labels (top %.0f%% → fraud)...", threshold * 100)

    df = scored_df.copy()
    n_fraud = int(len(df) * threshold)

    df = df.sort_values("fraud_score", ascending=False).reset_index(drop=True)
    df["heuristic_label"] = 0
    df.loc[:n_fraud - 1, "heuristic_label"] = 1

    n_labeled_fraud = df["heuristic_label"].sum()
    logger.info("  Labeled %d nodes as fraud, %d as normal",
                n_labeled_fraud, len(df) - n_labeled_fraud)

    labels_df = df[["node_id", "fraud_score", "heuristic_label"]].copy()
    config.ensure_dirs()
    labels_df.to_csv(config.LABELS_PATH, index=False)
    logger.info("  Saved labels to %s", config.LABELS_PATH)

    return labels_df


def evaluate_heuristic(labels_df, ground_truth_labels):
    logger.info("Evaluating heuristic labels against ground truth...")

    y_true = []
    y_pred = []
    for _, row in labels_df.iterrows():
        node_id = str(row["node_id"])
        if node_id in ground_truth_labels and int(ground_truth_labels[node_id]) != -1:
            y_true.append(int(ground_truth_labels[node_id]))
            y_pred.append(int(row["heuristic_label"]))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.size == 0:
        logger.warning("  No labeled nodes available for heuristic evaluation")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    logger.info("  Heuristic Baseline Metrics:")
    logger.info("  %s", "-" * 40)
    for metric_name, value in metrics.items():
        logger.info("    %-12s: %.4f", metric_name.capitalize(), value)
    logger.info("  %s", "-" * 40)

    return metrics


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import load_dataset, build_graph, set_seeds
    from src.features import compute_all_features

    set_seeds()
    config.ensure_dirs()

    df, account_labels = load_dataset()
    G = build_graph(df)

    features_df = compute_all_features(G)

    scored_df = compute_fraud_scores(features_df)
    labels_df = generate_heuristic_labels(scored_df)

    metrics = evaluate_heuristic(labels_df, account_labels)

    logger.info("\n✅ Heuristic fraud detection complete!")
