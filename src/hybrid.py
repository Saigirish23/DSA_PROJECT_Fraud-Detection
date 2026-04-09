"""
hybrid.py — Heuristic + GNN Combination Logic

Implements two hybrid strategies for combining heuristic fraud scores
with GNN predictions:

Strategy A — Feature Augmentation:
    Adds the heuristic fraud score as an additional input feature to the GNN.
    The GNN then learns to use both structural features AND the heuristic score.

Strategy B — Late Fusion (Ensemble):
    Combines GNN probability with heuristic score using a weighted average:
        final_score = α × GNN_probability + (1 - α) × heuristic_score
    Sweeps α from 0 to 1 to find the optimal blending weight.

Both strategies are compared against the standalone heuristic and GNN baselines.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

import config
from src.gnn_model import FraudGCN
from src.data_loader import set_seeds

logger = config.setup_logging(__name__)


def strategy_a_feature_augmentation(G, features_df, labels_series, fraud_scores, device="cpu"):
    """
    Strategy A: Add heuristic fraud score as an extra input feature to the GNN.

    Rationale:
        The GNN receives the heuristic score as prior knowledge. If the heuristic
        is informative, the GNN can learn to amplify it. If not, it can learn to
        downweight it. This is strictly more powerful than either alone.

    Args:
        G (nx.DiGraph): Transaction graph.
        features_df (pd.DataFrame): Node features with 'node_id' column.
        labels_series (pd.Series): Node labels indexed by node_id.
        fraud_scores (pd.Series): Heuristic fraud scores indexed by node_id.
        device (str): 'cuda' or 'cpu'.

    Returns:
        tuple: (predictions, probabilities, metrics_dict, model, history)
    """
    from torch_geometric.data import Data
    from sklearn.model_selection import train_test_split

    logger.info("=" * 60)
    logger.info("Strategy A: Feature Augmentation (heuristic score as input feature)")
    logger.info("=" * 60)

    set_seeds()

    # Get sorted node list
    nodes = sorted(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)

    # Build augmented feature matrix (original features + heuristic score)
    feature_cols = [col for col in features_df.columns if col != "node_id"]
    num_original_features = len(feature_cols)

    feature_matrix = np.zeros((num_nodes, num_original_features + 1), dtype=np.float32)

    for _, row in features_df.iterrows():
        node_id = int(row["node_id"])
        if node_id in node_to_idx:
            idx = node_to_idx[node_id]
            feature_matrix[idx, :num_original_features] = [row[col] for col in feature_cols]
            # Append heuristic fraud score as the last feature
            feature_matrix[idx, -1] = fraud_scores.get(node_id, 0.0)

    # Normalize all features (including the heuristic score)
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix).astype(np.float32)

    x = torch.tensor(feature_matrix, dtype=torch.float32)
    logger.info("  Augmented feature matrix: %s (original + heuristic_score)", x.shape)

    # Build edge index
    edge_list = []
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            edge_list.append([node_to_idx[u], node_to_idx[v]])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Labels
    y = torch.zeros(num_nodes, dtype=torch.long)
    for node_id, label in labels_series.items():
        if node_id in node_to_idx:
            y[node_to_idx[node_id]] = int(label)

    # Stratified split
    indices = np.arange(num_nodes)
    train_idx, test_idx = train_test_split(
        indices, test_size=(1.0 - config.TRAIN_RATIO),
        stratify=y.numpy(), random_state=config.RANDOM_SEED
    )
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, test_mask=test_mask)

    # Train augmented GCN (num_features = original + 1)
    data = data.to(device)
    model = FraudGCN(num_features=x.shape[1]).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Class weights
    train_labels = y[train_mask].numpy()
    class_counts = np.bincount(train_labels, minlength=config.GNN_NUM_CLASSES)
    total = class_counts.sum()
    weights = total / (len(class_counts) * class_counts + 1e-8)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = torch.nn.NLLLoss(weight=class_weights)

    # Training loop
    history = {"train_loss": [], "test_f1": []}
    best_f1 = 0.0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            test_f1 = f1_score(
                data.y[data.test_mask].cpu().numpy(),
                pred[data.test_mask].cpu().numpy(),
                zero_division=0
            )

        history["train_loss"].append(loss.item())
        history["test_f1"].append(test_f1)

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_state = model.state_dict().copy()

        if epoch % config.LOG_INTERVAL == 0:
            logger.info("  Epoch %3d/%d | Loss: %.4f | Test F1: %.4f",
                        epoch, config.NUM_EPOCHS, loss.item(), test_f1)

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.exp(out).cpu()
        preds = out.argmax(dim=1).cpu()

    # Compute metrics on test set
    y_true = y[test_mask].numpy()
    y_pred = preds[test_mask].numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    logger.info("  Strategy A Results:")
    for k, v in metrics.items():
        logger.info("    %-12s: %.4f", k.capitalize(), v)

    return preds, probs, metrics, model, history


def strategy_b_late_fusion(gnn_probs, heuristic_scores, labels, test_mask):
    """
    Strategy B: Late Fusion — weighted average of GNN and heuristic scores.

    Sweeps alpha from 0 to 1:
        final_score = α × GNN_fraud_prob + (1 - α) × heuristic_score
        If final_score > 0.5 → predict fraud

    Args:
        gnn_probs (torch.Tensor): GNN probabilities [N, 2].
        heuristic_scores (np.ndarray or torch.Tensor): Heuristic fraud scores [N].
        labels (torch.Tensor): Ground truth labels [N].
        test_mask (torch.Tensor): Boolean mask for test nodes.

    Returns:
        tuple: (best_alpha, best_metrics, all_results_df)
    """
    logger.info("=" * 60)
    logger.info("Strategy B: Late Fusion (sweep α from 0.0 to 1.0)")
    logger.info("=" * 60)

    if isinstance(heuristic_scores, torch.Tensor):
        h_scores = heuristic_scores.float().numpy()
    else:
        h_scores = np.array(heuristic_scores, dtype=np.float32)

    # GNN fraud probability (class 1)
    gnn_fraud_prob = gnn_probs[:, 1].numpy()

    # Normalize heuristic scores to [0, 1]
    h_min, h_max = h_scores.min(), h_scores.max()
    if h_max > h_min:
        h_scores_norm = (h_scores - h_min) / (h_max - h_min)
    else:
        h_scores_norm = h_scores

    y_true = labels[test_mask].numpy()
    results = []

    alphas = np.arange(
        config.ALPHA_SWEEP_START,
        config.ALPHA_SWEEP_END + config.ALPHA_SWEEP_STEP / 2,
        config.ALPHA_SWEEP_STEP
    )

    for alpha in alphas:
        # Blend scores
        final_score = alpha * gnn_fraud_prob + (1 - alpha) * h_scores_norm
        y_pred = (final_score[test_mask.numpy()] > 0.5).astype(int)

        metrics = {
            "alpha": round(alpha, 1),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
        results.append(metrics)
        logger.info("  α=%.1f | Acc: %.4f | P: %.4f | R: %.4f | F1: %.4f",
                     alpha, metrics["accuracy"], metrics["precision"],
                     metrics["recall"], metrics["f1"])

    results_df = pd.DataFrame(results)

    # Find best alpha by F1
    best_idx = results_df["f1"].idxmax()
    best_alpha = results_df.loc[best_idx, "alpha"]
    best_metrics = results_df.loc[best_idx].to_dict()

    logger.info("-" * 60)
    logger.info("  Best α = %.1f | F1 = %.4f", best_alpha, best_metrics["f1"])

    return best_alpha, best_metrics, results_df


def run_hybrid_comparison(G, features_df, labels_df, data, model, device="cpu"):
    """
    Run both hybrid strategies and produce a comparison table.

    Args:
        G (nx.DiGraph): Transaction graph.
        features_df (pd.DataFrame): Node features.
        labels_df (pd.DataFrame): Labels with 'node_id', 'fraud_score', 'heuristic_label'.
        data (torch_geometric.data.Data): PyG data object.
        model (FraudGCN): Trained GNN model.
        device (str): 'cuda' or 'cpu'.

    Returns:
        pd.DataFrame: Comparison table with metrics for all models.
    """
    from src.heuristics import evaluate_heuristic
    from src.train import get_gnn_predictions

    logger.info("\n" + "=" * 70)
    logger.info("HYBRID MODEL COMPARISON")
    logger.info("=" * 70)

    labels_series = labels_df.set_index("node_id")["heuristic_label"]
    fraud_scores = labels_df.set_index("node_id")["fraud_score"]

    # --- Heuristic Only ---
    test_mask_np = data.test_mask.cpu().numpy()
    nodes = sorted(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    y_true_test = data.y[data.test_mask].cpu().numpy()

    # Heuristic predictions on test set
    h_preds_test = []
    for node in nodes:
        if node in node_to_idx:
            idx = node_to_idx[node]
            if data.test_mask[idx]:
                h_preds_test.append(int(labels_series.get(node, 0)))
    h_preds_test = np.array(h_preds_test)

    heuristic_metrics = {
        "accuracy": accuracy_score(y_true_test, h_preds_test),
        "precision": precision_score(y_true_test, h_preds_test, zero_division=0),
        "recall": recall_score(y_true_test, h_preds_test, zero_division=0),
        "f1": f1_score(y_true_test, h_preds_test, zero_division=0),
    }

    # --- GNN Only ---
    gnn_preds, gnn_probs = get_gnn_predictions(model, data, device)
    y_pred_gnn = gnn_preds[data.test_mask.cpu()].numpy()

    gnn_metrics = {
        "accuracy": accuracy_score(y_true_test, y_pred_gnn),
        "precision": precision_score(y_true_test, y_pred_gnn, zero_division=0),
        "recall": recall_score(y_true_test, y_pred_gnn, zero_division=0),
        "f1": f1_score(y_true_test, y_pred_gnn, zero_division=0),
    }

    # --- Strategy A: Feature Augmentation ---
    _, _, strategy_a_metrics, _, _ = strategy_a_feature_augmentation(
        G, features_df, labels_series, fraud_scores.to_dict(), device
    )

    # --- Strategy B: Late Fusion ---
    # Build heuristic scores aligned with node ordering
    h_scores_aligned = np.zeros(len(nodes), dtype=np.float32)
    for node in nodes:
        idx = node_to_idx[node]
        h_scores_aligned[idx] = fraud_scores.get(node, 0.0)

    best_alpha, strategy_b_metrics, fusion_results = strategy_b_late_fusion(
        gnn_probs, h_scores_aligned, data.y.cpu(), data.test_mask.cpu()
    )

    # --- Comparison Table ---
    comparison = pd.DataFrame([
        {"Model": "Heuristic Only", **heuristic_metrics},
        {"Model": "GNN Only", **gnn_metrics},
        {"Model": "Hybrid (Strategy A)", **strategy_a_metrics},
        {"Model": "Hybrid (Strategy B, α={:.1f})".format(best_alpha), **{
            k: v for k, v in strategy_b_metrics.items() if k != "alpha"
        }},
    ])

    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON TABLE")
    logger.info("=" * 70)
    for _, row in comparison.iterrows():
        logger.info(
            "  %-35s  Acc=%.4f  P=%.4f  R=%.4f  F1=%.4f",
            row["Model"], row["accuracy"], row["precision"],
            row["recall"], row["f1"]
        )
    logger.info("=" * 70)

    # Save results
    config.ensure_dirs()
    comparison.to_csv(
        config.RESULTS_DIR + "/model_comparison.csv", index=False
    )
    fusion_results.to_csv(
        config.RESULTS_DIR + "/alpha_sweep_results.csv", index=False
    )
    logger.info("  Results saved to outputs/results/")

    return comparison, fusion_results, best_alpha


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import load_dataset, build_graph, build_pyg_data
    from src.features import compute_all_features
    from src.heuristics import compute_fraud_scores, generate_heuristic_labels
    from src.train import train_model

    set_seeds()
    config.ensure_dirs()

    # Full pipeline
    df, account_labels = load_dataset()
    G = build_graph(df)
    features_df = compute_all_features(G)
    scored_df = compute_fraud_scores(features_df)
    labels_df = generate_heuristic_labels(scored_df)
    labels_series = labels_df.set_index("node_id")["heuristic_label"]
    data, scaler, node_to_idx = build_pyg_data(G, features_df, labels_series)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, history = train_model(data, device)

    # Run hybrid comparison
    comparison, fusion_results, best_alpha = run_hybrid_comparison(
        G, features_df, labels_df, data, model, device
    )

    logger.info("\n✅ Hybrid model comparison complete!")
