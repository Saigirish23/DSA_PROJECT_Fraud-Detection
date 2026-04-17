
import os
import pandas as pd
import networkx as nx
from flask import Flask, jsonify, request, send_from_directory
import logging
app = Flask(__name__, static_folder="static")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURES_PATH = os.path.join(BASE, "data", "processed", "node_features.csv")
LABELS_PATH = os.path.join(BASE, "data", "processed", "labels.csv")
METRICS_PATH = os.path.join(BASE, "outputs", "results", "final_metrics.csv")
PREDICTIONS_PATH = os.path.join(BASE, "outputs", "results", "node_predictions.csv")


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s : %(message)s')
logger = logging.getLogger(__name__)

def load_json_safe(path):
    try:
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, str(e))
        return []


@app.route("/api/metrics")
def get_metrics():
    return jsonify(load_json_safe(METRICS_PATH))


@app.route("/api/features")
def get_features():
    limit = request.args.get("limit", default=500, type=int)
    if limit is None:
        limit = 500
    limit = max(1, min(int(limit), 5000))

    data = load_json_safe(FEATURES_PATH)
    return jsonify(data[:limit])


@app.route("/api/feature_metadata")
def get_feature_metadata():
    if not os.path.exists(FEATURES_PATH):
        return jsonify(
            {
                "available": False,
                "mode": "unknown",
                "feature_columns": [],
                "row_count": 0,
                "has_recent_transaction_sum": False,
                "has_betweenness": False,
                "has_in_out_degree": False,
                "recent_transaction_sum_stats": {"min": 0.0, "max": 0.0, "mean": 0.0},
                "top_recent_transactions": [],
            }
        )

    try:
        df = pd.read_csv(FEATURES_PATH)
    except Exception as e:
        logger.warning("Failed to read feature metadata from %s: %s", FEATURES_PATH, str(e))
        return jsonify(
            {
                "available": False,
                "mode": "unknown",
                "feature_columns": [],
                "row_count": 0,
                "has_recent_transaction_sum": False,
                "has_betweenness": False,
                "has_in_out_degree": False,
                "recent_transaction_sum_stats": {"min": 0.0, "max": 0.0, "mean": 0.0},
                "top_recent_transactions": [],
            }
        )

    columns = [str(c) for c in df.columns]
    has_recent_tx = "recent_transaction_sum" in columns
    has_betweenness = "betweenness" in columns or "betweenness_centrality" in columns
    has_in_out_degree = "in_degree" in columns and "out_degree" in columns
    mode = "dynamic" if has_recent_tx else "static"

    recent_stats = {"min": 0.0, "max": 0.0, "mean": 0.0}
    top_recent = []
    if has_recent_tx and "node_id" in df.columns:
        tx_df = df[["node_id", "recent_transaction_sum"]].copy()
        tx_df["node_id"] = tx_df["node_id"].astype(str)
        tx_df["recent_transaction_sum"] = pd.to_numeric(
            tx_df["recent_transaction_sum"], errors="coerce"
        ).fillna(0.0)

        if not tx_df.empty:
            recent_stats = {
                "min": float(tx_df["recent_transaction_sum"].min()),
                "max": float(tx_df["recent_transaction_sum"].max()),
                "mean": float(tx_df["recent_transaction_sum"].mean()),
            }
            top_recent = (
                tx_df.sort_values("recent_transaction_sum", ascending=False)
                .head(10)
                .to_dict(orient="records")
            )

    return jsonify(
        {
            "available": True,
            "mode": mode,
            "feature_columns": columns,
            "row_count": int(len(df)),
            "has_recent_transaction_sum": bool(has_recent_tx),
            "has_betweenness": bool(has_betweenness),
            "has_in_out_degree": bool(has_in_out_degree),
            "recent_transaction_sum_stats": recent_stats,
            "top_recent_transactions": top_recent,
        }
    )


@app.route("/api/labels")
def get_labels():
    try:
        df = pd.read_csv(LABELS_PATH)
        candidates = [c for c in df.columns if "fraud" in c.lower() or "label" in c.lower()]
        if not candidates:
            return jsonify({"fraud": 0, "normal": 0})

        col = candidates[-1]
        counts = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).value_counts().to_dict()
        return jsonify({"fraud": int(counts.get(1, 0)), "normal": int(counts.get(0, 0))})
    except Exception as e:
        logger.warning("Failed to load labels: %s", str(e))
        return jsonify({"fraud": 0, "normal": 0})


@app.route("/api/predictions")
def get_predictions():
    try:
        df = pd.read_csv(PREDICTIONS_PATH)
        df_sorted = df.sort_values('fraud_probability', ascending=False).head(100)
        records = df_sorted.to_dict(orient="records")
        logger.info("Loaded %d predictions from %s", len(records), PREDICTIONS_PATH)
        return jsonify(records)
    except Exception as e:
        logger.warning("Failed to load predictions from %s: %s", PREDICTIONS_PATH, str(e))
        return jsonify([])


@app.route("/api/graph_stats")
def get_graph_stats():
    try:
        tx_path = os.path.join(BASE, "data", "raw", "transactions.csv")
        if not os.path.exists(tx_path):
            logger.warning("Transactions file not found at %s", tx_path)
            return jsonify({"nodes": 0, "edges": 0, "density": 0.0, "avg_degree": 0.0})
        
        df = pd.read_csv(tx_path)
        
        G = nx.DiGraph()
        if "sender_id" in df.columns and "receiver_id" in df.columns:
            for _, row in df.iterrows():
                sender = str(row["sender_id"])
                receiver = str(row["receiver_id"])
                if sender != receiver:
                    G.add_edge(sender, receiver)
        
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G) if num_nodes > 1 else 0.0
        avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0.0
        
        logger.info("Graph stats: nodes=%d, edges=%d, density=%.4f, avg_degree=%.2f",
                   num_nodes, num_edges, density, avg_degree)
        
        return jsonify({
            "nodes": num_nodes,
            "edges": num_edges,
            "density": float(density),
            "avg_degree": float(avg_degree)
        })
    except Exception as e:
        logger.warning("Failed to compute graph stats: %s", str(e))
        return jsonify({"nodes": 0, "edges": 0, "density": 0.0, "avg_degree": 0.0})


@app.route("/api/cpp_status")
def get_cpp_status():
    binary = os.path.join(BASE, "cpp", "graph_algorithms")
    return jsonify({"available": os.path.exists(binary), "path": binary})


@app.route("/api/training_history")
def get_training_history():
    results_dir = os.path.join(BASE, "outputs", "results")
    tuned_path = os.path.join(results_dir, "elliptic_training_history.csv")
    default_path = os.path.join(results_dir, "training_history.csv")

    source = (request.args.get("source") or "auto").strip().lower()
    if source == "elliptic":
        return jsonify(load_json_safe(tuned_path))
    if source == "default":
        return jsonify(load_json_safe(default_path))

    if os.path.exists(tuned_path):
        tuned_data = load_json_safe(tuned_path)
        if tuned_data:
            return jsonify(tuned_data)

    return jsonify(load_json_safe(default_path))


@app.route("/")
def dashboard():
    return send_from_directory(os.path.dirname(__file__), "dashboard.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), "static"), filename)


if __name__ == "__main__":
    print("\nFraud Detection Dashboard")
    print("Open: http://localhost:5000\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
