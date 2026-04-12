"""Flask server for the fraud detection dashboard."""

import os
import pandas as pd
from flask import Flask, jsonify, send_from_directory

app = Flask(__name__, static_folder="static")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json_safe(path):
    """Load a CSV and return records; return empty list on failure."""
    try:
        return pd.read_csv(path).to_dict(orient="records")
    except Exception:
        return []


@app.route("/api/metrics")
def get_metrics():
    path = os.path.join(BASE, "outputs", "results", "final_metrics.csv")
    return jsonify(load_json_safe(path))


@app.route("/api/features")
def get_features():
    path = os.path.join(BASE, "data", "processed", "node_features.csv")
    data = load_json_safe(path)
    return jsonify(data[:200])


@app.route("/api/labels")
def get_labels():
    path = os.path.join(BASE, "data", "processed", "labels.csv")
    try:
        df = pd.read_csv(path)
        candidates = [c for c in df.columns if "fraud" in c.lower() or "label" in c.lower()]
        if not candidates:
            return jsonify({"fraud": 0, "normal": 0})

        col = candidates[-1]
        counts = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).value_counts().to_dict()
        return jsonify({"fraud": int(counts.get(1, 0)), "normal": int(counts.get(0, 0))})
    except Exception:
        return jsonify({"fraud": 0, "normal": 0})


@app.route("/api/cpp_status")
def get_cpp_status():
    binary = os.path.join(BASE, "cpp", "graph_algorithms")
    return jsonify({"available": os.path.exists(binary), "path": binary})


@app.route("/api/training_history")
def get_training_history():
    path = os.path.join(BASE, "outputs", "results", "training_history.csv")
    return jsonify(load_json_safe(path))


@app.route("/")
def dashboard():
    return send_from_directory(os.path.dirname(__file__), "dashboard.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), "static"), filename)


if __name__ == "__main__":
    print("\nFraud Detection Dashboard")
    print("Open: http://localhost:5000\n")
    app.run(debug=False, port=5000)
