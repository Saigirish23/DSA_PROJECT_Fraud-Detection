
import os
import subprocess
from io import StringIO

import pandas as pd

CPP_DIR = os.path.join(os.path.dirname(__file__))
BINARY = os.path.join(CPP_DIR, "graph_algorithms")


def run_cpp_algorithms(edge_file_path):
    if not os.path.exists(BINARY):
        print("  [C++ backend] Binary not found, falling back to NetworkX")
        return None

    try:
        result = subprocess.run(
            [BINARY, edge_file_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"  [C++ backend] Error: {result.stderr.strip()}")
            return None

        df = pd.read_csv(StringIO(result.stdout))
        expected_cols = {"node_id", "degree", "clustering", "pagerank", "betweenness"}
        if not expected_cols.issubset(df.columns):
            print("  [C++ backend] Invalid output schema, falling back to NetworkX")
            return None

        df = df.set_index("node_id")
        print(f"  [C++ backend] Computed features for {len(df)} nodes")
        return df
    except subprocess.TimeoutExpired:
        print("  [C++ backend] Timed out, falling back to NetworkX")
        return None
    except Exception as exc:
        print(f"  [C++ backend] Failed: {exc}, falling back to NetworkX")
        return None


def is_cpp_available():
    return os.path.exists(BINARY)
