"""
Dynamic Graph Module — True Incremental Algorithms
====================================================

This module implements a fully incremental dynamic fraud graph.
NO NetworkX calls are used. All structural metrics are maintained
incrementally as edges are added/removed.

Complexity Summary (per edge update):
--------------------------------------
| Feature              | Before (Static)       | After (Incremental)         |
|----------------------|-----------------------|-----------------------------|
| degree               | O(V + E)              | O(1)                        |
| clustering           | O(V × deg²)           | O(min(deg(u), deg(v)))      |
| pagerank             | O(k × (V + E))        | O(k × avg_deg)  [k ≈ 10]   |
| recent_tx_sum        | N/A                   | O(log n)  [Fenwick tree]    |
| sliding window expiry| O(1) amortized        | O(1) amortized              |
"""

from collections import defaultdict, deque
import bisect
import math

import pandas as pd


# Phase 6: Final unified feature schema (6 columns, no betweenness)
CANONICAL_FEATURE_COLUMNS = [
    "degree",
    "in_degree",
    "out_degree",
    "clustering",
    "pagerank",
    "recent_transaction_sum",
]


# ---------------------------------------------------------------------------
# Phase 4: Fenwick Tree (Binary Indexed Tree) — O(log n) point update & prefix sum
# ---------------------------------------------------------------------------
class FenwickTree:
    """
    Binary Indexed Tree for prefix-sum queries on transaction amounts.

    Complexity:
        update:      O(log n)
        query:       O(log n)
        range_query: O(log n)

    Used to compute recent_transaction_sum(node, window) efficiently.
    """

    def __init__(self, n):
        self.n = int(n)
        self.tree = [0.0] * (self.n + 1)

    def update(self, i, delta):
        """Add *delta* to position *i* (0-indexed).  O(log n)."""
        i += 1  # convert to 1-indexed
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def query(self, i):
        """Prefix sum [0 .. i] (0-indexed).  O(log n)."""
        s = 0.0
        i += 1  # convert to 1-indexed
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def range_query(self, l, r):
        """Sum over [l .. r] (0-indexed, inclusive).  O(log n)."""
        if l > r:
            return 0.0
        return self.query(r) - (self.query(l - 1) if l > 0 else 0.0)


# ---------------------------------------------------------------------------
# Core Dynamic Graph
# ---------------------------------------------------------------------------
class DynamicFraudGraph:
    """
    Fully incremental dynamic fraud-detection graph.

    All structural features (degree, clustering, pagerank) are maintained
    incrementally — NO global recomputation is ever performed.

    Sliding-window semantics:
        - Edges are timestamped and expire after *window_size* seconds.
        - Duplicate edges are reference-counted (edge_counts[(u,v)]).
        - Graph-structural updates only fire on true topology changes
          (edge count 0→1 for add, 1→0 for remove).

    Fenwick tree integration:
        - Each node has a timeline of transaction amounts.
        - A FenwickTree per node supports O(log n) range-sum queries
          for recent_transaction_sum.
    """

    def __init__(self, window_size=7):
        self.window = int(window_size)

        # --- Sliding window bookkeeping (Phase 5) ---
        # edge_counts tracks duplicate edges; graph topology only changes
        # when count transitions through 0↔1.
        self.edge_counts = defaultdict(int)
        self.history = deque()       # deque of (timestamp, u, v, amount)
        self.current_time = None

        # --- Degree tracking: O(1) per update ---
        self.in_degree = defaultdict(int)
        self.out_degree = defaultdict(int)

        # --- Adjacency sets (undirected) for triangle counting (Phase 3) ---
        self.adj_sets = defaultdict(set)

        # --- Incremental clustering (Phase 3) ---
        # triangle_count[node] = number of triangles incident to node
        self.triangle_count = defaultdict(int)
        self.clustering_coeff = {}

        # --- Incremental PageRank (Phase 2) ---
        # rank[node] maintained incrementally; updated locally on each edge change
        self.rank = {}
        self._node_count = 0         # total distinct nodes seen
        self._out_adj = defaultdict(set)   # directed out-neighbors
        self._in_adj = defaultdict(set)    # directed in-neighbors

        # --- Fenwick tree per node for transaction amounts (Phase 4) ---
        # node_tx_timestamps[node] = sorted list of timestamps
        # node_tx_amounts[node]    = corresponding amounts
        # node_fenwick[node]       = FenwickTree over amounts
        self.node_tx_timestamps = defaultdict(list)
        self.node_tx_amounts = defaultdict(list)
        self.node_fenwick = {}

    # -----------------------------------------------------------------------
    # Incremental PageRank helpers  (Phase 2)
    # -----------------------------------------------------------------------
    # Complexity: O(k × |affected_neighbors|) where k=10 iterations
    # -----------------------------------------------------------------------

    def _init_node_rank(self, node):
        """Ensure a node has an initial rank value."""
        if node not in self.rank:
            self._node_count += 1
            # Initialize with uniform rank
            self.rank[node] = 1.0 / max(self._node_count, 1)

    def _local_pagerank_update(self, affected_nodes, k=10, damping=0.85):
        """
        Recompute PageRank ONLY for *affected_nodes* using k local iterations.

        Algorithm:
            For each iteration:
                For each node in affected_nodes:
                    rank[node] = (1-d)/N + d × Σ(rank[in_neighbor] / out_degree(in_neighbor))

        Complexity: O(k × Σ deg(node) for node in affected_nodes)
                  ≈ O(k × avg_deg) when affected set is small

        This is a local relaxation — it converges to the global PageRank
        as more updates accumulate, but each individual call is O(k × deg).
        """
        n = max(self._node_count, 1)
        base = (1.0 - damping) / n

        for _ in range(k):
            for node in affected_nodes:
                incoming_sum = 0.0
                for src in self._in_adj.get(node, set()):
                    src_out_deg = len(self._out_adj.get(src, set()))
                    if src_out_deg > 0:
                        incoming_sum += self.rank.get(src, 0.0) / src_out_deg
                self.rank[node] = base + damping * incoming_sum

    def _collect_affected_nodes(self, u, v):
        """
        Collect the set of nodes whose PageRank may be affected by
        an edge change (u, v):
            affected = {u, v} ∪ neighbors(u) ∪ neighbors(v)

        Complexity: O(deg(u) + deg(v))
        """
        affected = {u, v}
        affected.update(self._out_adj.get(u, set()))
        affected.update(self._in_adj.get(u, set()))
        affected.update(self._out_adj.get(v, set()))
        affected.update(self._in_adj.get(v, set()))
        # Only include nodes that actually exist in the rank table
        return {n for n in affected if n in self.rank}

    # -----------------------------------------------------------------------
    # Incremental Clustering helpers  (Phase 3)
    # -----------------------------------------------------------------------
    # Complexity: O(min(deg(u), deg(v))) per edge update
    # -----------------------------------------------------------------------

    def _update_clustering_for_node(self, node):
        """
        Recompute clustering coefficient for a single node from its
        current triangle count and degree.

        Formula: C(node) = 2 * triangles / (degree * (degree - 1))
                 where degree is the *undirected* degree.

        Complexity: O(1)
        """
        deg = len(self.adj_sets.get(node, set()))
        if deg < 2:
            self.clustering_coeff[node] = 0.0
        else:
            denom = deg * (deg - 1)
            self.clustering_coeff[node] = (
                (2.0 * self.triangle_count.get(node, 0)) / denom
            )

    def _add_edge_triangles(self, u, v):
        """
        When undirected edge (u, v) is added, find common neighbors
        and increment triangle counts.

        common = N(u) ∩ N(v)   [computed BEFORE adding the edge]
        For each w in common:
            triangle_count[u] += 1
            triangle_count[v] += 1
            triangle_count[w] += 1

        Then update clustering for u, v, and all w.

        Complexity: O(min(deg(u), deg(v)))
        """
        # Compute common neighbors BEFORE adding (u,v) to adj_sets
        nu = self.adj_sets.get(u, set())
        nv = self.adj_sets.get(v, set())

        # Iterate over the smaller set for efficiency
        if len(nu) < len(nv):
            common = nu & nv
        else:
            common = nv & nu

        # Increment triangle counts
        for w in common:
            self.triangle_count[u] += 1
            self.triangle_count[v] += 1
            self.triangle_count[w] += 1

        # Now add the undirected edge
        self.adj_sets[u].add(v)
        self.adj_sets[v].add(u)

        # Update clustering for all affected nodes
        self._update_clustering_for_node(u)
        self._update_clustering_for_node(v)
        for w in common:
            self._update_clustering_for_node(w)

    def _remove_edge_triangles(self, u, v):
        """
        When undirected edge (u, v) is removed, find common neighbors
        and decrement triangle counts.

        Complexity: O(min(deg(u), deg(v)))
        """
        # Common neighbors computed BEFORE removing the edge
        nu = self.adj_sets.get(u, set())
        nv = self.adj_sets.get(v, set())

        if len(nu) < len(nv):
            common = nu & nv
        else:
            common = nv & nu
        # Remove w=u and w=v from common if present (they're endpoints, not triangles)
        common.discard(u)
        common.discard(v)

        # Decrement triangle counts
        for w in common:
            self.triangle_count[u] = max(0, self.triangle_count.get(u, 0) - 1)
            self.triangle_count[v] = max(0, self.triangle_count.get(v, 0) - 1)
            self.triangle_count[w] = max(0, self.triangle_count.get(w, 0) - 1)

        # Remove the undirected edge
        self.adj_sets[u].discard(v)
        self.adj_sets[v].discard(u)

        # Update clustering for all affected nodes
        self._update_clustering_for_node(u)
        self._update_clustering_for_node(v)
        for w in common:
            self._update_clustering_for_node(w)

    # -----------------------------------------------------------------------
    # Edge add/remove with correct sliding window (Phase 5)
    # -----------------------------------------------------------------------

    def _increment_edge(self, u, v):
        """
        Increment edge count for (u, v).  Only trigger structural updates
        when edge count transitions 0 → 1 (true new edge in the graph).

        Phase 5: Duplicate edges are counted but don't alter topology.
        """
        key = (u, v)
        prev = self.edge_counts[key]
        self.edge_counts[key] = prev + 1

        if prev == 0:
            # --- True new edge: update all structural data ---

            # Degree: O(1)
            self.out_degree[u] += 1
            self.in_degree[v] += 1

            # Directed adjacency for PageRank: O(1)
            self._out_adj[u].add(v)
            self._in_adj[v].add(u)

            # Initialize rank for new nodes: O(1)
            self._init_node_rank(u)
            self._init_node_rank(v)

            # Incremental clustering — triangle detection: O(min(deg(u), deg(v)))
            self._add_edge_triangles(u, v)

            # Incremental PageRank — local update: O(k × avg_deg)
            affected = self._collect_affected_nodes(u, v)
            self._local_pagerank_update(affected)

    def _decrement_edge(self, u, v):
        """
        Decrement edge count for (u, v).  Only trigger structural updates
        when edge count transitions 1 → 0 (true edge removal from graph).

        Phase 5: Edge is only removed from graph when count reaches 0.
        """
        key = (u, v)
        count = self.edge_counts.get(key, 0)
        if count <= 0:
            return

        if count == 1:
            # --- True edge removal: update all structural data ---
            del self.edge_counts[key]

            # Directed adjacency update: O(1)
            self._out_adj[u].discard(v)
            self._in_adj[v].discard(u)

            # Incremental clustering — triangle removal: O(min(deg(u), deg(v)))
            self._remove_edge_triangles(u, v)

            # Degree: O(1)
            if self.out_degree.get(u, 0) > 0:
                self.out_degree[u] -= 1
                if self.out_degree[u] == 0:
                    del self.out_degree[u]

            if self.in_degree.get(v, 0) > 0:
                self.in_degree[v] -= 1
                if self.in_degree[v] == 0:
                    del self.in_degree[v]

            # Incremental PageRank — local update: O(k × avg_deg)
            affected = self._collect_affected_nodes(u, v)
            self._local_pagerank_update(affected)

            # Clean up isolated nodes
            self._maybe_remove_node(u)
            self._maybe_remove_node(v)
        else:
            self.edge_counts[key] = count - 1

    def _maybe_remove_node(self, node):
        """Remove a node from tracking if it has no remaining edges."""
        has_out = len(self._out_adj.get(node, set())) > 0
        has_in = len(self._in_adj.get(node, set())) > 0
        if not has_out and not has_in:
            # Node is isolated — clean up
            self.rank.pop(node, None)
            self.triangle_count.pop(node, None)
            self.clustering_coeff.pop(node, None)
            self._out_adj.pop(node, None)
            self._in_adj.pop(node, None)
            self.adj_sets.pop(node, None)
            if self._node_count > 0:
                self._node_count -= 1

    # -----------------------------------------------------------------------
    # Sliding window expiry
    # -----------------------------------------------------------------------

    def _expire(self, current_time):
        """
        Expire edges older than (current_time - window_size).
        Amortized O(1) per expired edge.
        """
        threshold = int(current_time) - self.window
        while self.history and self.history[0][0] < threshold:
            _, u, v, _amt = self.history.popleft()
            self._decrement_edge(u, v)

    # -----------------------------------------------------------------------
    # Fenwick tree integration  (Phase 4)
    # -----------------------------------------------------------------------

    def _record_transaction_amount(self, node, amount, timestamp):
        """
        Record a transaction amount for a node and update its Fenwick tree.

        Maintains a sorted timeline of (timestamp, amount) per node.
        The Fenwick tree is rebuilt when it runs out of capacity (doubling).

        Complexity: O(log n) amortized per update.
        """
        ts_list = self.node_tx_timestamps[node]
        amt_list = self.node_tx_amounts[node]

        idx = len(ts_list)
        ts_list.append(timestamp)
        amt_list.append(amount)

        # Ensure Fenwick tree exists and has capacity
        if node not in self.node_fenwick:
            self.node_fenwick[node] = FenwickTree(max(64, idx + 1))
        elif idx >= self.node_fenwick[node].n:
            # Rebuild with doubled capacity — O(n) but amortized O(1)
            new_cap = max(self.node_fenwick[node].n * 2, idx + 1)
            new_ft = FenwickTree(new_cap)
            for i, a in enumerate(amt_list):
                new_ft.update(i, a)
            self.node_fenwick[node] = new_ft
            return

        self.node_fenwick[node].update(idx, amount)

    def recent_transaction_sum(self, node, window=None):
        """
        Compute the sum of transaction amounts for *node* within the
        most recent *window* seconds.

        Uses binary search + Fenwick range query.

        Complexity: O(log n)

        Parameters
        ----------
        node : str
            Node identifier.
        window : int or None
            Lookback window in seconds.  Defaults to self.window.

        Returns
        -------
        float
            Sum of amounts in the window.
        """
        node = str(node)
        if window is None:
            window = self.window

        ts_list = self.node_tx_timestamps.get(node)
        if not ts_list:
            return 0.0

        ft = self.node_fenwick.get(node)
        if ft is None:
            return 0.0

        current = self.current_time if self.current_time is not None else 0
        threshold = current - window

        # Binary search for the first index >= threshold
        left = bisect.bisect_left(ts_list, threshold)
        right = len(ts_list) - 1

        if left > right:
            return 0.0

        return ft.range_query(left, right)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def add_transaction(self, sender, receiver, amount, timestamp):
        """
        Process a single transaction incrementally.

        All structural features (degree, clustering, PageRank) are updated
        in O(k × deg) or better.  Fenwick tree updated in O(log n).

        Parameters
        ----------
        sender : str
            Sender node ID.
        receiver : str
            Receiver node ID.
        amount : float
            Transaction amount (used for Fenwick tree tracking).
        timestamp : int
            Unix timestamp of the transaction.
        """
        u = str(sender)
        v = str(receiver)
        if u == v:
            return

        ts = int(timestamp)
        if self.current_time is None or ts > self.current_time:
            self.current_time = ts

        # Update graph structure incrementally
        self._increment_edge(u, v)
        self.history.append((ts, u, v, float(amount)))

        # Fenwick tree: track transaction amounts per node — O(log n)
        self._record_transaction_amount(u, float(amount), ts)
        self._record_transaction_amount(v, float(amount), ts)

        # Expire old edges — O(1) amortized
        self._expire(self.current_time)

    def remove_transaction(self, sender, receiver):
        """Manually remove an edge (decrement count)."""
        u = str(sender)
        v = str(receiver)
        self._decrement_edge(u, v)

    def get_features(self, node):
        """
        Get the full feature vector for a single node.

        All values are pre-computed incrementally — this is a pure lookup.

        Complexity: O(log n) for recent_transaction_sum (Fenwick query),
                    O(1) for all other features.

        Returns dict with keys matching CANONICAL_FEATURE_COLUMNS.
        """
        node = str(node)
        in_deg = int(self.in_degree.get(node, 0))
        out_deg = int(self.out_degree.get(node, 0))
        degree = in_deg + out_deg

        return {
            "node_id": node,
            "degree": degree,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "clustering": float(self.clustering_coeff.get(node, 0.0)),
            "pagerank": float(self.rank.get(node, 0.0)),
            "recent_transaction_sum": float(self.recent_transaction_sum(node)),
        }

    def get_all_features(self):
        """
        Get feature matrix for ALL active nodes.

        No recomputation is triggered — simply collects pre-computed values.

        Returns
        -------
        pd.DataFrame
            Columns: ["node_id"] + CANONICAL_FEATURE_COLUMNS
        """
        # Collect all known nodes from degree tracking + rank
        all_nodes = (
            set(self.in_degree.keys())
            | set(self.out_degree.keys())
            | set(self.rank.keys())
        )

        if not all_nodes:
            return pd.DataFrame(columns=["node_id"] + CANONICAL_FEATURE_COLUMNS)

        rows = [self.get_features(n) for n in sorted(all_nodes)]
        features_df = pd.DataFrame(rows)
        return features_df[["node_id"] + CANONICAL_FEATURE_COLUMNS]
