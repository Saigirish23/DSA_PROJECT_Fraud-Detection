"""GNN model definition using GAT + GraphSAGE fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv

import config

logger = config.setup_logging(__name__)


class GATLayer(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, heads, dropout):
        super().__init__()
        self.conv1 = GATConv(in_ch, hidden_ch, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_ch * heads, out_ch, heads=1, concat=False, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_ch * heads)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = self.drop(x)
        x = self.conv2(x, edge_index)
        return x


class SAGELayer(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, dropout):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hidden_ch)
        self.conv2 = SAGEConv(hidden_ch, out_ch)
        self.bn1 = nn.BatchNorm1d(hidden_ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.drop(x)
        x = self.conv2(x, edge_index)
        return x


class FraudGCN(nn.Module):
    """Compatibility class name with reference-style GAT+SAGE internals."""

    def __init__(self, num_features, hidden_dim=None, num_classes=None, dropout=None):
        super(FraudGCN, self).__init__()

        self.hidden_dim = hidden_dim or config.GNN_HIDDEN_DIM
        self.num_classes = num_classes or config.GNN_NUM_CLASSES
        self.dropout_rate = dropout or config.GNN_DROPOUT
        self.heads = 4

        self.gat = GATLayer(num_features, self.hidden_dim, self.hidden_dim, self.heads, self.dropout_rate)
        self.sage = SAGELayer(num_features, self.hidden_dim, self.hidden_dim, self.dropout_rate)
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

        logger.info("FraudGCN initialized (GAT+SAGE fusion):")
        logger.info("  Input: %d features → Hidden: %d → Output: %d classes",
                num_features, self.hidden_dim, self.num_classes)
        logger.info("  Dropout: %.2f", self.dropout_rate)

    def forward(self, x, edge_index):
        """
        Forward pass through the GCN.

        Args:
            x (torch.Tensor): Node feature matrix [N, F].
            edge_index (torch.Tensor): Edge list in COO format [2, E].

        Returns:
            torch.Tensor: Log-softmax probabilities [N, num_classes].
        """
        gat_out = self.gat(x, edge_index)
        sage_out = self.sage(x, edge_index)
        combined = torch.cat([gat_out, sage_out], dim=-1)
        return self.fusion(combined)

    def predict_proba(self, x, edge_index):
        logits = self.forward(x, edge_index)
        return F.softmax(logits, dim=-1)[:, 1]

    def get_embeddings(self, x, edge_index):
        """
        Extract learned node embeddings from the hidden layer.

        Useful for visualization and for the hybrid model (Strategy A).

        Args:
            x (torch.Tensor): Node feature matrix [N, F].
            edge_index (torch.Tensor): Edge list in COO format [2, E].

        Returns:
            torch.Tensor: Node embeddings [N, hidden_dim].
        """
        self.eval()
        with torch.no_grad():
            gat_out = self.gat(x, edge_index)
            sage_out = self.sage(x, edge_index)
            h = torch.cat([gat_out, sage_out], dim=-1)
        return h


if __name__ == "__main__":
    # Quick test: build model and run a forward pass with dummy data
    import sys
    sys.path.insert(0, ".")

    num_features = 6
    model = FraudGCN(num_features)

    # Create dummy data
    x = torch.randn(10, num_features)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)

    # Forward pass
    out = model(x, edge_index)
    logger.info("Output shape: %s", out.shape)
    logger.info("Output (first 3 nodes): %s", out[:3])
    logger.info("\n✅ GCN model test passed!")
