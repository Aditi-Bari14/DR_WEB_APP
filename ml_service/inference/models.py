# inference/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch_geometric.nn import GCNConv

# --- GCN Encoder ---
class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=128, dropout=0.0):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# --- Proto Layer ---
class ProtoLayer(nn.Module):
    def __init__(self, num_prototypes, proto_dim):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, proto_dim))

    def forward(self, x):
        dist = torch.sum((x.unsqueeze(1) - self.prototypes.unsqueeze(0))**2, dim=2)
        sim = -dist
        return sim, dist

# --- Full CNN+GCN+Proto Fusion Model ---
class CNN_GCN_Proto(nn.Module):
    def __init__(self,
                 num_tab_features,
                 num_classes,
                 proto_dim=256,
                 num_prototypes=20,
                 gcn_in_dim=6,
                 gcn_hidden=128,
                 gcn_out=128,
                 tab_mlp_hidden=(64, 32),
                 pretrained=True,
                 dropout=0.3):
        super().__init__()

        # CNN backbone (DenseNet121)
        self.cnn = models.densenet121(pretrained=pretrained)
        in_features = self.cnn.classifier.in_features
        self.cnn.classifier = nn.Identity()

        # Projection from CNN features -> proto_dim
        self.img_proj = nn.Linear(in_features, proto_dim)

        # Proto layer
        self.proto = ProtoLayer(num_prototypes=num_prototypes, proto_dim=proto_dim)
        self.num_prototypes = num_prototypes
        self.proto_dim = proto_dim

        # Tabular MLP
        self.tab_mlp = nn.Sequential(
            nn.Linear(num_tab_features, tab_mlp_hidden[0]),
            nn.ReLU(),
            nn.Linear(tab_mlp_hidden[0], tab_mlp_hidden[1]),
            nn.ReLU()
        )
        self.tab_out_dim = tab_mlp_hidden[1]

        # GCN
        self.gcn = GCNEncoder(in_dim=gcn_in_dim, hidden=gcn_hidden, out_dim=gcn_out, dropout=0.0)
        self.gcn_out_dim = gcn_out

        # Fusion classifier
        fusion_dim = proto_dim + self.tab_out_dim + self.gcn_out_dim + self.num_prototypes
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, tabular, graph_node_feats, edge_index, node_idx):
        device = image.device

        # CNN branch
        img_feat = self.cnn(image)
        proj = self.img_proj(img_feat)

        # Proto branch
        proto_sim, proto_dist = self.proto(proj)

        # Tabular branch
        tab_feat = self.tab_mlp(tabular)

        # GCN branch
        gcn_node_emb = self.gcn(graph_node_feats, edge_index)
        B = node_idx.shape[0]
        graph_feat = torch.zeros((B, self.gcn_out_dim), device=device, dtype=gcn_node_emb.dtype)

        mask = node_idx >= 0
        if mask.any():
            valid_idx = node_idx[mask].long()
            graph_feat[mask] = gcn_node_emb[valid_idx]

        # Fuse features
        fused = torch.cat([proj, tab_feat, graph_feat, proto_sim], dim=1)
        logits = self.classifier(fused)

        return {
            "logits": logits,
            "proto_sim": proto_sim,
            "proto_dist": proto_dist,
            "gcn_node_emb": gcn_node_emb
        }
