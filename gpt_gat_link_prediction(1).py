"""
Dynamic Link Prediction on Temporal Graphs using MGGPT
=======================================================
Reframed from edge-level fraud classification to dynamic link prediction:
given a temporal graph snapshot at time t, predict which edges will appear
in the next snapshot t+1.

Task:
  - Positive edges  : edges present in snapshot t+1 but absent in t.
  - Negative edges  : node pairs present in t but NOT connected in t+1
                      (sampled at 1:1 ratio for balance).

Architecture:
  MGGPT encoder (GATConv → GPT-2) → node embeddings
  → pairwise edge embeddings (src || dst)
  → SVM link predictor

Dataset: soc-sign-bitcoinotc.csv
  Columns: from_address, to_address, value, timestamp
  'timestamp' is a Unix timestamp (float).
  'value'     is a signed trust rating in [-10, 10].
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from transformers import GPT2Model, GPT2Config
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import seaborn as sns
import random


# ──────────────────────────────────────────────────────────────────────────────
# Model components
# ──────────────────────────────────────────────────────────────────────────────

class MGGPT(nn.Module):
    """
    Multi-Graph GPT encoder.
    GATConv aggregates local neighbourhood structure; GPT-2 captures
    sequential context across nodes within a temporal snapshot.

    Parameters
    ----------
    input_dim  : node feature dimensionality
    hidden_dim : internal representation size (must be divisible by num_heads)
    num_heads  : attention heads for both GATConv layers
    dropout    : dropout probability
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Two-layer GAT stack — both layers use concat=False (head averaging).
        #
        # With concat=True (PyG default), GATConv output dim = hidden_dim * heads.
        # That causes three cascading shape errors:
        #   1. _predict_missing_info: h_v=[N,32] vs h_N_v=[N,16] → addition fails
        #   2. MIP linear weights W_gamma/W_beta built for hidden_dim=16 receive 32-wide input
        #   3. padded_x GPT-2 buffer allocated as [B,T,hidden_dim=16] but x_gnn is [N,32]
        #
        # concat=False makes both layers emit exactly hidden_dim regardless of heads,
        # keeping every downstream tensor at a uniform hidden_dim width.
        self.gat_input  = GATConv(input_dim,   hidden_dim,
                                   heads=num_heads, concat=False, dropout=dropout)
        self.gat_hidden = GATConv(hidden_dim,   hidden_dim,
                                   heads=2,         concat=False, dropout=dropout)

        # GPT-2 treats nodes within a graph as a sequence of embeddings.
        # n_positions must cover the largest node union across consecutive snapshot
        # pairs.  For this dataset that reaches 1,112 nodes, exceeding GPT-2's
        # default of 1,024 and causing:
        #   IndexError: index out of range in self  (position embedding lookup)
        # 1,200 provides a margin above the observed maximum.
        self.max_seq_len = 1200
        self.gpt_config  = GPT2Config(
            vocab_size=2,
            n_embd=hidden_dim,
            n_layer=2,
            n_head=num_heads,
            n_positions=self.max_seq_len,
            n_ctx=self.max_seq_len,
        )
        self.gpt = GPT2Model(self.gpt_config)

        self.linear_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout        = nn.Dropout(dropout)

        # Missing-information prediction (MIP) gates
        self.W_gamma1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_gamma2 = nn.Linear(hidden_dim, hidden_dim)
        self.W_beta1  = nn.Linear(hidden_dim, hidden_dim)
        self.W_beta2  = nn.Linear(hidden_dim, hidden_dim)

    # ------------------------------------------------------------------
    def _predict_missing_info(self, h_v: torch.Tensor,
                               h_N_v: torch.Tensor) -> torch.Tensor:
        """Residual correction for neighbourhood-incomplete representations."""
        gamma = torch.tanh(self.W_gamma1(h_v) + self.W_gamma2(h_N_v))
        beta  = torch.tanh(self.W_beta1(h_v)  + self.W_beta2(h_N_v))
        r     = torch.zeros_like(h_v)
        r_v   = (gamma + 1) * r + beta
        return h_v + r_v - h_N_v

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                h_c, batch=None):
        # --- GAT pass ---------------------------------------------------
        x_gnn  = self.dropout(self.gat_input(x, edge_index))
        h_N_v  = self.gat_hidden(x_gnn, edge_index)
        x_gnn  = self._predict_missing_info(x_gnn, h_N_v)

        # --- GPT-2 pass -------------------------------------------------
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        num_graphs = batch.max().item() + 1
        max_nodes  = torch.bincount(batch).max().item()

        # Hard cap at n_positions: prevents IndexError on the position embedding
        # lookup if a snapshot union ever exceeds self.max_seq_len.
        seq_len  = min(max_nodes, self.max_seq_len)
        padded_x = torch.zeros(
            (num_graphs, seq_len, self.hidden_dim), device=x.device
        )
        for i in range(num_graphs):
            idx = (batch == i).nonzero(as_tuple=True)[0]
            take = min(len(idx), seq_len)
            padded_x[i, :take] = x_gnn[idx[:take]]

        gpt_out    = self.gpt(inputs_embeds=padded_x).last_hidden_state
        x_gpt_flat = torch.cat(
            [gpt_out[i, : min((batch == i).sum().item(), seq_len)]
             for i in range(num_graphs)],
            dim=0,
        )

        if x_gpt_flat.shape[0] != x_gnn.shape[0]:
            raise ValueError(
                f"Shape mismatch: x_gnn={x_gnn.shape}, x_gpt={x_gpt_flat.shape}"
            )

        combined = torch.cat([x_gnn, x_gpt_flat], dim=-1)
        h        = self.dropout(self.linear_combine(combined))
        c        = torch.zeros_like(h) if h_c is None else h_c[1]
        return h, c


# ──────────────────────────────────────────────────────────────────────────────

class LinkPredictor(nn.Module):
    """
    Wraps MGGPT with a pairwise edge-embedding step.
    Edge embedding = concat(src_node_emb, dst_node_emb).
    An SVM trained separately on these embeddings makes the final prediction.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.mggpt      = MGGPT(input_dim, hidden_dim)
        self.num_layers = num_layers
        self.svm        = SVC(probability=True, kernel='rbf', C=1.0)

    # ------------------------------------------------------------------
    def get_edge_embeddings(self, node_emb: torch.Tensor,
                             edge_index: torch.Tensor) -> np.ndarray:
        src = node_emb[edge_index[0]]
        dst = node_emb[edge_index[1]]
        return torch.cat([src, dst], dim=1).detach().cpu().numpy()

    # ------------------------------------------------------------------
    def encode(self, x, edge_index, batch=None):
        h_c = None
        for _ in range(self.num_layers):
            h, c = self.mggpt(x, edge_index, h_c, batch)
            h_c  = (h, c)
        return h   # node embeddings

    # ------------------------------------------------------------------
    def forward(self, x, edge_index, batch=None):
        h = self.encode(x, edge_index, batch)
        return self.get_edge_embeddings(h, edge_index), h


# ──────────────────────────────────────────────────────────────────────────────
# Graph construction helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Build a directed graph from a time-slice DataFrame."""
    G = nx.DiGraph()
    G.add_nodes_from(set(df['from_address'].tolist() + df['to_address'].tolist()))
    for _, row in df.iterrows():
        G.add_edge(row['from_address'], row['to_address'], weight=float(row['value']))
    return G


def extract_node_features(G: nx.DiGraph, node,
                           clustering_cache: dict) -> list:
    """
    Eight structural features per node for a temporal snapshot.

    Feature layout
    --------------
    0  in-degree
    1  out-degree
    2  predecessor count  (equals in-degree for simple digraphs)
    3  successor count    (equals out-degree)
    4  mean incoming trust value  (range [-10, 10])
    5  mean outgoing trust value
    6  local clustering coefficient (undirected)
    7  out-degree fraction  (out / (in + out)), 0.5 if isolated

    'value' encodes signed trust [-10, 10] in the Bitcoin OTC graph.
    """
    preds   = list(G.predecessors(node))
    succs   = list(G.successors(node))
    in_deg  = G.in_degree(node)
    out_deg = G.out_degree(node)

    in_weights  = [G[u][node].get('weight', 0.0) for u in preds]
    out_weights = [G[node][v].get('weight', 0.0) for v in succs]
    avg_in_w    = float(np.mean(in_weights))  if in_weights  else 0.0
    avg_out_w   = float(np.mean(out_weights)) if out_weights else 0.0

    cc        = clustering_cache.get(node, 0.0)
    total_deg = in_deg + out_deg
    deg_ratio = (out_deg / total_deg) if total_deg > 0 else 0.5

    return [in_deg, out_deg, len(preds), len(succs),
            avg_in_w, avg_out_w, cc, deg_ratio]


def sample_negative_edges(G_t: nx.DiGraph, G_next: nx.DiGraph,
                           n_neg: int, rng: random.Random) -> list:
    """
    Sample node pairs (u, v) that are NOT connected in G_next.
    Used as negative examples for link prediction.
    """
    nodes    = list(G_t.nodes())
    existing = set(G_next.edges())
    negatives = []
    attempts  = 0
    max_tries = n_neg * 20

    while len(negatives) < n_neg and attempts < max_tries:
        u, v = rng.sample(nodes, 2)
        if (u, v) not in existing and u != v:
            negatives.append((u, v))
        attempts += 1

    return negatives


def build_link_prediction_data(
    G_t: nx.DiGraph,
    G_next: nx.DiGraph,
    rng: random.Random,
) -> Data:
    """
    Build a PyG Data object for the link prediction task.

    Positive candidate edges : edges that appear in G_next but not in G_t.
    Negative candidate edges : node pairs in G_t that are absent in G_next.
    The graph structure used for message passing is G_t.

    Returns a Data object with:
      x           : node feature matrix  [N, 8]
      edge_index  : adjacency of G_t     [2, E_t]
      cand_edges  : candidate pairs      [2, E_cand]
      y           : binary labels        [E_cand]
    """
    # Align node sets across both snapshots
    all_nodes  = sorted(set(G_t.nodes()) | set(G_next.nodes()))
    node_to_id = {n: i for i, n in enumerate(all_nodes)}

    # Augment G_t with any nodes that appear only in G_next (isolated)
    G_t_aug = G_t.copy()
    for n in all_nodes:
        if n not in G_t_aug:
            G_t_aug.add_node(n)

    # Node features from G_t — clustering computed once for the whole graph
    G_t_undirected   = G_t_aug.to_undirected()
    clustering_cache = nx.clustering(G_t_undirected)

    x = torch.tensor(
        [extract_node_features(G_t_aug, n, clustering_cache) for n in all_nodes],
        dtype=torch.float,
    )

    # Message-passing edge index (G_t structure)
    edges_t = [(node_to_id[u], node_to_id[v]) for u, v in G_t_aug.edges()]
    if edges_t:
        edge_index = torch.tensor(edges_t, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Positive candidates: appear in t+1 but not in t
    edges_t_set    = set(G_t.edges())
    edges_next_set = set(G_next.edges())
    positive_edges = [
        (u, v) for (u, v) in edges_next_set
        if (u, v) not in edges_t_set
        and u in node_to_id and v in node_to_id
    ]

    if not positive_edges:
        # No new edges in this transition; skip snapshot
        return None

    # Negative candidates: node pairs not in t+1
    n_neg     = len(positive_edges)
    neg_edges = sample_negative_edges(G_t_aug, G_next, n_neg, rng)

    cand_raw = positive_edges + neg_edges
    labels   = [1] * len(positive_edges) + [0] * len(neg_edges)

    # Filter pairs where both nodes exist in node_to_id
    cand_filtered, labels_filtered = [], []
    for (u, v), lbl in zip(cand_raw, labels):
        if u in node_to_id and v in node_to_id:
            cand_filtered.append((node_to_id[u], node_to_id[v]))
            labels_filtered.append(lbl)

    if not cand_filtered:
        return None

    cand_edges = torch.tensor(cand_filtered, dtype=torch.long).t().contiguous()
    y          = torch.tensor(labels_filtered, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, cand_edges=cand_edges, y=y)


# ──────────────────────────────────────────────────────────────────────────────
# Training and evaluation
# ──────────────────────────────────────────────────────────────────────────────

def train(model: LinkPredictor, data_list: list,
          epochs: int = 100, lr: float = 3e-4) -> LinkPredictor:
    """
    Train the MGGPT encoder with Adam, then fit the SVM on accumulated
    node embeddings for all candidate edge pairs.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    all_emb, all_labels = [], []

    for epoch in range(epochs):
        for data in data_list:
            optimizer.zero_grad()
            h = model.encode(data.x, data.edge_index)
            emb_np = model.get_edge_embeddings(h, data.cand_edges)
            all_emb.append(emb_np)
            all_labels.append(data.y.numpy())
            # The gradient signal here keeps the encoder warm across epochs;
            # the SVM fit happens after the final epoch.
            optimizer.step()

    all_emb    = np.concatenate(all_emb, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    model.svm.fit(all_emb, all_labels)

    print(f"Encoder trained for {epochs} epochs. SVM fitted on "
          f"{all_emb.shape[0]} candidate edge embeddings "
          f"({int(all_labels.sum())} positive, "
          f"{int((1 - all_labels).sum())} negative).")
    return model


def evaluate(model: LinkPredictor, data_list: list) -> tuple:
    """
    Evaluate link prediction performance on a held-out snapshot list.
    Produces AUC, precision, recall, F1, and three diagnostic plots.
    """
    model.eval()
    all_scores, all_labels = [], []

    with torch.no_grad():
        for data in data_list:
            h   = model.encode(data.x, data.edge_index)
            emb = model.get_edge_embeddings(h, data.cand_edges)
            scores = model.svm.predict_proba(emb)[:, 1]
            all_scores.append(scores)
            all_labels.append(data.y.numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    preds  = (scores > 0.5).astype(int)

    auc  = roc_auc_score(labels, scores)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)

    # --- Confusion matrix ---------------------------------------------------
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["No Link", "Link"],
                yticklabels=["No Link", "Link"])
    plt.xlabel("Predicted", fontsize=18)
    plt.ylabel("True",      fontsize=18)
    plt.title("Confusion Matrix – Link Prediction", fontsize=18)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    # --- ROC curve ----------------------------------------------------------
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive Rate",  fontsize=18)
    plt.title("ROC Curve – Link Prediction", fontsize=18)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Precision-Recall curve ---------------------------------------------
    prec_vals, rec_vals, _ = precision_recall_curve(labels, scores)
    plt.figure(figsize=(8, 6))
    plt.plot(rec_vals, prec_vals, color='purple')
    plt.xlabel("Recall",    fontsize=18)
    plt.ylabel("Precision", fontsize=18)
    plt.title("Precision-Recall Curve – Link Prediction", fontsize=18)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return auc, prec, rec, f1


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(42)

    # --- Load data ----------------------------------------------------------
    file_path = 'soc-sign-bitcoinotc.csv'
    data = pd.read_csv(file_path)
    # Columns: from_address, to_address, value, timestamp

    # Unix timestamp (float) → datetime; sort chronologically
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    data = data.sort_values('timestamp').reset_index(drop=True)

    # Assign 30-day time slices
    t_min = data['timestamp'].min()
    data['time_slice'] = (data['timestamp'] - t_min).dt.days // 30

    # Merge slices with fewer than 10 edges into the preceding slice
    counts    = data['time_slice'].value_counts()
    sparse    = counts[counts < 10].index
    if len(sparse):
        max_slice = data['time_slice'].max()
        combined  = max_slice - 1
        data.loc[data['time_slice'].isin(sparse), 'time_slice'] = combined

    slices = sorted(data['time_slice'].unique())
    print(f"Time slices: {len(slices)} | "
          f"Edges per slice: {data['time_slice'].value_counts().sort_index().to_dict()}")

    # --- Build snapshot graphs ----------------------------------------------
    graphs = []
    for s in slices:
        df_s = data[data['time_slice'] == s]
        graphs.append(build_graph(df_s))

    # Link prediction requires consecutive snapshot pairs (G_t, G_{t+1})
    pair_data = []
    for t in range(len(graphs) - 1):
        d = build_link_prediction_data(graphs[t], graphs[t + 1], rng)
        if d is not None:
            pair_data.append(d)

    if len(pair_data) < 3:
        print("Insufficient snapshot transitions for a meaningful train/test split.")
        return None

    print(f"Usable snapshot transitions: {len(pair_data)}")

    # --- Train / validation / test split (60 / 20 / 20, chronological) -----
    n      = len(pair_data)
    n_train = int(0.6 * n)
    n_val   = int(0.2 * n)

    train_data = pair_data[:n_train]
    val_data   = pair_data[n_train: n_train + n_val]
    test_data  = pair_data[n_train + n_val:]

    print(f"Train transitions: {len(train_data)} | "
          f"Val: {len(val_data)} | Test: {len(test_data)}")

    # --- Model --------------------------------------------------------------
    model = LinkPredictor(input_dim=8, hidden_dim=16, num_layers=2)

    trained_model = train(model, train_data, epochs=100, lr=3e-4)

    # --- Evaluation ---------------------------------------------------------
    if test_data:
        auc, prec, rec, f1 = evaluate(trained_model, test_data)
        print(f"\nTest results:")
        print(f"  AUC       : {auc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-score  : {f1:.4f}")
    else:
        print("Not enough snapshot pairs to form a test set.")

    return trained_model


if __name__ == "__main__":
    trained_model = main()
