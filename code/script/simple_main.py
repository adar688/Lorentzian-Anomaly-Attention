# -*- coding: utf-8 -*-

"""
MAIN (consistent anomaly plotting):
1) Load snapshots
2) Build dynamic Node2Vec embeddings [N, T, F]
3) Run Dynhat with temporal attention
4) Short CE training on idx_train (no validation)
5) Compute IF/LOF anomaly scores over time, canonicalize to a non-negative scale,
   and generate multiple consistent plots (bars, scatter, histograms, time series)
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import from_scipy_sparse_matrix
import matplotlib.pyplot as plt

# internal modules
from models.Dynhat import Dynhat
from script.utils.dynamic_node2vec import load_manifest_and_snapshots, build_dynamic_node2vec
from script.utils.dataUtils import load_citation_data  # returns adj, features_sp, labels, idx_train, idx_val, idx_test
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# ======================
#        ARGS
# ======================
def parse_args():
    p = argparse.ArgumentParser(description="Simple Dynhat run with dynamic Node2Vec + temporal attention (CE only).")

    # --- Paths ---
    p.add_argument("--data-root", type=str, default="script/data/custom_out")

    # --- Node2Vec ---
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--walk-length", type=int, default=30)
    p.add_argument("--num-walks", type=int, default=200)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--t-max", type=int, default=None)

    # --- Dynhat-required args ---
    p.add_argument("--manifold", type=str, default="Hyperboloid")
    p.add_argument("--fix_curvature", action="store_true", default=False)
    p.add_argument("--curvature", type=float, default=1.0)
    p.add_argument("--c0", type=float, default=1.0)

    p.add_argument("--nhid", type=int, default=32)
    p.add_argument("--nout", type=int, default=32)
    p.add_argument("--heads", type=int, default=1)  # structural heads
    p.add_argument("--temporal_attention_layer_heads", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--aggregation", type=str, default="att", choices=["att"])

    # Temporal layer selector
    p.add_argument("--seq-model", dest="seq_model", type=str, default="Attention")

    # --- Training ---
    p.add_argument("--max-epoch", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--norm-scale", type=float, default=0.1)

    # --- Device ---
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


# ======================
#   CONSISTENCY HELPERS
# ======================
def _normalize_minmax(M: np.ndarray) -> np.ndarray:
    """
    Normalize all scores to [0, 1] range for consistent visualization across methods.
    Avoids extreme scaling from large negative or infinite values.
    """
    if M is None:
        return None
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    vmin, vmax = np.min(M), np.max(M)
    if vmax > vmin:
        M = (M - vmin) / (vmax - vmin)
    else:
        M = np.zeros_like(M)
    return M


def _debug_stats(name: str, arr: np.ndarray):
    """Quick sanity stats printed once to ensure alignment across plots."""
    a = np.nan_to_num(arr, nan=0.0)
    print(f"[{name}] shape={a.shape}  min={a.min():.6f}  max={a.max():.6f}  mean={a.mean():.6f}")


# ======================
#    IF/LOF OVER TIME
# ======================
def run_if_lof_over_time(att, contamination=0.05, lof_k=30, topk=20):
    """
    att: torch.Tensor of shape [N, T, C] or [N, C]

    Returns a dict containing:
      - mu_if, std_if, mu_lof, std_lof: per-node statistics (after canonicalization)
      - top indices: top_mu_if_idx, top_std_if_idx, top_mu_lof_idx, top_std_lof_idx
      - AS_if, AS_lof: full time-series matrices [N, T] (after canonicalization)
    """
    # Ensure [N, T, C]
    if att.dim() == 2:
        att = att.unsqueeze(1)
    A = att.detach().cpu().numpy()  # [N, T, C]
    N, T, C = A.shape

    AS_if  = np.full((N, T), np.nan, dtype=np.float32)
    AS_lof = np.full((N, T), np.nan, dtype=np.float32)

    lof_k = max(2, min(lof_k, N - 1))
    contamination = float(np.clip(contamination, 1.0 / max(N, 1), 0.2))

    # Per-timeframe anomaly scoring
    for t in range(T):
        X_t = A[:, t, :]
        X_t = np.nan_to_num(X_t, nan=0.0, posinf=1e6, neginf=-1e6)
        col_std = X_t.std(axis=0)
        informative = col_std > 0
        if informative.sum() == 0:
            continue
        X_t = X_t[:, informative]
        X_t = MinMaxScaler().fit_transform(X_t)

        # Isolation Forest (negating decision_function so that larger=more anomalous)
        try:
            if_clf = IsolationForest(
                n_estimators=100,
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            ).fit(X_t)
            AS_if[:, t] = -if_clf.decision_function(X_t)
        except Exception:
            pass

        # Local Outlier Factor (convert to larger=more anomalous)
        try:
            lof = LocalOutlierFactor(
                n_neighbors=lof_k,
                contamination=contamination,
                novelty=False,
                n_jobs=-1
            )
            _ = lof.fit_predict(X_t)
            AS_lof[:, t] = -(lof.negative_outlier_factor_)
        except Exception:
            pass

    # Canonicalize both methods to non-negative scale (global min -> 0)
    AS_if  = _normalize_minmax(AS_if)
    AS_lof = _normalize_minmax(AS_lof)


    _debug_stats("AS_if (canon)", AS_if)
    _debug_stats("AS_lof (canon)", AS_lof)

    # Per-node summaries on the canonized matrices
    mu_if   = np.nanmean(AS_if,  axis=1)
    std_if  = np.nanstd(AS_if,   axis=1)
    mu_lof  = np.nanmean(AS_lof, axis=1)
    std_lof = np.nanstd(AS_lof,  axis=1)

    def topk_idx(v, k):
        v = np.nan_to_num(v, nan=-1e30)
        k = min(k, len(v))
        return np.argsort(-v)[:k]

    return {
        "mu_if":   mu_if,
        "std_if":  std_if,
        "mu_lof":  mu_lof,
        "std_lof": std_lof,
        "top_mu_if_idx":   topk_idx(mu_if,  topk),
        "top_std_if_idx":  topk_idx(std_if, topk),
        "top_mu_lof_idx":  topk_idx(mu_lof, topk),
        "top_std_lof_idx": topk_idx(std_lof, topk),
        "AS_if": AS_if,
        "AS_lof": AS_lof,
    }


# ======================
#        PLOTS
# ======================
# def plot_anomaly_scores(mu_if, std_if, mu_lof, std_lof, top_k=None, save_dir="plots"):
#     """
#     Scatter plots of anomaly stats (mean/std) per node for IF and LOF.
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     N = len(mu_if)
#     x = np.arange(N)

#     # Mean per node
#     plt.figure(figsize=(12, 6))
#     plt.scatter(x, mu_if, s=8, color='blue', alpha=0.6, label='IF mean (μ)')
#     plt.scatter(x, mu_lof, s=8, color='orange', alpha=0.6, label='LOF mean (μ)')
#     plt.xlabel("Node ID (i)")
#     plt.ylabel("Mean anomaly score")
#     plt.title("Mean (μ) anomaly scores per node (IF vs LOF)")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, "anomaly_mean_scores.png"))
#     plt.close()
#     print(f"💾 Saved: {os.path.join(save_dir, 'anomaly_mean_scores.png')}")

#     # Std per node
#     plt.figure(figsize=(12, 6))
#     plt.scatter(x, std_if, s=8, color='green', alpha=0.6, label='IF std (σ)')
#     plt.scatter(x, std_lof, s=8, color='red', alpha=0.6, label='LOF std (σ)')
#     plt.xlabel("Node ID (i)")
#     plt.ylabel("Standard deviation of anomaly score")
#     plt.title("Standard deviation (σ) of anomaly scores per node (IF vs LOF)")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, "anomaly_std_scores.png"))
#     plt.close()
#     print(f"💾 Saved: {os.path.join(save_dir, 'anomaly_std_scores.png')}")

#     # Top-K bars (by IF mean)
#     if top_k is not None:
#         top_nodes = np.argsort(-mu_if)[:top_k]
#         plt.figure(figsize=(10, 5))
#         plt.bar(top_nodes, mu_if[top_nodes], color='steelblue', label='Top IF mean')
#         plt.bar(top_nodes, std_if[top_nodes], color='lightcoral', alpha=0.6, label='Top IF std')
#         plt.xlabel("Node ID (Top-K by IF mean)")
#         plt.ylabel("Score value")
#         plt.title(f"Top-{top_k} nodes by IF mean & std")
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, f"top_{top_k}_if_scores.png"))
#         plt.close()
#         print(f"💾 Saved: {os.path.join(save_dir, f'top_{top_k}_if_scores.png')}")


def plot_mean_std(mu: np.ndarray, std: np.ndarray, top_k: int = 10, save_dir: str = "plots"):
    """
    Two separate bar charts:
      1) Mean (μ) per node, highlighting Top-K
      2) Std (σ) per node, highlighting Top-K
    """
    os.makedirs(save_dir, exist_ok=True)

    N = len(mu)
    x = np.arange(N)

    mu = np.nan_to_num(mu, nan=0.0, posinf=np.max(mu[np.isfinite(mu)]) if np.any(np.isfinite(mu)) else 0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=0.0, posinf=np.max(std[np.isfinite(std)]) if np.any(np.isfinite(std)) else 0.0, neginf=0.0)

    # Mean
    plt.figure(figsize=(12, 6))
    plt.bar(x, mu, color='skyblue', alpha=0.7, label='Mean (μ)')
    top_mean_idx = np.argsort(-mu)[:top_k]
    plt.scatter(top_mean_idx, mu[top_mean_idx], color='red', s=80, label=f'Top {top_k} Mean')
    plt.title('Isolation Forest — Mean (μ) per Node', fontsize=14)
    plt.xlabel('Node index (i)', fontsize=12)
    plt.ylabel('Mean anomaly score', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    mean_path = os.path.join(save_dir, "plot_mean.png")
    plt.savefig(mean_path, dpi=150)
    plt.close()
    print(f"💾 Saved: {mean_path}")

    # Std
    plt.figure(figsize=(12, 6))
    plt.bar(x, std, color='orange', alpha=0.7, label='Std Dev (σ)')
    top_std_idx = np.argsort(-std)[:top_k]
    plt.scatter(top_std_idx, std[top_std_idx], color='purple', s=80, label=f'Top {top_k} Std')
    plt.title('Isolation Forest — Standard Deviation (σ) per Node', fontsize=14)
    plt.xlabel('Node index (i)', fontsize=12)
    plt.ylabel('Standard deviation of anomaly score', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    std_path = os.path.join(save_dir, "plot_std.png")
    plt.savefig(std_path, dpi=150)
    plt.close()
    print(f"💾 Saved: {std_path}")


def plot_nodes_hist_mean(mu: np.ndarray, method: str = "IF", save_dir: str = "plots"):
    """
    Bar chart over all nodes: X = node id, Y = mean anomaly score (μ).
    """
    os.makedirs(save_dir, exist_ok=True)
    N = len(mu)
    x = np.arange(N)

    mu = np.nan_to_num(mu, nan=0.0, posinf=np.max(mu[np.isfinite(mu)]) if np.any(np.isfinite(mu)) else 0.0, neginf=0.0)

    plt.figure(figsize=(14, 6))
    plt.bar(x, mu, alpha=0.8)
    plt.title(f"{method} — Mean (μ) per node")
    plt.xlabel("Node ID")
    plt.ylabel("Mean anomaly score")
    plt.tight_layout()
    out = os.path.join(save_dir, f"{method.lower()}_nodes_mean_bar.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"💾 Saved: {out}")


def plot_nodes_hist_std(std: np.ndarray, method: str = "IF", save_dir: str = "plots"):
    """
    Bar chart over all nodes: X = node id, Y = std of anomaly score (σ).
    """
    os.makedirs(save_dir, exist_ok=True)
    N = len(std)
    x = np.arange(N)

    std = np.nan_to_num(std, nan=0.0, posinf=np.max(std[np.isfinite(std)]) if np.any(np.isfinite(std)) else 0.0, neginf=0.0)

    plt.figure(figsize=(14, 6))
    plt.bar(x, std, alpha=0.8)
    plt.title(f"{method} — Std (σ) per node")
    plt.xlabel("Node ID")
    plt.ylabel("Std of anomaly score")
    plt.tight_layout()
    out = os.path.join(save_dir, f"{method.lower()}_nodes_std_bar.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"💾 Saved: {out}")


def plot_top_node_timeseries_by_method(
    AS: np.ndarray,
    mu: np.ndarray,
    method_name: str,
    save_dir: str = "plots"
) -> int:
    """
    Plot time-series for the most anomalous node according to the given method.
    - AS: [N, T] anomaly matrix for the method (already normalized to [0,1])
    - mu: [N] per-node mean anomaly scores for the method
    - method_name: "IF" or "LOF" (used for labels and filename)
    Returns: selected node index.
    """
    os.makedirs(save_dir, exist_ok=True)
    if AS is None or mu is None or len(mu) == 0:
        print(f"[{method_name}] No data to plot.")
        return -1

    node_idx = int(np.argmax(np.nan_to_num(mu, nan=-1e30)))
    y = AS[node_idx, :]
    t = np.arange(y.shape[0])

    plt.figure(figsize=(12, 5))
    plt.plot(t, y, marker='o', linewidth=1.5, label=method_name)
    plt.title(f"Most anomalous node time-series by {method_name} (node {node_idx})")
    plt.xlabel("Time (snapshot index)")
    plt.ylabel("Anomaly score")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(save_dir, f"top_node_{method_name.lower()}_{node_idx}_timeseries.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"💾 Saved: {out}")
    return node_idx



def plot_hist_distribution(values: np.ndarray, title: str, xlabel: str, save_path: str):
    """
    Statistical histogram of anomaly values across nodes.
    """
    values = np.nan_to_num(values, nan=0.0)
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of nodes")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"💾 Saved: {save_path}")


# ===============
#      MAIN
# ===============
def main():
    args = parse_args()

    args.fix_curvature = True  # required so Dynhat creates self.c
    if not hasattr(args, "curvature"):
        args.curvature = 1.0
    if not hasattr(args, "device"):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    # 1) Load snapshots + dynamic Node2Vec
    num_nodes, T_bins, snapshots = load_manifest_and_snapshots(args.data_root)
    if args.t_max is not None:
        T_bins = min(T_bins, args.t_max)
        snapshots = {t: snapshots[t] for t in range(T_bins)}

    embedding_matrix = build_dynamic_node2vec(
        snapshots=snapshots,
        num_nodes=num_nodes,
        T=T_bins,
        emb_dim=args.emb_dim,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        workers=args.workers,
        window=args.window,
    )  # torch.Tensor [N, T, F]
    print(f"✅ embedding_matrix shape: {tuple(embedding_matrix.shape)}")

    args.nfeat = int(embedding_matrix.shape[-1])     # F
    args.num_nodes = int(embedding_matrix.shape[0])  # N

    # 2) Load graph/labels + splits
    adj, features_sp, labels_np, idx_train, idx_val, idx_test = load_citation_data(
        dataset_str="dblpv13",
        use_feats=True,
        data_path=args.data_root
    )
    edge_index, _ = from_scipy_sparse_matrix(adj)
    edge_index = edge_index.to(device)

    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    if labels.ndim == 2 and labels.shape[1] > 1:
        labels = labels.argmax(dim=1)
    args.num_classes = int(labels.max().item() + 1)

    # 3) Dynhat + classifier head
    model = Dynhat(args, time_length=T_bins).to(device)
    cls_head = torch.nn.Linear(args.nhid + 1, args.num_classes).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(cls_head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    print(f"🔧 Dynhat ready (nhid={args.nhid}, temporal_heads={args.temporal_attention_layer_heads}, classes={args.num_classes})")

    # 4) Training (no validation)
    model.train(); cls_head.train()
    for epoch in range(args.max_epoch):
        optimizer.zero_grad()

        temporal_outputs = []
        for t in range(T_bins):
            x_t = embedding_matrix[:, t, :].to(device)  # [N, F]
            x_t = torch.nn.functional.normalize(x_t, p=2, dim=1) * float(args.norm_scale)
            h_t = model(edge_index, x=x_t)              # [N, C=nhid+1]
            temporal_outputs.append(h_t)

        X = torch.stack(temporal_outputs, dim=1)        # [N, T, C]
        att = model.seq_model(X)                        # [N, T, C] or [N, C]
        if att.ndim == 2:
            att = att.unsqueeze(1)                      # [N, 1, C]

        feat_last = att[:, -1, :]                       # [N, C]
        logits = cls_head(feat_last)                    # [N, num_classes]
        loss = F.cross_entropy(logits[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        print(f"[Epoch {epoch}] Train Loss: {loss.item():.4f}")

    # 5) Final representations (optional)
    model.eval(); cls_head.eval()
    with torch.no_grad():
        outs = []
        for t in range(T_bins):
            x_t = embedding_matrix[:, t, :].to(device)
            x_t = torch.nn.functional.normalize(x_t, p=2, dim=1) * float(args.norm_scale)
            h_t = model(edge_index, x=x_t)
            outs.append(h_t)
        X_eval = torch.stack(outs, dim=1)     # [N, T, C]
        att_out = model.seq_model(X_eval)     # [N, T, C] or [N, C]
        if att_out.ndim == 2:
            att_out = att_out.unsqueeze(1)

    print("✅ Done. Shapes:",
          f"N={att_out.shape[0]}, T={att_out.shape[1]}, C={att_out.shape[2]}")
    # torch.save({"att_output": att_out.cpu()}, "att_output.pt")

    # 6) Anomaly over time (IF/LOF) + canonicalization
    res = run_if_lof_over_time(att_out, contamination=0.05, lof_k=30, topk=20)
    print("Top-20 IF(mean):",   res["top_mu_if_idx"])
    print("Top-20 IF(std):",    res["top_std_if_idx"])
    print("Top-20 LOF(mean):",  res["top_mu_lof_idx"])
    print("Top-20 LOF(std):",   res["top_std_lof_idx"])

    print("Min std_if:", np.min(res["std_if"]))
    print("Min std_lof:", np.min(res["std_lof"]))

    common_top = set(res["top_mu_if_idx"]) & set(res["top_mu_lof_idx"])
    print(f"✅ Common top anomalies between IF and LOF: {len(common_top)} nodes")
    print(f"Common indices: {sorted(list(common_top))}")

    # 7) Plots (all consistent, non-negative scale)
    plot_mean_std(mu=res["mu_if"], std=res["std_if"], top_k=20, save_dir="plots")

    # Bar charts per node
    # plot_nodes_hist_mean(res["mu_if"], method="IF", save_dir="plots")
    # plot_nodes_hist_std(res["std_if"], method="IF", save_dir="plots")

    # plot_nodes_hist_mean(res["mu_lof"], method="LOF", save_dir="plots")
    # plot_nodes_hist_std(res["std_lof"], method="LOF", save_dir="plots")

    if "AS_if" in res and res["AS_if"] is not None:
        plot_top_node_timeseries_by_method(
            AS=res["AS_if"],
            mu=res["mu_if"],
            method_name="IF",
            save_dir="plots"
        )

    if "AS_lof" in res and res["AS_lof"] is not None:
        plot_top_node_timeseries_by_method(
            AS=res["AS_lof"],
            mu=res["mu_lof"],
            method_name="LOF",
            save_dir="plots"
        )
    # Statistical histograms of distributions
    plot_hist_distribution(res["mu_if"], "Distribution of Mean (μ) anomaly scores - IF", "Mean (μ) value", "plots/hist_mean_if.png")
    plot_hist_distribution(res["std_if"], "Distribution of Std (σ) anomaly scores - IF", "Std (σ) value", "plots/hist_std_if.png")

    plot_hist_distribution(res["mu_lof"], "Distribution of Mean (μ) anomaly scores - LOF", "Mean (μ) value", "plots/hist_mean_lof.png")
    plot_hist_distribution(res["std_lof"], "Distribution of Std (σ) anomaly scores - LOF", "Std (σ) value", "plots/hist_std_lof.png")


if __name__ == "__main__":
    main()
