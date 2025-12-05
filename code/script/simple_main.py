# -*- coding: utf-8 -*-

"""
MAIN (consistent anomaly plotting + noise-injection validation):
1) Load snapshots
2) Build dynamic Node2Vec embeddings [N, T, F]
3) Run Dynhat with temporal attention
4) Short CE training on idx_train (no validation)
5) Compute IF/LOF anomaly scores over time, canonicalize to a non-negative scale,
   and generate multiple consistent plots (bars, scatter, histograms, time series)
6) Validation via noise injection (fake nodes) with full pipeline (Node2Vec embeddings -> Dynhat -> Attention -> IF/LOF)
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_scipy_sparse_matrix

from models.Dynhat import Dynhat
from script.utils.dynamic_node2vec import load_manifest_and_snapshots, build_dynamic_node2vec
from script.utils.dataUtils import load_citation_data  # returns adj, features_sp, labels, idx_train, idx_val, idx_test
from script.utils.graphUtils import *

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# ======================
#        ARGS
# ======================
def parse_args():
    p = argparse.ArgumentParser(description="Dynhat + dynamic Node2Vec + temporal attention + IF/LOF + noise validation.")

    # --- Paths ---
    p.add_argument("--data-root", type=str, default="script/data/custom_out")

    # --- Node2Vec ---
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--walk-length", type=int, default=40)
    p.add_argument("--num-walks", type=int, default=300)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--t-max", type=int, default=None)

    # --- Dynhat-required args ---
    p.add_argument("--manifold", type=str, default="Hyperboloid")
    p.add_argument("--fix_curvature", action="store_true", default=False)
    p.add_argument("--curvature", type=float, default=1.0)
    p.add_argument("--c0", type=float, default=1.0)

    p.add_argument("--nhid", type=int, default=64)
    p.add_argument("--nout", type=int, default=64)
    p.add_argument("--heads", type=int, default=1)  # structural heads
    p.add_argument("--temporal_attention_layer_heads", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--aggregation", type=str, default="att", choices=["att"])

    # Temporal layer selector
    p.add_argument("--seq-model", dest="seq_model", type=str, default="Attention")

    # --- Training ---
    p.add_argument("--max-epoch", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--norm-scale", type=float, default=0.1)

    # --- Device ---
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

        # --- Noise validation (Stage 4) ---
    p.add_argument("--noise_val_iters", type=int, default=5)
    p.add_argument("--noise_val_k_percent", type=float, default=20.0)
    p.add_argument("--noise_val_top_frac", type=float, default=0.10)

    # Average degrees for fake nodes in noise validation
    p.add_argument("--noise_val_avg_degree_iso", type=int, default=5)
    p.add_argument("--noise_val_avg_degree_spam", type=int, default=400)


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
def run_if_lof_over_time(
    att,
    contamination: float = 0.05,
    lof_k: int = 30,
    topk: int = 20,
    jitter_eps: float = 1e-4,
    print_dup_stats: bool = False,
):
    """
    att: torch.Tensor of shape [N, T, C] or [N, C]

    Returns a dict containing:
      - mu_if, std_if, mu_lof, std_lof: per-node statistics (after canonicalization)
      - top indices: top_mu_if_idx, top_std_if_idx, top_mu_lof_idx, top_std_lof_idx
      - AS_if, AS_lof: full time-series matrices [N, T] (after canonicalization)
    """

    # Helper: print duplicate stats for debugging
    def _duplication_stats(X: np.ndarray, t: int):
        """Print basic duplication stats for a given time step."""
        unique_rows, counts = np.unique(X, axis=0, return_counts=True)
        num_samples = X.shape[0]
        num_unique = unique_rows.shape[0]
        max_dup = counts.max()
        num_duplicated = np.sum(counts > 1)
        print(
            f"[LOF debug] t={t}: samples={num_samples}, unique={num_unique}, "
            f"max_duplicates_for_single_vector={int(max_dup)}, "
            f"num_vectors_with_duplicates={int(num_duplicated)}"
        )

    # Ensure [N, T, C]
    if att.dim() == 2:
        att = att.unsqueeze(1)
    A = att.detach().cpu().numpy()  # [N, T, C]
    N, T, C = A.shape

    AS_if = np.full((N, T), np.nan, dtype=np.float32)
    AS_lof = np.full((N, T), np.nan, dtype=np.float32)

    lof_k = max(2, min(lof_k, N - 1))
    contamination = float(np.clip(contamination, 1.0 / max(N, 1), 0.2))

    rng = np.random.RandomState(42)  # fixed seed for reproducibility

    # Per-timeframe anomaly scoring
    for t in range(T):
        X_t = A[:, t, :]
        X_t = np.nan_to_num(X_t, nan=0.0, posinf=1e6, neginf=-1e6)

        # Drop non-informative columns (zero std)
        col_std = X_t.std(axis=0)
        informative = col_std > 0
        if informative.sum() == 0:
            continue
        X_t = X_t[:, informative]

        # Scale to [0, 1]
        X_t = MinMaxScaler().fit_transform(X_t)

        # Optional: show duplication stats before jitter
        if print_dup_stats and t == 0:
            _duplication_stats(X_t, t)

        # Add very small jitter to break exact duplicates (for LOF stability)
        if jitter_eps is not None and jitter_eps > 0.0:
            # jitter is applied after scaling, so values stay very close
            X_t = X_t + rng.normal(loc=0.0, scale=jitter_eps, size=X_t.shape)

        # Isolation Forest (negating decision_function so that larger=more anomalous)
        try:
            if_clf = IsolationForest(
                n_estimators=100,
                contamination=contamination,
                random_state=42,
                n_jobs=-1,
            ).fit(X_t)
            AS_if[:, t] = -if_clf.decision_function(X_t)
        except Exception as e:
            print(f"[IF warning] t={t}: {e}")

        # Local Outlier Factor (convert to larger=more anomalous)
        try:
            lof = LocalOutlierFactor(
                n_neighbors=lof_k,
                contamination=contamination,
                novelty=False,
                n_jobs=-1,
            )
            _ = lof.fit_predict(X_t)
            AS_lof[:, t] = -(lof.negative_outlier_factor_)
        except Exception as e:
            print(f"[LOF warning] t={t}: {e}")

    # Canonicalize both methods to non-negative scale (global min -> 0)
    AS_if = _normalize_minmax(AS_if)
    AS_lof = _normalize_minmax(AS_lof)

    _debug_stats("AS_if (canon)", AS_if)
    _debug_stats("AS_lof (canon)", AS_lof)

    # Per-node summaries on the canonized matrices
    mu_if = np.nanmean(AS_if, axis=1)
    std_if = np.nanstd(AS_if, axis=1)
    mu_lof = np.nanmean(AS_lof, axis=1)
    std_lof = np.nanstd(AS_lof, axis=1)

    def topk_idx(v, k):
        v = np.nan_to_num(v, nan=-1e30)
        k = min(k, len(v))
        return np.argsort(-v)[:k]

    return {
        "mu_if": mu_if,
        "std_if": std_if,
        "mu_lof": mu_lof,
        "std_lof": std_lof,
        "top_mu_if_idx": topk_idx(mu_if, topk),
        "top_std_if_idx": topk_idx(std_if, topk),
        "top_mu_lof_idx": topk_idx(mu_lof, topk),
        "top_std_lof_idx": topk_idx(std_lof, topk),
        "AS_if": AS_if,
        "AS_lof": AS_lof,
    }


# ===========================
#   NOISE INJECTION VALIDATION (full pipeline, but simple)
# ===========================
def _create_fake_embeddings_from_real(
    embedding_matrix: torch.Tensor,
    num_fake: int,
    noise_scale_iso: float = 2.0,
    noise_scale_spam: float = 5.0,
) -> torch.Tensor:
    """
    Create a mixed set of fake dynamic embeddings:
      - Half "isolated" style (weak, short temporal activity, low/zero degree)
      - Half "spammy" style (very strong magnitude, over-connected)
    embedding_matrix: [N_real, T, F]
    Returns: [num_fake, T, F]
    """
    if embedding_matrix.dim() != 3:
        raise ValueError("embedding_matrix must be [N, T, F]")

    N_real, T, F = embedding_matrix.shape
    device = embedding_matrix.device

    # Global feature std for scaling noise
    flat = embedding_matrix.reshape(-1, F)
    feat_std = flat.std(dim=0, keepdim=True)  # [1, F]
    feat_std = torch.where(feat_std == 0, torch.full_like(feat_std, 1e-6), feat_std)

    num_spam = num_fake // 2
    num_iso = num_fake - num_spam

    fake_list = []

    # --- Isolated-style fake nodes: short bursts in time, moderate noise + strong spike ---
    for _ in range(num_iso):
        base_idx = torch.randint(0, N_real, (1,), device=device).item()
        base_seq = embedding_matrix[base_idx].clone()  # [T, F]

        noise = torch.randn_like(base_seq) * (noise_scale_iso * feat_std)
        fake_seq = base_seq + noise  # base pattern + moderate noise

        # Temporal mask: only a small number of time steps are "active"
        mask = torch.zeros(T, 1, device=device)
        num_active = torch.randint(low=1, high=min(3, T + 1), size=(1,), device=device).item()
        active_indices = torch.randint(0, T, (num_active,), device=device)
        mask[active_indices] = 1.0

        fake_seq = fake_seq * mask  # mostly "off" in time

        # Add a strong spike at one random time step (temporal anomaly)
        random_t = torch.randint(0, T, (1,), device=device).item()
        jump = torch.randn(1, F, device=device) * (noise_scale_iso * 5.0 * feat_std)
        fake_seq[random_t] += jump.squeeze(0)

        fake_list.append(fake_seq)

    # --- Spammy-style fake nodes: strong magnitude, active at many time steps ---
    for _ in range(num_spam):
        base_idx = torch.randint(0, N_real, (1,), device=device).item()
        base_seq = embedding_matrix[base_idx].clone()  # [T, F]

        noise = torch.randn_like(base_seq) * (noise_scale_spam * feat_std)
        fake_seq = base_seq + noise

        # Amplify overall magnitude so it is clearly out-of-distribution
        fake_seq = fake_seq * 10.0
        fake_list.append(fake_seq)

    return torch.stack(fake_list, dim=0)  # [num_fake, T, F]




def _extend_edge_index_with_fake_nodes(
    edge_index: torch.Tensor,
    num_real: int,
    num_fake: int,
    avg_degree_iso: int = 1,
    avg_degree_spam: int = 200,
) -> torch.Tensor:
    """
    Extend base edge_index with two types of fake nodes:
      - First half: "isolated" style (very low or zero degree)
      - Second half: "spammy" style (very high degree, many edges to random real nodes)
    edge_index: [2, E]
    Nodes 0..num_real-1 are real; new fake nodes get indices num_real..num_real+num_fake-1.
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must be of shape [2, E]")

    device = edge_index.device
    row_parts = [edge_index[0]]
    col_parts = [edge_index[1]]

    num_spam = num_fake // 2
    num_iso = num_fake - num_spam

    # --- Isolated-style fake nodes: very low / zero degree ---
    for i in range(num_iso):
        fake_idx = num_real + i

        # Increase chance to be fully isolated: about 60% no edges at all
        if torch.rand(1, device=device).item() < 0.6:
            continue

        deg = max(1, int(np.random.poisson(lam=max(avg_degree_iso, 1))))
        deg = min(deg, num_real)
        neighbors = torch.randint(0, num_real, (deg,), device=device)
        fake_vec = torch.full((deg,), fake_idx, device=device, dtype=torch.long)

        # Undirected edges: fake <-> real
        row_parts.append(torch.cat([fake_vec, neighbors]))
        col_parts.append(torch.cat([neighbors, fake_vec]))

    # --- Spammy-style fake nodes: very high degree ---
    for j in range(num_spam):
        fake_idx = num_real + num_iso + j
        deg = max(10, int(np.random.poisson(lam=max(avg_degree_spam, 10))))
        deg = min(deg, num_real)

        # Use unique neighbors where possible
        neighbors = torch.randperm(num_real, device=device)[:deg]
        fake_vec = torch.full((deg,), fake_idx, device=device, dtype=torch.long)

        row_parts.append(torch.cat([fake_vec, neighbors]))
        col_parts.append(torch.cat([neighbors, fake_vec]))

    new_row = torch.cat(row_parts, dim=0)
    new_col = torch.cat(col_parts, dim=0)
    return torch.stack([new_row, new_col], dim=0)



def noise_injection_validation_full_pipeline(
    embedding_matrix: torch.Tensor,
    edge_index: torch.Tensor,
    model: torch.nn.Module,
    num_iterations: int = 5,
    k_percent: float = 5.0,
    top_frac: float = 0.05,
    contamination: float = 0.05,
    lof_k: int = 30,
    norm_scale: float = 0.1,
    avg_degree_iso: int = 1,
    avg_degree_spam: int = 200,
):
    """
    Implementation of Algorithm 4 (noise injection via fake nodes),
    using a mixed strategy:
      - Half of the fake nodes are "isolated" (low/zero degree, short temporal activity + spike)
      - Half are "spammy" (high degree, strong magnitude).
    The full pipeline is applied: embeddings -> Dynhat -> temporal attention -> IF/LOF.
    """
    if embedding_matrix.dim() != 3:
        raise ValueError("embedding_matrix must be [N, T, F]")

    N_real, T, F = embedding_matrix.shape
    if N_real == 0:
        print("[Noise Validation] No real nodes, skipping.")
        return None

    device = next(model.parameters()).device
    embedding_matrix = embedding_matrix.to(device)
    edge_index = edge_index.to(device)

    num_fake_per_iter = max(1, int(round(k_percent / 100.0 * N_real)))
    top_frac = float(np.clip(top_frac, 1e-4, 0.5))

    tpr_if_list, fpr_if_list = [], []
    tpr_lof_list, fpr_lof_list = [], []

    print(
        f"\n===== Noise Injection Validation (full pipeline, mixed fake nodes) =====\n"
        f"Real nodes: {N_real}, fake per iteration: {num_fake_per_iter} ({k_percent}%), "
        f"iterations: {num_iterations}, top_frac={top_frac}\n"
    )

    for it in range(num_iterations):
        print(f"[Noise Validation] Iteration {it + 1}/{num_iterations}...")

        # 1) Generate mixed fake nodes in embedding space
        fake_emb = _create_fake_embeddings_from_real(
            embedding_matrix=embedding_matrix,
            num_fake=num_fake_per_iter,
            noise_scale_iso=2.0,
            noise_scale_spam=5.0,
        )  # [num_fake, T, F]

        # 2) Augment embeddings and graph structure
        emb_aug = torch.cat([embedding_matrix, fake_emb.to(device)], dim=0)  # [N_all, T, F]
        edge_aug = _extend_edge_index_with_fake_nodes(
            edge_index=edge_index,
            num_real=N_real,
            num_fake=num_fake_per_iter,
            avg_degree_iso=avg_degree_iso,
            avg_degree_spam=avg_degree_spam,
        )  # [2, E_aug]

        N_all = emb_aug.shape[0]

        # 3) Full Dynhat forward (structural + temporal encoding)
        model.eval()
        with torch.no_grad():
            outs_iter = []
            for t_idx in range(T):
                x_t = emb_aug[:, t_idx, :]  # [N_all, F]
                x_t = torch.nn.functional.normalize(x_t, p=2, dim=1) * float(norm_scale)
                h_t = model(edge_aug, x=x_t)  # [N_all, C]
                outs_iter.append(h_t)

            X_iter = torch.stack(outs_iter, dim=1)  # [N_all, T, C]
            att_iter = model.seq_model(X_iter)      # [N_all, T, C] or [N_all, C]
            if att_iter.ndim == 2:
                att_iter = att_iter.unsqueeze(1)

        # 4) Anomaly detection on augmented set
        res_aug = run_if_lof_over_time(
            att_iter,
            contamination=contamination,
            lof_k=lof_k,
            topk=0,
        )

        AS_if = res_aug.get("AS_if", None)
        AS_lof = res_aug.get("AS_lof", None)

        # --- IF ---
        if AS_if is not None:
            max_if = np.nanmax(AS_if, axis=1)  # [N_all]
            thr_if = float(np.quantile(max_if, 1.0 - top_frac))
            flags_if = max_if >= thr_if

            flags_real_if = flags_if[:N_real]
            flags_fake_if = flags_if[N_real:]

            num_real_flagged_if = int(flags_real_if.sum())
            num_fake_flagged_if = int(flags_fake_if.sum())

            tpr_if = num_fake_flagged_if / float(len(flags_fake_if)) if len(flags_fake_if) > 0 else 0.0
            fpr_if = num_real_flagged_if / float(N_real) if N_real > 0 else 0.0

            tpr_if_list.append(tpr_if)
            fpr_if_list.append(fpr_if)

            print(
                f"  [IF] thr={thr_if:.4f} | TPR={tpr_if:.3f}, FPR={fpr_if:.3f} "
                f"(fake flagged: {num_fake_flagged_if}/{len(flags_fake_if)}, "
                f"real flagged: {num_real_flagged_if}/{N_real})"
            )

        # --- LOF ---
        if AS_lof is not None:
            max_lof = np.nanmax(AS_lof, axis=1)  # [N_all]
            thr_lof = float(np.quantile(max_lof, 1.0 - top_frac))
            flags_lof = max_lof >= thr_lof

            flags_real_lof = flags_lof[:N_real]
            flags_fake_lof = flags_lof[N_real:]

            num_real_flagged_lof = int(flags_real_lof.sum())
            num_fake_flagged_lof = int(flags_fake_lof.sum())

            tpr_lof = num_fake_flagged_lof / float(len(flags_fake_lof)) if len(flags_fake_lof) > 0 else 0.0
            fpr_lof = num_real_flagged_lof / float(N_real) if N_real > 0 else 0.0

            tpr_lof_list.append(tpr_lof)
            fpr_lof_list.append(fpr_lof)

            print(
                f"  [LOF] thr={thr_lof:.4f} | TPR={tpr_lof:.3f}, FPR={fpr_lof:.3f} "
                f"(fake flagged: {num_fake_flagged_lof}/{len(flags_fake_lof)}, "
                f"real flagged: {num_real_flagged_lof}/{N_real})"
            )

    def _summary(vals):
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            return "n/a"
        return f"{vals.mean():.3f} ± {vals.std():.3f}"

    print("\n===== Noise Injection Validation Summary (full pipeline, mixed fake nodes) =====")
    print(f"IF : TPR={_summary(tpr_if_list)}, FPR={_summary(fpr_if_list)}")
    print(f"LOF: TPR={_summary(tpr_lof_list)}, FPR={_summary(fpr_lof_list)}")
    print("==========================================================================\n")

    return {
        "tpr_if": tpr_if_list,
        "fpr_if": fpr_if_list,
        "tpr_lof": tpr_lof_list,
        "fpr_lof": fpr_lof_list,
        "num_fake_per_iter": num_fake_per_iter,
        "k_percent": k_percent,
        "top_frac": top_frac,
    }


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

    args.nfeat = int(embedding_matrix.shape[-1])  # F
    args.num_nodes = int(embedding_matrix.shape[0])  # N

    # 2) Load graph/labels + splits
    adj, features_sp, labels_np, idx_train, idx_val, idx_test = load_citation_data(
        dataset_str="dblpv13",
        use_feats=True,
        data_path=args.data_root,
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
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(
        f"🔧 Dynhat ready (nhid={args.nhid}, temporal_heads={args.temporal_attention_layer_heads}, "
        f"classes={args.num_classes})"
    )

    # 4) Training (no validation)
    model.train()
    cls_head.train()
    for epoch in range(args.max_epoch):
        optimizer.zero_grad()

        temporal_outputs = []
        for t in range(T_bins):
            x_t = embedding_matrix[:, t, :].to(device)  # [N, F]
            x_t = torch.nn.functional.normalize(x_t, p=2, dim=1) * float(args.norm_scale)
            h_t = model(edge_index, x=x_t)  # [N, C=nhid+1]
            temporal_outputs.append(h_t)

        X = torch.stack(temporal_outputs, dim=1)  # [N, T, C]
        att = model.seq_model(X)  # [N, T, C] or [N, C]
        if att.ndim == 2:
            att = att.unsqueeze(1)  # [N, 1, C]

        feat_last = att[:, -1, :]  # [N, C]
        logits = cls_head(feat_last)  # [N, num_classes]
        loss = F.cross_entropy(logits[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        print(f"[Epoch {epoch}] Train Loss: {loss.item():.4f}")

    # 5) Final representations (optional)
    model.eval()
    cls_head.eval()
    with torch.no_grad():
        outs = []
        for t in range(T_bins):
            x_t = embedding_matrix[:, t, :].to(device)
            x_t = torch.nn.functional.normalize(x_t, p=2, dim=1) * float(args.norm_scale)
            h_t = model(edge_index, x=x_t)
            outs.append(h_t)
        X_eval = torch.stack(outs, dim=1)  # [N, T, C]
        att_out = model.seq_model(X_eval)  # [N, T, C] or [N, C]
        if att_out.ndim == 2:
            att_out = att_out.unsqueeze(1)

    print(
        "✅ Done main pipeline. Shapes:",
        f"N={att_out.shape[0]}, T={att_out.shape[1]}, C={att_out.shape[2]}",
    )

    # 6) Anomaly over time (IF/LOF) + canonicalization
    res = run_if_lof_over_time(
    att_out,
    contamination=0.05,
    lof_k=100,
    topk=20,
    jitter_eps=1e-4,
    print_dup_stats=True,  # פעם-פעמיים להרצה דיאגנוסטית
)

    print("Top-20 IF(mean):", res["top_mu_if_idx"])
    print("Top-20 IF(std):", res["top_std_if_idx"])
    print("Top-20 LOF(mean):", res["top_mu_lof_idx"])
    print("Top-20 LOF(std):", res["top_std_lof_idx"])

    print("Min std_if:", np.min(res["std_if"]))
    print("Min std_lof:", np.min(res["std_lof"]))

    common_top = set(res["top_mu_if_idx"]) & set(res["top_mu_lof_idx"])
    print(f"✅ Common top anomalies between IF and LOF: {len(common_top)} nodes")
    print(f"Common indices: {sorted(list(common_top))}")

    common_top_list = [int(i) for i in common_top]
    if common_top_list:
        plot_if_lof_timeseries_for_nodes(
            AS_if=res["AS_if"],
            AS_lof=res["AS_lof"],
            node_indices=common_top_list,
            save_dir="plots/common_if_lof_timeseries",
        )

    # 7) Plots (all consistent, non-negative scale)
    plot_mean_std(mu=res["mu_if"], std=res["std_if"], top_k=20, save_dir="plots")

    if "AS_if" in res and res["AS_if"] is not None:
        plot_top_node_timeseries_by_method(
            AS=res["AS_if"],
            mu=res["mu_if"],
            method_name="IF",
            save_dir="plots",
        )

    if "AS_lof" in res and res["AS_lof"] is not None:
        plot_top_node_timeseries_by_method(
            AS=res["AS_lof"],
            mu=res["mu_lof"],
            method_name="LOF",
            save_dir="plots",
        )

    # Statistical histograms of distributions
    plot_hist_distribution(
        res["mu_if"],
        "Distribution of Mean (μ) anomaly scores - IF",
        "Mean (μ) value",
        "plots/hist_mean_if.png",
    )
    plot_hist_distribution(
        res["std_if"],
        "Distribution of Std (σ) anomaly scores - IF",
        "Std (σ) value",
        "plots/hist_std_if.png",
    )

    plot_hist_distribution(
        res["mu_lof"],
        "Distribution of Mean (μ) anomaly scores - LOF",
        "Mean (μ) value",
        "plots/hist_mean_lof.png",
    )
    plot_hist_distribution(
        res["std_lof"],
        "Distribution of Std (σ) anomaly scores - LOF",
        "Std (σ) value",
        "plots/hist_std_lof.png",
    )

    # 8) Noise-injection validation (Stage 4: Algorithm 4)
    val_stats = noise_injection_validation_full_pipeline(
        embedding_matrix=embedding_matrix,   # Stage 1 output (real nodes)
        edge_index=edge_index,              # base graph
        model=model,                        # trained Dynhat
        num_iterations=args.noise_val_iters,
        k_percent=args.noise_val_k_percent,
        top_frac=args.noise_val_top_frac,
        contamination=0.05,
        lof_k=100,
        norm_scale=args.norm_scale,
        avg_degree_iso=args.noise_val_avg_degree_iso,
        avg_degree_spam=args.noise_val_avg_degree_spam,
    )

    if val_stats is not None:
        plot_noise_validation_tpr(
            tpr_if=val_stats.get("tpr_if", []),
            tpr_lof=val_stats.get("tpr_lof", []),
            save_path="plots/noise_validation_tpr.png",
        )
        plot_noise_validation_fpr(
            fpr_if=val_stats.get("fpr_if", []),
            fpr_lof=val_stats.get("fpr_lof", []),
            save_path="plots/noise_validation_fpr.png",
        )
if __name__ == "__main__":
    main()