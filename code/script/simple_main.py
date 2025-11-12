# -*- coding: utf-8 -*-

"""
MAIN פשוט:
1) טעינת סנאפשוטים
2) הפקת אמבדינגים דינמיים (Node2Vec) [N, T, F]
3) בניית Dynhat והרצה עם שכבת קשב טמפורלי
4) אימון קצר עם Cross-Entropy על idx_train בלבד (ללא ולידציה)
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import from_scipy_sparse_matrix
import matplotlib.pyplot as plt

# מודלים/כלים פנימיים
from models.Dynhat import Dynhat
from script.utils.dynamic_node2vec import load_manifest_and_snapshots, build_dynamic_node2vec
from script.utils.dataUtils import load_citation_data  # מחזיר adj, features_sp, labels, idx_train, idx_val, idx_test
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# ======================
#   ארגומנטים מינימליים
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

    # --- Dynhat-required args (החשובים שחסרו) ---
    p.add_argument("--manifold", type=str, default="Hyperboloid")        # ← נדרש ע"י Dynhat
    p.add_argument("--fix_curvature", action="store_true", default=False)
    p.add_argument("--curvature", type=float, default=1.0)
    p.add_argument("--c0", type=float, default=1.0)

    p.add_argument("--nhid", type=int, default=32)
    p.add_argument("--nout", type=int, default=32)
    p.add_argument("--heads", type=int, default=1)                        # structural heads
    p.add_argument("--temporal_attention_layer_heads", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--aggregation", type=str, default="att", choices=["att"])

    # שכבת הזמן של Dynhat מגדירה TemporalAttentionLayer ומצפה לשדה seq_model לשם בלבד
    p.add_argument("--seq-model", dest="seq_model", type=str, default="Attention")

    # --- Training ---
    p.add_argument("--max-epoch", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--norm-scale", type=float, default=0.1)

    # --- Device ---
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def run_if_lof_over_time(att, contamination=0.05, lof_k=30, topk=20):
    """
    att: torch.Tensor of shape [N, T, C] or [N, C]
    מחזיר dict עם ציוני IF/LOF (ממוצע וסטיית תקן לכל צומת) + אינדקסים של Top-K.
    """
    # הפוך ל-[N, T, C]
    if att.dim() == 2:
        att = att.unsqueeze(1)
    A = att.detach().cpu().numpy()  # [N, T, C]
    N, T, C = A.shape

    # מכולות תוצאות
    AS_if  = np.full((N, T), np.nan, dtype=np.float32)
    AS_lof = np.full((N, T), np.nan, dtype=np.float32)

    # פרמטרי LOF תקפים
    lof_k = max(2, min(lof_k, N-1))
    # הגנה קטנה על contamination
    contamination = float(np.clip(contamination, 1.0 / max(N,1), 0.2))

    # לכל זמן — נרמל עמודות, ואז IF/LOF
    for t in range(T):
        X_t = A[:, t, :]                   # [N, C]
        # נירמול פשוט ו־NaN guards
        X_t = np.nan_to_num(X_t, nan=0.0, posinf=1e6, neginf=-1e6)
        col_std = X_t.std(axis=0)
        informative = col_std > 0
        if informative.sum() == 0:
            continue
        X_t = X_t[:, informative]
        X_t = MinMaxScaler().fit_transform(X_t)

        # IF
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

        # LOF
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

    # סיכום לכל צומת
    mu_if   = np.nanmean(AS_if,  axis=1)
    std_if  = np.nanstd(AS_if,   axis=1)
    mu_lof  = np.nanmean(AS_lof, axis=1)
    std_lof = np.nanstd(AS_lof,  axis=1)

    # Top-K אינדקסים (גדול=יותר חריג)
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
    }



import os
import matplotlib.pyplot as plt
import numpy as np

def plot_anomaly_scores(mu_if, std_if, mu_lof, std_lof, top_k=None, save_dir="plots"):
    """
    מצייר ושומר גרפים של ציוני אנומליה (mean ו־std) עבור כל צומת.
    כל גרף נשמר בתיקייה 'plots' כברירת מחדל.
    """
    os.makedirs(save_dir, exist_ok=True)  # יצירת התיקייה אם לא קיימת

    N = len(mu_if)
    x = np.arange(N)

    # --- גרף 1: mean per node (IF vs LOF) ---
    plt.figure(figsize=(12, 6))
    plt.scatter(x, mu_if, s=8, color='blue', alpha=0.6, label='IF mean (μ)')
    plt.scatter(x, mu_lof, s=8, color='orange', alpha=0.6, label='LOF mean (μ)')
    plt.xlabel("Node ID (i)")
    plt.ylabel("Mean anomaly score")
    plt.title("Mean (μ) anomaly scores per node (IF vs LOF)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "anomaly_mean_scores.png"))
    plt.close()
    print(f"💾 Saved: {os.path.join(save_dir, 'anomaly_mean_scores.png')}")

    # --- גרף 2: std per node (IF vs LOF) ---
    plt.figure(figsize=(12, 6))
    plt.scatter(x, std_if, s=8, color='green', alpha=0.6, label='IF std (σ)')
    plt.scatter(x, std_lof, s=8, color='red', alpha=0.6, label='LOF std (σ)')
    plt.xlabel("Node ID (i)")
    plt.ylabel("Standard deviation of anomaly score")
    plt.title("Standard deviation (σ) of anomaly scores per node (IF vs LOF)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "anomaly_std_scores.png"))
    plt.close()
    print(f"💾 Saved: {os.path.join(save_dir, 'anomaly_std_scores.png')}")

    # --- גרף 3 (אופציונלי): Top-K nodes לפי IF mean/std ---
    if top_k is not None:
        top_nodes = np.argsort(-mu_if)[:top_k]
        plt.figure(figsize=(10, 5))
        plt.bar(top_nodes, mu_if[top_nodes], color='steelblue', label='Top IF mean')
        plt.bar(top_nodes, std_if[top_nodes], color='lightcoral', alpha=0.6, label='Top IF std')
        plt.xlabel("Node ID (Top-K by IF mean)")
        plt.ylabel("Score value")
        plt.title(f"Top-{top_k} nodes by IF mean & std")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"top_{top_k}_if_scores.png"))
        plt.close()
        print(f"💾 Saved: {os.path.join(save_dir, f'top_{top_k}_if_scores.png')}")

def plot_mean_std(mu: np.ndarray, std: np.ndarray, top_k: int = 10,  save_dir="plots"):
    N = len(mu)
    x = np.arange(N)

    plt.figure(figsize=(11, 6))

    # ציור mean ו-std כעמודות
    plt.bar(x - 0.2, mu, width=0.4, color='skyblue', alpha=0.7, label='Mean (μ)')
    plt.bar(x + 0.2, std, width=0.4, color='orange', alpha=0.7, label='Std Dev (σ)')

    # הדגשת חריגים לפי mean
    top_mean_idx = np.argsort(-mu)[:top_k]
    plt.scatter(top_mean_idx, mu[top_mean_idx], color='red', s=80, label=f'Top {top_k} Mean')

    # הדגשת חריגים לפי std
    top_std_idx = np.argsort(-std)[:top_k]
    plt.scatter(top_std_idx, std[top_std_idx], color='purple', s=80, label=f'Top {top_k} Std')

    plt.title('Isolation Forest — Mean & Std per Node', fontsize=14)
    plt.xlabel('Node index (i)', fontsize=12)
    plt.ylabel('Score value', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_dir, "plot_mean_std.png"))
    plt.close()
    print(f"💾 Saved: {os.path.join(save_dir, 'plot_mean_std.png')}")

# ===============
#     MAIN
# ===============
def main():
    args = parse_args()

    args.fix_curvature = True          # ← חובה כדי ש-Dynhat ייצור self.c
    if not hasattr(args, "curvature"):
        args.curvature = 1.0           # ערך סטנדרטי ללורנץ
    if not hasattr(args, "device"):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    # ---------------------------------------------
    # 1) טעינת סנאפשוטים ובניית Node2Vec דינמי
    # ---------------------------------------------
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

    # קיבוע מאפייני קלט למודל
    args.nfeat = int(embedding_matrix.shape[-1])     # F
    args.num_nodes = int(embedding_matrix.shape[0])  # N

    # -------------------------------------------------
    # 2) טעינת גרף/לייבלים + פיצול (train/val/test)
    # -------------------------------------------------
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

    # ----------------------------
    # 3) מודל Dynhat + ראש סיווג
    # ----------------------------
    model = Dynhat(args, time_length=T_bins).to(device)
    cls_head = torch.nn.Linear(args.nhid + 1, args.num_classes).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(cls_head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    print(f"🔧 Dynhat ready (nhid={args.nhid}, temporal_heads={args.temporal_attention_layer_heads}, classes={args.num_classes})")

    # ----------------------------
    # 4) אימון (ללא ולידציה)
    # ----------------------------
    model.train(); cls_head.train()
    for epoch in range(args.max_epoch):
        optimizer.zero_grad()

        # מריצים את Dynhat על כל T ואוספים ייצוגים
        temporal_outputs = []
        for t in range(T_bins):
            x_t = embedding_matrix[:, t, :].to(device)                # [N, F]
            # נרמול עדין (ליציבות קלה; עדיין פשוט)
            x_t = torch.nn.functional.normalize(x_t, p=2, dim=1) * float(args.norm_scale)
            h_t = model(edge_index, x=x_t)                             # [N, C=nhid+1]
            temporal_outputs.append(h_t)

        X = torch.stack(temporal_outputs, dim=1)                       # [N, T, C]
        att = model.seq_model(X)                                       # [N, T, C] או [N, C]
        if att.ndim == 2:
            att = att.unsqueeze(1)                                     # [N, 1, C]

        feat_last = att[:, -1, :]                                      # [N, C]
        logits = cls_head(feat_last)                                   # [N, num_classes]
        loss = F.cross_entropy(logits[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        print(f"[Epoch {epoch}] Train Loss: {loss.item():.4f}")

    # ----------------------------------------
    # 5) יציאה: ייצוגים סופיים (לא חובה לשמור)
    # ----------------------------------------
    model.eval(); cls_head.eval()
    with torch.no_grad():
        outs = []
        for t in range(T_bins):
            x_t = embedding_matrix[:, t, :].to(device)
            x_t = torch.nn.functional.normalize(x_t, p=2, dim=1) * float(args.norm_scale)
            h_t = model(edge_index, x=x_t)
            outs.append(h_t)
        X_eval = torch.stack(outs, dim=1)                # [N, T, C]
        att_out = model.seq_model(X_eval)                # [N, T, C] או [N, C]
        if att_out.ndim == 2:
            att_out = att_out.unsqueeze(1)

    print("✅ Done. Shapes:",
          f"N={att_out.shape[0]}, T={att_out.shape[1]}, C={att_out.shape[2]}")
    # אפשר להוסיף כאן שמירה לקובץ אם רוצים:
    # torch.save({"att_output": att_out.cpu()}, "att_output.pt")


    res = run_if_lof_over_time(att, contamination=0.05, lof_k=30, topk=20)
    print("Top-20 IF(mean):",   res["top_mu_if_idx"])
    print("Top-20 IF(std):",    res["top_std_if_idx"])
    print("Top-20 LOF(mean):",  res["top_mu_lof_idx"])
    print("Top-20 LOF(std):",   res["top_std_lof_idx"])

    plot_anomaly_scores(
        mu_if=res["mu_if"],
        std_if=res["std_if"],
        mu_lof=res["mu_lof"],
        std_lof=res["std_lof"],
        top_k=20,
        save_dir="plots"   # ניתן לשנות לנתיב אחר, למשל "results/graphs"
    )
    plot_mean_std( mu_if=res["mu_if"], std_if=res["std_if"], top_k=20, save_dir="plots" )


if __name__ == "__main__":
    main()
