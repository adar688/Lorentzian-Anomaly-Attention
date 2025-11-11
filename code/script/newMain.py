# -*- coding: utf-8 -*-

"""
מריצים את הקובץ הזה לאחר שטענו את הדאטה סט ועשינו לו עיבוד.
כאן כבר מתחילים לעבוד איתו: טוענים את הקבצים שעיבדנו, מפיקים אמבדינגים דינמיים (Node2Vec),
מריצים Dynhat, מחשבים IF/LOF טמפורלי, ומדפיסים/שומרים מדדים להבנת התוצאות.
"""

import os
import sys
import argparse
import copy
import re
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.utils import from_scipy_sparse_matrix, from_networkx
from models.Dynhat import Dynhat

from script.utils.dynamic_node2vec import (
    load_manifest_and_snapshots,   # מצפה ל-manifest.json + snapshots.npz
    build_dynamic_node2vec         # בונה Tensor [N, T, F]
)
from script.utils.dataUtils import load_citation_data  # המימוש המינימלי שכתבנו

# === אנליזה ואבחון ===
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import spearmanr
import pandas as pd


# ==============================
#    ארגומנטים
# ==============================
def parse_args():
    p = argparse.ArgumentParser(description="Unified main: Dynamic Node2Vec + Dynhat + IF/LOF (with diagnostics)")

    # נתיבי קלט
    p.add_argument("--data-root", type=str, default="script/data/custom_out",
                   help="תיקייה עם קבצי הפלט של prepareData (manifest.json, snapshots.npz, graph.json, ...)")

    # פרמטרי Node2Vec
    p.add_argument("--emb-dim", type=int, default=128, help="F: ממד האמבדינג")
    p.add_argument("--walk-length", type=int, default=30, help="Node2Vec: אורך הליכה")
    p.add_argument("--num-walks", type=int, default=200, help="Node2Vec: מספר הליכות לצומת")
    p.add_argument("--workers", type=int, default=2, help="Node2Vec: מספר תהליכי רקע")
    p.add_argument("--window", type=int, default=10, help="Word2Vec window")
    p.add_argument("--t-max", type=int, default=None,
                   help="אופציונלי: שימוש רק ב-T הראשונים (לבדיקות/קיצור)")

    # נרמול קלט לאימון/אינפרנס (גורם כיווץ עדין אחרי L2 normalize)
    p.add_argument("--norm-scale", type=float, default=0.1, help="מכפיל אחרי normalize כדי לשמור יציבות נומרית")

    # מניפולד/גיאומטריה (דגלים רק למילוי args עבור Dynhat, לא משנים את הקוד שלו)
    p.add_argument("--manifold", type=str, default="Hyperboloid")
    p.add_argument("--fix_curvature", action="store_true")
    p.add_argument("--curvature", type=float, default=1.0)
    p.add_argument("--c0", type=float, default=1.0)

    # היפר־פרמטרים של Dynhat (נדרש רק כדי למלא args; Dynhat כבר מגדיר פנימית את agg_feat_size)
    p.add_argument("--nhid", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--attn-dropout", type=float, default=0.0)
    p.add_argument("--feat-dropout", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--nheads", type=int, default=1)
    p.add_argument("--temporal_attention_layer_heads", type=int, default=1)
    p.add_argument("--bias", type=int, choices=[0, 1], default=1)
    p.add_argument("--residual", action="store_true")
    p.add_argument("--batch-norm", action="store_true")
    p.add_argument("--aggregation", type=str, default="att", choices=["att", "mean", "sum", "max"])
    p.add_argument("--nfeat", type=int, default=32)
    p.add_argument("--nout", type=int, default=32)
    p.add_argument("--seq-model", dest="seq_model", type=str, default="gru",
                   choices=["gru", "lstm", "transformer", "none"])
    p.add_argument("--seq-hidden", dest="seq_hidden", type=int, default=128)
    p.add_argument("--seq-layers", dest="seq_layers", type=int, default=1)
    p.add_argument("--seq-dropout", dest="seq_dropout", type=float, default=0.0)

    # אימון
    p.add_argument("--max-epoch", type=int, default=32, help="מספר אפוקים לאימון Dynhat")
    p.add_argument("--lr", type=float, default=3e-3, help="למידה - Adam LR")
    p.add_argument("--weight-decay", type=float, default=5e-4, help="למידה - Adam weight decay")

    # מכשיר ושמירה
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                   help="cuda / cpu")
    p.add_argument("--save-bundle", type=str, default="",
                   help="נתיב לשמירת חבילה אחת בסוף (embedding_matrix + graph tensors). ריק = לא שומר.")

    # אנומליות
    p.add_argument("--contamination", type=float, default=0.05, help="אחוז אנומליות משוער ל-IF/LOF")
    p.add_argument("--lof-n-neighbors", type=int, default=30, help="k של LOF")
    p.add_argument("--topk", type=int, default=20, help="Top-K להצגה/ניתוח (לא חובה)")

    # רעש (Stage 4)
    p.add_argument("--noise-percent", type=float, default=0.1, help="k% צמתי רעש מכלל הצמתים")
    p.add_argument("--noise-connect-prob", type=float, default=0.5, help="הסתברות חיבור רעש↔מקוריים")
    p.add_argument("--noise-iters", type=int, default=30, help="מספר איטרציות ולידציה עם רעש")
    p.add_argument("--random-state", type=int, default=1024, help="זרע רנדומי לשחזוריות")

    return p.parse_args()


def _ensure_dynhat_defaults(args):
    defaults = {
        "nhid": 64,
        "dropout": 0.0,
        "attn_dropout": 0.0,
        "feat_dropout": 0.0,
        "alpha": 0.2,
        "nheads": 1,
        "temporal_attention_layer_heads": 1,
        "fix_curvature": False,
        "curvature": 1.0,
        "c0": 1.0,
        "bias": True,
        "residual": False,
        "batch_norm": False,
        "manifold": "Hyperboloid",
        "aggregation": "att",
        "seq_model": "gru",
        "seq_hidden": 128,
        "seq_layers": 1,
        "seq_dropout": 0.1,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    if not hasattr(args, "heads") and hasattr(args, "nheads"):
        args.heads = args.nheads
    if isinstance(getattr(args, "bias", True), int):
        args.bias = bool(args.bias)
    if not hasattr(args, "c"):
        args.c = float(getattr(args, "curvature", getattr(args, "c0", 1.0)))
    return args


# ==============================
#   כלי אבחון / תוצרים
# ==============================
def _summ(name, arr):
    arr = np.asarray(arr).ravel()
    if not np.isfinite(arr).any():
        print(f"\n{name}: (no finite data)")
        return
    q = np.percentile(arr[np.isfinite(arr)], [0, 1, 5, 25, 50, 75, 95, 99, 100])
    print(f"\n{name}:")
    print(f"  mean={np.nanmean(arr):.6f}  std={np.nanstd(arr):.6f}  min={q[0]:.6f}  p1={q[1]:.6f}  p5={q[2]:.6f}")
    print(f"  p25={q[3]:.6f}  p50={q[4]:.6f}  p75={q[5]:.6f}  p95={q[6]:.6f}  p99={q[7]:.6f}  max={q[8]:.6f}")

def _print_topk(title, scores, k=10):
    arr = np.asarray(scores).ravel()
    if not np.isfinite(arr).any():
        print(f"\n{title} (no finite data)")
        return np.array([], dtype=int)
    k = min(k, len(arr))
    order = np.argsort(-np.nan_to_num(arr, nan=-1e30))[:k]
    print(f"\n{title} (Top-{k}):")
    for rank, idx in enumerate(order, 1):
        s = float(arr[idx])
        print(f"  {rank:>2}. node={int(idx):>6}  score={s if np.isfinite(s) else float('nan')}")
    return order

def _save_hist(arr, fname, bins=40, title=None):
    a = np.asarray(arr).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        print(f"ℹ️  skipped histogram {fname}: no finite data.")
        return
    plt.figure(figsize=(8, 4.5))
    plt.hist(a, bins=bins)
    if title: plt.title(title)
    plt.xlabel("score"); plt.ylabel("count"); plt.tight_layout()
    plt.savefig(fname); plt.close()
    print(f"💾 saved: {fname}")

def dbg_stats(tag, tensor, per_time=False, max_show=5):
    x = tensor.detach().cpu().numpy()
    finite = np.isfinite(x)
    pct = 100.0 * finite.mean()
    shp = tuple(x.shape)
    if per_time and x.ndim == 3:
        T = x.shape[1]
        per = []
        for t in range(T):
            per.append(100.0 * np.isfinite(x[:, t, :]).mean())
        print(f"[DBG] {tag}: shape={shp}, finite%={pct:.2f}%, finite% per t: {per[:max_show]} ...")
    else:
        print(f"[DBG] {tag}: shape={shp}, finite%={pct:.2f}%")

def dbg_seq_temporal_layer_once(seq_layer, seq_input):
    print("\n===== TEMPORAL ATTENTION DEBUG =====")
    dbg_stats("seq_input(N,T,C)", seq_input, per_time=True)
    try:
        has_proj = False
        for nm, _ in getattr(seq_layer, 'named_parameters', lambda: [])():
            if any(k in nm.lower() for k in ['q', 'k', 'v']):
                has_proj = True; break
        if has_proj:
            print("[DBG] Q/K/V projectors detected on TemporalAttentionLayer (names vary) – not printing weights.")
        else:
            print("[DBG] Q/K/V projectors not found on TemporalAttentionLayer (looked for common names). Skipping internals and calling forward normally.")
    except Exception:
        print("[DBG] Could not iterate parameters safely; skipping.")
    with torch.no_grad():
        y = seq_layer(seq_input)
    dbg_stats("seq_output", y, per_time=True)
    print("===== END TEMPORAL ATTENTION DEBUG =====\n")


# ========================================================
#  Stage 4 unified: validate_with_noise_injection (IF בלבד)
# ========================================================
def validate_with_noise_injection(
    G_original,
    embedding_matrix,
    model,
    T,
    k_percent=0.05,
    connect_prob=0.5,
    n_iters=30,
    contamination=0.05,
    random_state=42,
):
    rng = np.random.default_rng(random_state)
    torch.manual_seed(random_state)

    device = embedding_matrix.device
    N, T_actual, F = embedding_matrix.shape
    assert T == T_actual, "T לא תואם את ממד הזמן של embedding_matrix"

    model.eval()

    tpr_list, fpr_list = [], []
    results_per_iter = []

    for it in range(n_iters):
        t_i = int(rng.integers(low=0, high=T))
        n_noise_nodes = max(1, int(round(k_percent * N)))

        G_noisy = copy.deepcopy(G_original)
        original_N = G_noisy.number_of_nodes()
        new_node_ids = list(range(original_N, original_N + n_noise_nodes))
        for node in new_node_ids:
            G_noisy.add_node(node)
            for target in range(original_N):
                if rng.random() < connect_prob:
                    G_noisy.add_edge(node, target)
                if rng.random() < connect_prob:
                    G_noisy.add_edge(target, node)

        mu = embedding_matrix.mean(dim=(0, 1)).to(device)
        noise_std = 0.1
        noise_tensor = mu + noise_std * torch.randn((n_noise_nodes, F), device=device)

        noisy_slices = []
        for t in range(T):
            base_t = embedding_matrix[:, t, :]
            if t == t_i:
                feats_t = torch.cat([base_t, noise_tensor], dim=0)
            else:
                feats_t = torch.cat([base_t, torch.zeros_like(noise_tensor)], dim=0)
            noisy_slices.append(feats_t)
        node_features_over_time_noisy = torch.stack(noisy_slices, dim=1).to(device)

        G_noisy_simple = nx.DiGraph()
        G_noisy_simple.add_nodes_from(G_noisy.nodes())
        G_noisy_simple.add_edges_from(G_noisy.edges())
        edge_index_noisy = from_networkx(G_noisy_simple).edge_index.to(device)

        with torch.no_grad():
            outputs_t = []
            for t in range(T):
                h_t = model(edge_index_noisy, x=node_features_over_time_noisy[:, t, :])
                outputs_t.append(h_t)
            X_noisy = torch.stack(outputs_t, dim=1)
            att_output_noisy = model.seq_model(X_noisy)
            if att_output_noisy.ndim == 2:
                att_output_noisy = att_output_noisy.unsqueeze(1)
            Np, Tp, Fp = att_output_noisy.shape
            assert Tp == T
            X_flat_noisy = att_output_noisy.reshape(Np, Tp * Fp).detach().cpu().numpy()

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_flat_noisy)
        clf = IsolationForest(n_estimators=100, contamination=contamination, random_state=random_state)
        clf.fit(X_scaled)
        anomaly_scores = -clf.decision_function(X_scaled)

        real_scores = anomaly_scores[:original_N]
        fake_scores = anomaly_scores[original_N:]

        thr = np.percentile(real_scores, 95) if real_scores.size else np.nan
        real_flags = (real_scores > thr) if real_scores.size else np.array([])
        fake_flags = (fake_scores > thr) if fake_scores.size else np.array([])

        FPR = real_flags.mean() if real_flags.size else float("nan")
        TPR = fake_flags.mean() if fake_flags.size else float("nan")

        tpr_list.append(TPR)
        fpr_list.append(FPR)

        results_per_iter.append({
            "iteration": it,
            "t_injected": int(t_i),
            "n_noise": int(n_noise_nodes),
            "threshold_p95_real": float(thr) if not np.isnan(thr) else float("nan"),
            "mean_real_score": float(real_scores.mean()) if real_scores.size else float("nan"),
            "mean_fake_score": float(fake_scores.mean()) if fake_scores.size else float("nan"),
            "TPR": float(TPR),
            "FPR": float(FPR),
        })

    summary = {
        "iters": int(n_iters),
        "k_percent": float(k_percent),
        "connect_prob": float(connect_prob),
        "tpr_mean": float(np.nanmean(tpr_list)) if len(tpr_list) else float("nan"),
        "tpr_std": float(np.nanstd(tpr_list)) if len(tpr_list) else float("nan"),
        "fpr_mean": float(np.nanmean(fpr_list)) if len(tpr_list) else float("nan"),
        "fpr_std": float(np.nanstd(fpr_list)) if len(tpr_list) else float("nan"),
    }
    return {"summary": summary, "results_per_iter": results_per_iter}


# ==============================
#   התאמת nhid אוטומטית (רשות)
# ==============================
def _auto_fit_nhid_and_rebuild(model_cls, args, device, edge_index, probe_x, time_length, max_tries=2):
    for _ in range(max_tries):
        model = model_cls(args, time_length=time_length).to(device)
        try:
            with torch.no_grad():
                _ = model(edge_index, x=probe_x)
            return model
        except RuntimeError as e:
            msg = str(e)
            m = re.search(r"mat1 and mat2 shapes cannot be multiplied \(\d+x(\d+) and (\d+)x(\d+)\)", msg)
            if m is None:
                raise
            u_dim = int(m.group(1))
            suggested_nhid = max(1, u_dim - 1)
            if getattr(args, "nhid", None) == suggested_nhid:
                raise
            print(f"🔧 Adjusting args.nhid from {getattr(args,'nhid',None)} to {suggested_nhid} based on probe (U={u_dim}).")
            args.nhid = suggested_nhid
            continue
    raise RuntimeError("Failed to auto-fit nhid after probe attempts.")


# ==============================
#   Main
# ==============================
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args()
    args = _ensure_dynhat_defaults(args)
    device = torch.device(args.device)

    # -----------------------------------------------------------
    # שלב 1: Node2Vec דינמי [N, T, F]
    # -----------------------------------------------------------
    num_nodes, T_bins, snapshots = load_manifest_and_snapshots(args.data_root)
    if args.t_max is not None:
        use_T = min(args.t_max, T_bins)
        snapshots = {t: snapshots[t] for t in range(use_T)}
        T_bins = use_T

    embedding_matrix = build_dynamic_node2vec(
        snapshots=snapshots,
        num_nodes=num_nodes,
        T=T_bins,
        emb_dim=args.emb_dim,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        workers=args.workers,
        window=args.window,
    )  # [N, T, F]
    print("✅ embedding_matrix shape:", tuple(embedding_matrix.shape))

    with torch.no_grad():
        em_np = embedding_matrix.detach().cpu().numpy()
        fin = np.isfinite(em_np)
        print(f"[DBG] embedding_matrix raw: shape={em_np.shape}, finite%={100.0*fin.mean():.2f}%")

    args.nfeat = int(embedding_matrix.shape[-1])
    args.num_nodes = int(embedding_matrix.shape[0])

    # -----------------------------------------------------------
    # שלב 2: טעינת גרף/פיצ'רים/לייבלים + ספליטים
    # -----------------------------------------------------------
    adj, features_sp, labels_np, idx_train, idx_val, idx_test = load_citation_data(
        dataset_str="dblpv13",
        use_feats=True,
        data_path=args.data_root
    )

    edge_index, _ = from_scipy_sparse_matrix(adj)
    labels = torch.LongTensor(labels_np).to(device)
    edge_index = edge_index.to(device)

    if labels.dim() == 2 and labels.size(1) > 1:
        labels = labels.argmax(1)
    if labels.dtype is not torch.long:
        labels = labels.long()

    args.num_classes = int(labels.max().item() + 1)
    if not hasattr(args, "nclass"):
        args.nclass = args.num_classes
    single_class_problem = args.num_classes < 2

    print(f"🔧 Dynhat init params: num_nodes={args.num_nodes}, nfeat={args.nfeat}, nclass={args.nclass}, nout={args.nout}")

    Dynhat.c = float(getattr(args, "c", getattr(args, "curvature", getattr(args, "c0", 1.0))))

    # -----------------------------------------------------------
    # שלב 3: מודל Dynhat + אימון (ללא ולידציה בתוך הלולאה)
    # -----------------------------------------------------------
    probe_x = embedding_matrix[:, 0, :].to(device)
    model = _auto_fit_nhid_and_rebuild(Dynhat, args, device, edge_index, probe_x, time_length=T_bins).to(device)
    args.fix_curvature = True

    cls_head = torch.nn.Linear(args.nhid + 1, args.num_classes).to(device)

    labels_train = torch.as_tensor(labels[idx_train], device=device)
    counts = torch.bincount(labels_train, minlength=args.num_classes).float()
    class_weights = (counts.sum() / (counts + 1e-6)).to(device)
    class_weights = class_weights / class_weights.mean()
    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(cls_head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    print(f"✅ Dynhat ready with nhid={args.nhid} (agg_feat_size={args.nhid + 1}).")

    train_losses = []

    if not single_class_problem and args.max_epoch:
        for epoch in range(args.max_epoch):
            model.train(); cls_head.train()
            optimizer.zero_grad()

            temporal_outputs = []
            for t in range(T_bins):
                x_t = embedding_matrix[:, t, :].to(device)
                x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1e6, neginf=-1e6)
                x_t = x_t - x_t.mean(dim=0, keepdim=True)
                x_t = torch.nn.functional.normalize(x_t, p=2, dim=1) * float(args.norm_scale)

                h_t = model(edge_index, x=x_t)
                h_t = torch.nan_to_num(h_t, nan=0.0, posinf=1e6, neginf=-1e6)
                temporal_outputs.append(h_t)

            X = torch.stack(temporal_outputs, dim=1)     # [N, T, C]

            if epoch == 0:
                dbg_seq_temporal_layer_once(model.seq_model, X)

            att = model.seq_model(X)                     # [N, T, C] או [N, C]
            feat_last = att if att.ndim == 2 else att[:, -1, :]  # [N, C]

            logits = cls_head(feat_last)                 # [N, num_classes]
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

            loss = ce_loss(logits[idx_train], labels[idx_train])
            if not torch.isfinite(loss):
                print(f"[Epoch {epoch}] Loss not finite ({loss.item():.4f}). Reducing LR ×0.1 and skipping.")
                for g in optimizer.param_groups:
                    g['lr'] = max(g['lr'] * 0.1, 1e-5)
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(cls_head.parameters()),
                max_norm=1.0
            )
            optimizer.step()

            train_losses.append(loss.item())
            print(f"[Epoch {epoch}] Train Loss: {loss.item():.4f} | grad_norm={(float(grad_norm) if torch.isfinite(grad_norm) else float('nan')):.6f}")

    elif not single_class_problem and args.max_epoch is None:
        print("ℹ Training skipped because --max-epoch=None (set a value to train).")
    else:
        print("⚠ Detected a single class in labels. Skipping supervised cross-entropy training.")
        model.eval(); cls_head.eval()

    # -----------------------------------------------------------
    # גרף: Loss בלבד (ללא ולידציה)
    # -----------------------------------------------------------
    if len(train_losses) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train Loss")
        plt.legend()
        plt.grid(True)
        plot_path = "loss_curve.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"\n✅ Saved training loss plot to: {plot_path}")

    # -----------------------------------------------------------
    # === att_output במצב eval לשימוש ב-IF/LOF ===
    # -----------------------------------------------------------
    model.eval(); cls_head.eval()
    with torch.no_grad():
        outs_eval = []
        for t in range(T_bins):
            x_t = embedding_matrix[:, t, :].to(device)
            x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1e6, neginf=-1e6)
            x_t = x_t - x_t.mean(dim=0, keepdim=True)
            x_t = torch.nn.functional.normalize(x_t, p=2, dim=1) * float(args.norm_scale)

            h_t = model(edge_index, x=x_t)
            h_t = torch.nan_to_num(h_t, nan=0.0, posinf=1e6, neginf=-1e6)
            outs_eval.append(h_t)

        X_eval = torch.stack(outs_eval, dim=1)   # [N, T, C]
        att_output = model.seq_model(X_eval)
        if att_output.ndim == 2:
            att_output = att_output.unsqueeze(1) # [N,1,C]
        dbg_stats("att_output eval", att_output, per_time=True)

    # -----------------------------------------------------------
    # שלב 5: IF+LOF טמפורלי על att_output
    # -----------------------------------------------------------
    N_eval, T_eval, C_eval = att_output.shape
    lof_k = max(2, min(args.lof_n_neighbors, N_eval - 1))
    contam = min(max(args.contamination, max(1, int(0.02 * N_eval)) / N_eval), 0.20)
    print(f"\n📏 Shapes: N={N_eval}, T={T_eval}, C={C_eval}  |  LOF.k={lof_k}, IF.contamination={contam:.4f}")

    AS_if  = np.full((N_eval, T_eval), np.nan, dtype=np.float32)
    AS_lof = np.full((N_eval, T_eval), np.nan, dtype=np.float32)
    valid_t = []

    for t in range(T_eval):
        X_t = att_output[:, t, :].detach().cpu().numpy()

        if not np.isfinite(X_t).all():
            n_bad = np.size(X_t) - np.isfinite(X_t).sum()
            print(f"[WARN] Found {n_bad} non-finite entries at t={t}. Fixing with zscore guards.")
            mu = np.nanmean(X_t, axis=0, keepdims=True)
            sigma = np.nanstd(X_t, axis=0, keepdims=True)
            sigma[sigma == 0] = 1.0
            X_t = (X_t - mu) / sigma
            X_t = np.nan_to_num(X_t, nan=0.0, posinf=1e6, neginf=-1e6)

    #     סינון תכונות אינפורמטיביות
        if X_t.shape[1] == 0:
            print(f"[INFO] t={t}: empty C. Skipping."); continue
        col_std = X_t.std(axis=0)
        informative = col_std > 0
        if informative.sum() == 0:
            print(f"[INFO] t={t}: not enough informative features after filtering (C'=0). Skipping.")
            continue
        X_t = X_t[:, informative]

        scaler_t = MinMaxScaler()
        X_t_scaled = scaler_t.fit_transform(X_t)

        try:
            if_clf_t = IsolationForest(
                n_estimators=100,
                contamination=contam,
                random_state=args.random_state,
                n_jobs=-1
            )
            if_clf_t.fit(X_t_scaled)
            AS_if[:, t] = -if_clf_t.decision_function(X_t_scaled)
        except Exception as e:
            print(f"[IF] t={t} failed: {e}")

        try:
            lof_t = LocalOutlierFactor(
                n_neighbors=lof_k,
                contamination=contam,
                novelty=False,
                n_jobs=-1
            )
            _ = lof_t.fit_predict(X_t_scaled)
            AS_lof[:, t] = -(lof_t.negative_outlier_factor_)
        except Exception as e:
            print(f"[LOF] t={t} failed: {e}")

        valid_t.append(t)

    if len(valid_t) == 0:
        print("⚠ No valid timesteps for anomaly scoring (all t were constant/invalid).")
        mu_if = np.full((N_eval,), np.nan, dtype=np.float32)
        std_if = np.full((N_eval,), np.nan, dtype=np.float32)
        mu_lof = np.full((N_eval,), np.nan, dtype=np.float32)
        std_lof = np.full((N_eval,), np.nan, dtype=np.float32)
    else:
        print(f"[INFO] mean valid t per node: {len(valid_t)/T_eval:.2f} / {T_eval}")
        mu_if   = np.nanmean(AS_if[:, valid_t],  axis=1)
        std_if  = np.nanstd(AS_if[:, valid_t],   axis=1)
        mu_lof  = np.nanmean(AS_lof[:, valid_t], axis=1)
        std_lof = np.nanstd(AS_lof[:, valid_t],  axis=1)

    _print_topk("IF by mean (μ)",  mu_if,  k=min(20, N_eval))
    _print_topk("IF by std (σ)",   std_if, k=min(20, N_eval))
    _print_topk("LOF by mean (μ)", mu_lof, k=min(20, N_eval))
    _print_topk("LOF by std (σ)",  std_lof, k=min(20, N_eval))

    try:
        if np.isfinite(mu_if).any() and np.isfinite(mu_lof).any():
            rho, p = spearmanr(mu_if, mu_lof)
            if np.isfinite(rho):
                print(f"\n🔗 Spearman(IF.mean, LOF.mean) = {rho:.4f}  (p={p:.2e})")
            else:
                print("\n🔗 Spearman skipped: non-finite result.")
        else:
            print("\n🔗 Spearman skipped: non-finite vectors.")
    except Exception as e:
        print("Spearman failed:", e)

    _summ("IF.mean over nodes",  mu_if)
    _summ("IF.std  over nodes",  std_if)
    _summ("LOF.mean over nodes", mu_lof)
    _summ("LOF.std  over nodes", std_lof)

    df_summary = pd.DataFrame({
        "mu_if":  mu_if,  "std_if": std_if,
        "mu_lof": mu_lof, "std_lof": std_lof,
    })
    df_summary.index.name = "node_id"
    df_summary.to_csv("temporal_anomaly_summary.csv")
    print("💾 saved: temporal_anomaly_summary.csv")

    _save_hist(mu_if,  "hist_mu_if.png",  title="IF mean per node")
    _save_hist(std_if, "hist_std_if.png", title="IF std per node")
    _save_hist(mu_lof, "hist_mu_lof.png", title="LOF mean per node")
    _save_hist(std_lof,"hist_std_lof.png",title="LOF std per node")

    # -----------------------------------------------------------
    # שלב 6: Noise Injection Validation (אחרי IF/LOF)
    # -----------------------------------------------------------
    G_nx = nx.DiGraph()
    rows, cols = adj.nonzero()
    G_nx.add_nodes_from(range(adj.shape[0]))
    G_nx.add_edges_from(zip(rows.tolist(), cols.tolist()))

    noise_res = validate_with_noise_injection(
        G_original=G_nx,
        embedding_matrix=embedding_matrix.to(device),
        model=model,
        T=T_bins,
        k_percent=args.noise_percent,
        connect_prob=args.noise_connect_prob,
        n_iters=args.noise_iters,
        contamination=contam,
        random_state=args.random_state,
    )

    print("\n=== Stage-4 Validation Summary ===")
    print(noise_res["summary"])

    per_iter = noise_res.get("results_per_iter", [])
    if per_iter:
        delta_means = [r["mean_fake_score"] - r["mean_real_score"] for r in per_iter]
        tprs = [r["TPR"] for r in per_iter]
        print(f"\n🎯 Noise-vs-Real separation: Δmean (noise - real) avg = {np.mean(delta_means):.4f} ± {np.std(delta_means):.4f}")
        print(f"🎯 Mean TPR@95% over iters = {np.mean(tprs):.3f}  (std={np.std(tprs):.3f})")

    if args.save_bundle:
        os.makedirs(os.path.dirname(args.save_bundle), exist_ok=True)
        torch.save({
            "embedding_matrix": embedding_matrix.cpu(),
            "edge_index": edge_index.cpu(),
            "labels": labels.cpu(),
            "att_output": att_output.cpu(),
            "noise_validation": noise_res
        }, args.save_bundle)
        print(f"💾 Saved bundle to: {args.save_bundle}")


if __name__ == "__main__":
    main()
