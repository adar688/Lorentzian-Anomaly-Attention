# -*- coding: utf-8 -*-

"""
מריצים את הקובץ הזה לאחר שטענו את הדאטה סט ועשינו לו עיבוד.
כאן כבר מתחילים לעבוד איתו: טוענים את הקבצים שעיבדנו, מפיקים אמבדינגים דינמיים (Node2Vec),
מריצים Dynhat, מחשבים IF/LOF טמפורלי, ומדפיסים/שומרים מדדים להבנת התוצאות.
(גרסת דיבוג בלבד לשכבת הקשב הטמפורלי – ללא שינוי התנהגות)
"""

import os
import sys
import argparse
import copy
import re
import json
import math
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


# ======================================================
#           UTILITIES (דיבוג בלבד – לא משנים זרימה)
# ======================================================

def dbg_stats(name, t: torch.Tensor, per_time=False, max_print=5):
    """מדפיס סטטיסטיקות בסיסיות על טנזור: finite%, min/max/mean/std. אופציה לפירוק לפי זמן."""
    try:
        x = t.detach()
    except Exception:
        x = t
    x = x.float()
    x_np = x.reshape(-1).cpu().numpy()
    finite = np.isfinite(x_np)
    pct_fin = 100.0 * finite.mean() if x_np.size else 0.0
    msg = f"[DBG] {name}: shape={tuple(x.shape)}, finite%={pct_fin:.2f}%"
    if x_np.size and finite.any():
        xf = x_np[finite]
        msg += f", min={xf.min():.3e}, max={xf.max():.3e}, mean={xf.mean():.3e}, std={xf.std():.3e}"
    print(msg)

    if per_time and t.ndim == 3:
        N, T, C = t.shape
        per = []
        for ti in range(T):
            xi = t[:, ti, :].reshape(-1).float().cpu().numpy()
            fi = np.isfinite(xi)
            per.append(round(100.0 * fi.mean() if xi.size else 0.0, 2))
        print(f"[DBG] {name}: finite% per t: {per[:max_print]}{' ...' if T > max_print else ''}")


def _find_first_attr(obj, names):
    """מנסה לאתר אטריבוט ראשון מבין רשימת שמות נפוצים (ל-Q/K/V projectors)."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n), n
    return None, None


def dbg_seq_temporal_layer_once(seq_layer, X_ntc: torch.Tensor):
    """
    דיבוג חד-פעמי לשכבת הקשב הטמפורלי.
    לא משנה התנהגות: רק מדפיס סטטיסטיקות קלט/פלט ומנסה (אם אפשר) לחשב Q/K/V
    על דגימה קטנה כדי לזהות Overflow/NaN לפני softmax.
    """
    print("\n===== TEMPORAL ATTENTION DEBUG =====")
    dbg_stats("seq_input(N,T,C)", X_ntc, per_time=True)

    # מנסים למצוא מקרני Q/K/V בשמות נפוצים
    q_lin, qn = _find_first_attr(seq_layer, ["q_proj", "W_q", "q_linear", "fc_q", "query", "lin_q"])
    k_lin, kn = _find_first_attr(seq_layer, ["k_proj", "W_k", "k_linear", "fc_k", "key", "lin_k"])
    v_lin, vn = _find_first_attr(seq_layer, ["v_proj", "W_v", "v_linear", "fc_v", "value", "lin_v"])

    if q_lin is None or k_lin is None or v_lin is None:
        print(f"[DBG] Q/K/V projectors not found on {type(seq_layer).__name__} "
              f"(looked for common names). Skipping internals and calling forward normally.")
        try:
            with torch.no_grad():
                out = seq_layer(X_ntc)  # קריאה רגילה
                dbg_stats("seq_output", out, per_time=True)
        except Exception as e:
            print(f"[DBG] seq_layer forward raised: {e}")
        print("===== END TEMPORAL ATTENTION DEBUG =====\n")
        return

    print(f"[DBG] Found projectors: Q={qn}, K={kn}, V={vn}")
    N, T, C = X_ntc.shape
    X2 = X_ntc  # לא משנים צורה; רק מודדים

    # מחשבים Q/K/V על דגימה קטנה כדי לא להכביד
    with torch.no_grad():
        Nprobe = min(1024, N)
        Xs = X2[:Nprobe]                 # [n,T,C]
        Xflat = Xs.reshape(-1, C)        # [n*T, C]

        try:
            Q = q_lin(Xflat)             # [n*T, d]
            K = k_lin(Xflat)
            V = v_lin(Xflat)
            dbg_stats("Q(flat)", Q)
            dbg_stats("K(flat)", K)
            dbg_stats("V(flat)", V)
            d = Q.shape[-1]
            # נבחן צעד זמן ראשון
            Qt = Q.reshape(Nprobe, T, -1)[:, 0, :]  # [n, d]
            Kt = K.reshape(Nprobe, T, -1)[:, 0, :]
            # ציוני attention בסיסיים בין m דוגמאות
            m = min(128, Nprobe)
            Qt = Qt[:m]
            Kt = Kt[:m]
            scores = (Qt @ Kt.T) / math.sqrt(max(d, 1))
            dbg_stats("scores(t=0 small)", scores)
            # בדיקת softmax ידנית (עם הסטה) למניעת overflow
            scores_np = scores.cpu().numpy()
            if np.isfinite(scores_np).all():
                s_max = scores_np.max(axis=1, keepdims=True)
                sm = np.exp(scores_np - s_max)          # stabilization
                w = sm / (sm.sum(axis=1, keepdims=True) + 1e-12)
                fin = np.isfinite(w).mean() * 100.0
                print(f"[DBG] softmax(weights) finite% (manual small) = {fin:.2f}%")
                print(f"[DBG] weights row max≈ {w.max():.3f}, min≈ {w.min():.3f}")
            else:
                print("[DBG] scores already non-finite before softmax ⇒ זו כנראה נקודת הכשל.")
        except Exception as e:
            print(f"[DBG] projector path failed: {e}")

        # forward מלא של השכבה לבדיקת פלט בפועל
        try:
            out = seq_layer(X2)
            dbg_stats("seq_output", out, per_time=True)
        except Exception as e:
            print(f"[DBG] seq_layer forward raised: {e}")

    print("===== END TEMPORAL ATTENTION DEBUG =====\n")


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

    # מניפולד/גיאומטריה
    p.add_argument("--manifold", type=str, default="Hyperboloid",
                   help="שם המניפולד למודל Dynhat (למשל: 'Hyperboloid'/'lorentz', 'poincare', 'euclidean')")
    p.add_argument("--fix_curvature", action="store_true",
                   help="אם מצוין: העקמומיות (curvature) מקובעת ולא מתעדכנת באימון")
    p.add_argument("--curvature", type=float, default=1.0,
                   help="ערך עקמומיות התחלתי |K| (לדוגמה 1.0). חלק מהיישומים קוראים לזה c או c0")
    p.add_argument("--c0", type=float, default=1.0,
                   help="שם אלטרנטיבי לעקמומיות התחלתית אם המודל משתמש בשם זה")

    # היפר־פרמטרים של Dynhat
    p.add_argument("--nhid", type=int, default=32, help="גודל השכבה החבויה (hidden size)")
    p.add_argument("--dropout", type=float, default=0.5, help="Dropout כולל")
    p.add_argument("--attn-dropout", type=float, default=0.0, help="Dropout בשכבת הקשב (אם קיים)")
    p.add_argument("--feat-dropout", type=float, default=0.0, help="Dropout על פיצ'רים (אם קיים)")
    p.add_argument("--alpha", type=float, default=0.2, help="LeakyReLU negative slope")
    p.add_argument("--nheads", type=int, default=1, help="מספר ראשים בקשב גרפי (אם קיים)")
    p.add_argument("--temporal_attention_layer_heads", type=int, default=1,
                   help="מספר הראשים בשכבת הקשב הטמפורלי")
    p.add_argument("--bias", type=int, choices=[0, 1], default=1,
                   help="להשתמש ב-bias (1) או לא (0) בשכבות שרלוונטיות")
    p.add_argument("--residual", action="store_true", help="לאפשר חיבורי residual אם קיים במודל")
    p.add_argument("--batch-norm", action="store_true", help="לאפשר BatchNorm אם קיים במודל")
    p.add_argument("--aggregation", type=str, default="att",
                   choices=["att", "mean", "sum", "max"],
                   help="סוג האגרגציה הטמפורלית/גרפית במודל (att/mean/sum/max)")
    p.add_argument("--nfeat", type=int, default=32,
                   help="מספר הפיצ'רים לקלט המודל (ברירת מחדל: ייגזר מ-embedding_matrix)")
    p.add_argument("--nout", type=int, default=32,
                   help="מספר יחידות פלט של המודל (ברירת מחדל: יוגדר לפי num_classes)")
    p.add_argument("--seq-model", dest="seq_model", type=str, default="gru",
                   choices=["gru", "lstm", "transformer", "none"],
                   help="מודל רצף טמפורלי בתוך Dynhat")
    p.add_argument("--seq-hidden", dest="seq_hidden", type=int, default=128,
                   help="גודל החבוי של מודל הרצף")
    p.add_argument("--seq-layers", dest="seq_layers", type=int, default=1,
                   help="מספר שכבות במודל הרצף")
    p.add_argument("--seq-dropout", dest="seq_dropout", type=float, default=0.0,
                   help="Dropout בתוך מודל הרצף")

    # אימון
    p.add_argument("--max-epoch", type=int, default=32, help="מספר אפוקים לאימון Dynhat")
    p.add_argument("--lr", type=float, default=1e-2, help="למידה - Adam LR")
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

    # סקייל קטן לאמבדינגים לפני המודל (כמו שיש לך כבר בלוגים) – רק פרמטר, לא שינוי התנהגות
    p.add_argument("--norm-scale", type=float, default=0.1, help="סקייל לנורמליזציה (debug parity)")

    return p.parse_args()


def _ensure_dynhat_defaults(args):
    """רשת ביטחון: ממלא דיפולטים אם חסר ארגומנט שה-Dynhat עשוי לבקש. לא דורסת ערכים קיימים."""
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

    # התאמות שמות נפוצות
    if not hasattr(args, "heads") and hasattr(args, "nheads"):
        args.heads = args.nheads
    if isinstance(getattr(args, "bias", True), int):
        args.bias = bool(args.bias)
    if not hasattr(args, "c"):
        args.c = float(getattr(args, "curvature", getattr(args, "c0", 1.0)))

    return args


# ==============================
#   כלי אבחון / תוצרים קיימים
# ==============================
def _summ(name, arr):
    arr = np.asarray(arr).ravel()
    if arr.size == 0 or not np.isfinite(arr).any():
        print(f"\n{name}: (no finite data)")
        return
    q = np.percentile(arr[np.isfinite(arr)], [0, 1, 5, 25, 50, 75, 95, 99, 100])
    print(f"\n{name}:")
    print(f"  mean={arr[np.isfinite(arr)].mean():.6f}  std={arr[np.isfinite(arr)].std():.6f}  "
          f"min={q[0]:.6f}  p1={q[1]:.6f}  p5={q[2]:.6f}")
    print(f"  p25={q[3]:.6f}  p50={q[4]:.6f}  p75={q[5]:.6f}  p95={q[6]:.6f}  p99={q[7]:.6f}  max={q[8]:.6f}")


def _print_topk(title, scores, k=10):
    k = min(k, len(scores))
    order = np.argsort(-scores)[:k]
    print(f"\n{title} (Top-{k}):")
    for rank, idx in enumerate(order, 1):
        print(f"  {rank:>2}. node={int(idx):>6}  score={float(scores[idx]):.6f}")
    return order


def _save_hist(arr, fname, bins=40, title=None):
    a = np.asarray(arr).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        print(f"ℹ️  skipped histogram {fname}: no finite data.")
        return
    plt.figure(figsize=(8, 4.5))
    plt.hist(a, bins=bins)
    if title:
        plt.title(title)
    plt.xlabel("score"); plt.ylabel("count"); plt.tight_layout()
    plt.savefig(fname); plt.close()
    print(f"💾 saved: {fname}")


# ========================================================
#  Stage 4 unified: validate_with_noise_injection (IF בלבד)
#   (ללא שינוי – השארתי כפי שהיה אצלך)
# ========================================================
def validate_with_noise_injection(
    G_original,
    embedding_matrix,           # Tensor [N, T, F]
    model,                      # Dynhat
    T,                          # num timesteps
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

        mu = embedding_matrix.mean(dim=(0, 1)).to(device)  # [F]
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
        node_features_over_time_noisy = torch.stack(noisy_slices, dim=1).to(device)  # [N', T, F]

        G_noisy_simple = nx.DiGraph()
        G_noisy_simple.add_nodes_from(G_noisy.nodes())
        G_noisy_simple.add_edges_from(G_noisy.edges())
        edge_index_noisy = from_networkx(G_noisy_simple).edge_index.to(device)

        with torch.no_grad():
            outputs_t = []
            for t in range(T):
                h_t = model(edge_index_noisy, x=node_features_over_time_noisy[:, t, :])
                outputs_t.append(h_t)
            X_noisy = torch.stack(outputs_t, dim=1)                 # [N', T, C]
            # דיבוג בלבד: לפני קריאת שכבת הקשב
            dbg_stats("noisy_seq_input(N,T,C)", X_noisy, per_time=True)
            att_output_noisy = model.seq_model(X_noisy)             # קריאה רגילה (לא משנים)
            if att_output_noisy.ndim == 2:
                att_output_noisy = att_output_noisy.unsqueeze(1)    # [N', 1, C]
            dbg_stats("noisy_seq_output", att_output_noisy, per_time=True)

            Np, Tp, Fp = att_output_noisy.shape
            assert Tp == T or Tp == 1, "שכבת הרצף צריכה לשמר את ציר הזמן או להחזיר [N,1,C]"
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
#   Main
# ==============================
def main():
    # UTF-8 למסופים מסוימים
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args()
    args = _ensure_dynhat_defaults(args)
    device = torch.device(args.device)

    # -----------------------------------------------------------
    # שלב 1: טעינת סנאפשוטים ובניית Node2Vec דינמי [N, T, F]
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
    )  # torch.Tensor [N, T, F]
    print("✅ embedding_matrix shape:", tuple(embedding_matrix.shape))

    # דיבוג Node2Vec בסיסי (כבר היה אצלך בלוגים)
    em = embedding_matrix
    em_np = em.detach().cpu().numpy()
    fin_all = np.isfinite(em_np).mean() * 100.0
    print(f"[DBG] embedding_matrix raw: shape={em_np.shape}, finite%={fin_all:.2f}%")
    per_t = [round(float(np.isfinite(em_np[:, t, :]).mean() * 100.0), 2) for t in range(em_np.shape[1])]
    print(f"[DBG] embedding_matrix raw: finite% per t: {per_t}")
    per_t_cols_std = []
    for t in range(em_np.shape[1]):
        cols = em_np[:, t, :]
        per_t_cols_std.append(round(100.0 * (np.std(cols, axis=0) > 0).mean(), 2))
    print(f"[DBG] embedding_matrix raw: %cols std>0 per t: {per_t_cols_std}")

    # קיבוע nfeat & num_nodes לפי אמבדינגים
    args.nfeat = int(embedding_matrix.shape[-1])     # F
    args.num_nodes = int(embedding_matrix.shape[0])  # N

    # -----------------------------------------------------------
    # שלב 2: טעינת גרף/פיצ'רים/לייבלים + ספליטים (פורמט custom_out)
    # -----------------------------------------------------------
    adj, features_sp, labels_np, idx_train, idx_val, idx_test = load_citation_data(
        dataset_str="dblpv13",
        use_feats=True,
        data_path=args.data_root
    )

    # ======== DEBUG BLOCK: class counts overall + per split ========
    print("\n[DEBUG] labels_np shape:", labels_np.shape, "dtype:", getattr(labels_np, "dtype", type(labels_np)))
    labels_vec = labels_np
    if getattr(labels_vec, "ndim", 1) == 2 and labels_vec.shape[1] > 1:
        labels_vec = labels_vec.argmax(1)

    uniq_all, cnt_all = np.unique(labels_vec, return_counts=True)
    print("[DEBUG] classes in ALL:", dict(zip(uniq_all.tolist(), cnt_all.tolist())))

    def _dist(name, v):
        u, c = np.unique(v, return_counts=True)
        print(f"[DEBUG] classes in {name}:", dict(zip(u.tolist(), c.tolist())))

    _dist("TRAIN", labels_vec[idx_train])
    _dist("VAL",   labels_vec[idx_val])
    _dist("TEST",  labels_vec[idx_test])

    lb_path = os.path.join(args.data_root, "label_binarizer.json")
    if os.path.exists(lb_path):
        try:
            with open(lb_path, "r", encoding="utf-8") as f:
                lbj = json.load(f)
            print("[DEBUG] label_binarizer classes_:", lbj.get("classes"))
        except Exception as e:
            print("[DEBUG] failed reading label_binarizer.json:", e)
    else:
        print("[DEBUG] label_binarizer.json not found at", lb_path)
    # ======== END DEBUG BLOCK ========

    edge_index, _ = from_scipy_sparse_matrix(adj)
    features = torch.FloatTensor(features_sp.todense()).to(device)
    labels = torch.LongTensor(labels_np).to(device)
    edge_index = edge_index.to(device)

    # לייבלים – ודא אינדקסים (לא one-hot)
    if labels.dim() == 2 and labels.size(1) > 1:
        labels = labels.argmax(1)
    if labels.dtype is not torch.long:
        labels = labels.long()

    # num_classes / nclass
    args.num_classes = int(labels.max().item() + 1)
    if not hasattr(args, "nclass"):
        args.nclass = args.num_classes

    # בעיית מחלקה אחת → אין CrossEntropy
    single_class_problem = args.num_classes < 2

    print(f"🔧 Dynhat init params: num_nodes={args.num_nodes}, nfeat={args.nfeat}, nclass={args.nclass}, nout={args.nout}")

    # fallback: ערך עקמומיות ברמת המחלקה (אם Dynhat.__init__ משתמש לפני הגדרה)
    Dynhat.c = float(getattr(args, "c", getattr(args, "curvature", getattr(args, "c0", 1.0))))

    # -----------------------------------------------------------
    # שלב 3: מודל Dynhat + אימון עם ראש סיווג (ללא שינוי התנהגות)
    # -----------------------------------------------------------
    probe_x = embedding_matrix[:, 0, :].to(device)
    model = Dynhat(args, time_length=T_bins).to(device)

    # ראש סיווג
    cls_head = torch.nn.Linear(args.nhid + 1, args.num_classes).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(cls_head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    print(f"✅ Dynhat ready with nhid={args.nhid} (agg_feat_size={args.nhid + 1}).")

    train_losses: list[float] = []

    if not single_class_problem and args.max_epoch:
        for epoch in range(args.max_epoch):
            model.train(); cls_head.train()
            optimizer.zero_grad()

            temporal_outputs = []
            for t in range(T_bins):
                x_t = embedding_matrix[:, t, :].to(device)
                # guard נגד NaN/Inf + נרמול עדין (כמו בלוגים שלך)
                x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1e6, neginf=-1e6)
                x_t = torch.nn.functional.normalize(x_t, p=2, dim=1) * float(args.norm_scale)
                # דיבוג דוגמה (כבר הופיע אצלך בלוגים)
                if t == 0 and epoch == 0:
                    rows_std = x_t.std(dim=1)
                    cols_std = x_t.std(dim=0)
                    pct_rows = float((rows_std > 0).float().mean().item() * 100.0)
                    pct_cols = float((cols_std > 0).float().mean().item() * 100.0)
                    print(f"[DBG] x_t after normalize (*norm-scale): shape={tuple(x_t.shape)}, finite%={(torch.isfinite(x_t).float().mean().item()*100):.2f}%")
                    print(f"[DBG] x_t after normalize (*norm-scale): rows std>0%={pct_rows:.2f}%, cols std>0%={pct_cols:.2f}%")

                h_t = model(edge_index, x=x_t)           # [N, C]
                h_t = torch.nan_to_num(h_t, nan=0.0, posinf=1e6, neginf=-1e6)
                if t == 0 and epoch == 0:
                    rows_std = h_t.std(dim=1)
                    cols_std = h_t.std(dim=0)
                    pct_rows = float((rows_std > 0).float().mean().item() * 100.0)
                    pct_cols = float((cols_std > 0).float().mean().item() * 100.0)
                    print(f"[DBG] h_t from Dynhat (t=0): shape={tuple(h_t.shape)}, finite%={(torch.isfinite(h_t).float().mean().item()*100):.2f}%")
                    print(f"[DBG] h_t from Dynhat (t=0): rows std>0%={pct_rows:.2f}%, cols std>0%={pct_cols:.2f}%")

                temporal_outputs.append(h_t)

            X = torch.stack(temporal_outputs, dim=1)     # [N, T, C]

            # --- דיבוג חד-פעמי של שכבת הקשב (אפוק ראשון בלבד) ---
            if epoch == 0:
                with torch.no_grad():
                    dbg_seq_temporal_layer_once(model.seq_model, X)

            att = model.seq_model(X)                     # [N, T, C] או [N, C]
            if att.ndim == 2:
                att = att.unsqueeze(1)

            feat_last = att[:, -1, :]                    # [N, C]
            logits = cls_head(feat_last)                 # [N, num_classes]
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

            loss = F.cross_entropy(logits[idx_train], labels[idx_train])
            if not torch.isfinite(loss):
                print(f"[Epoch {epoch}] Loss not finite ({loss.item():.4f}). Reducing LR ×0.1 and skipping.")
                for g in optimizer.param_groups:
                    g['lr'] = max(g['lr'] * 0.1, 1e-5)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(cls_head.parameters()), max_norm=1.0)

            # grad_norm להדפסה (לא קריטי)
            total_norm = 0.0
            for p in list(model.parameters()) + list(cls_head.parameters()):
                if p.grad is not None and torch.isfinite(p.grad).all():
                    total_norm += p.grad.data.norm(2).item()

            optimizer.step()
            train_losses.append(loss.item())
            print(f"[Epoch {epoch}] Train Loss: {loss.item():.4f} | grad_norm={total_norm if np.isfinite(total_norm) else 0.0:.6f}")
    elif not single_class_problem and args.max_epoch is None:
        print("ℹ Training skipped because --max-epoch=None (set a value to train).")
    else:
        print("⚠ Detected a single class in labels. Skipping supervised cross-entropy training.")
        model.eval(); cls_head.eval()

    # -----------------------------------------------------------
    # === חישוב att_output במצב eval כדי שישמש ל-IF + LOF ===
    # -----------------------------------------------------------
    model.eval(); cls_head.eval()
    with torch.no_grad():
        outs = []
        for t in range(T_bins):
            x_t = embedding_matrix[:, t, :].to(device)
            x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1e6, neginf=-1e6)
            x_t = torch.nn.functional.normalize(x_t, p=2, dim=1) * float(args.norm_scale)
            h_t = model(edge_index, x=x_t)              # [N, C]
            h_t = torch.nan_to_num(h_t, nan=0.0, posinf=1e6, neginf=-1e6)
            outs.append(h_t)
        X_eval = torch.stack(outs, dim=1)               # [N, T, C]

        # --- דיבוג חד-פעמי לפני הרצת הקשב ב-eval ---
        dbg_seq_temporal_layer_once(model.seq_model, X_eval)

        att_output = model.seq_model(X_eval)            # [N, T, C] או [N, C]
        if att_output.ndim == 2:
            att_output = att_output.unsqueeze(1)        # [N, 1, C]

        # דיבוג: מצב הפלט הטמפורלי
        dbg_stats("att_output eval", att_output, per_time=True)

    # -----------------------------------------------------------
    # שלב 5: IF+LOF טמפורלי (ללא שינוי לוגיקה)
    # -----------------------------------------------------------
    N_eval, T_eval, C_eval = att_output.shape
    lof_k = max(2, min(args.lof_n_neighbors, N_eval - 1))
    contam = min(max(args.contamination, max(1, int(0.02 * N_eval)) / N_eval), 0.20)
    print(f"\n📏 Shapes: N={N_eval}, T={T_eval}, C={C_eval}  |  LOF.k={lof_k}, IF.contamination={contam:.4f}")

    AS_if  = np.full((N_eval, T_eval), np.nan, dtype=np.float32)
    AS_lof = np.full((N_eval, T_eval), np.nan, dtype=np.float32)

    valid_t = 0
    for t in range(T_eval):
        X_t = att_output[:, t, :].detach().cpu().numpy()

        if not np.isfinite(X_t).all():
            n_bad = np.size(X_t) - np.isfinite(X_t).sum()
            print(f"[WARN] Found {n_bad} non-finite entries at t={t}. Fixing with zscore guards.")
            # guard סטנדרטי לבדיקות; לא משנה התנהגות בסיסית
            mu = np.nanmean(X_t, axis=0, keepdims=True)
            sigma = np.nanstd(X_t, axis=0, keepdims=True) + 1e-12
            Z = (X_t - mu) / sigma
            Z[~np.isfinite(Z)] = 0.0
            X_t = Z

        # אם כל העמודות קבועות → אין מידע
        if (np.std(X_t, axis=0) <= 1e-12).all():
            print(f"[INFO] t={t}: not enough informative features after filtering (C'=0). Skipping.")
            continue

        scaler_t = MinMaxScaler()
        X_t_scaled = scaler_t.fit_transform(X_t)

        if_clf_t = IsolationForest(
            n_estimators=100,
            contamination=contam,
            random_state=args.random_state,
            n_jobs=-1
        )
        if_clf_t.fit(X_t_scaled)
        AS_if[:, t] = -if_clf_t.decision_function(X_t_scaled)

        lof_t = LocalOutlierFactor(
            n_neighbors=lof_k,
            contamination=contam,
            novelty=False,
            n_jobs=-1
        )
        _ = lof_t.fit_predict(X_t_scaled)
        AS_lof[:, t] = -(lof_t.negative_outlier_factor_)

        valid_t += 1

    if valid_t == 0:
        print("⚠ No valid timesteps for anomaly scoring (all t were constant/invalid).")
    print(f"[INFO] mean valid t per node: {valid_t:.2f} / {T_eval}")

    # פרופילים סטטיסטיים פר-צומת על פני הזמן
    mu_if   = np.nanmean(AS_if,  axis=1)
    std_if  = np.nanstd(AS_if,   axis=1)
    mu_lof  = np.nanmean(AS_lof, axis=1)
    std_lof = np.nanstd(AS_lof,  axis=1)

    # Top-K
    K = max(1, min(args.topk, N_eval))
    _print_topk("IF by mean (μ)",  mu_if,  k=min(20, N_eval))
    _print_topk("IF by std (σ)",   std_if, k=min(20, N_eval))
    _print_topk("LOF by mean (μ)", mu_lof, k=min(20, N_eval))
    _print_topk("LOF by std (σ)",  std_lof, k=min(20, N_eval))

    # קורלציה בין IF ל-LOF (בממוצעים)
    try:
        if np.isfinite(mu_if).any() and np.isfinite(mu_lof).any():
            rho, p = spearmanr(mu_if, mu_lof)
            if np.isfinite(rho):
                print(f"\n🔗 Spearman(IF.mean, LOF.mean) = {rho:.4f}  (p={p:.2e})")
            else:
                print("\n🔗 Spearman skipped: non-finite vectors.")
        else:
            print("\n🔗 Spearman skipped: non-finite vectors.")
    except Exception as e:
        print("Spearman failed:", e)

    # סטטיסטיקות התפלגות
    _summ("IF.mean over nodes",  mu_if)
    _summ("IF.std  over nodes",  std_if)
    _summ("LOF.mean over nodes", mu_lof)
    _summ("LOF.std  over nodes", std_lof)

    # שמירה לקבצים (CSV + היסטוגרמות)
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

    # -----------------------------------------------------------
    # שמירת bundle (אופציונלי)
    # -----------------------------------------------------------
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
