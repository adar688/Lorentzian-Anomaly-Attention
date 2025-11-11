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
from sklearn.preprocessing import MinMaxScaler  # נשאר ל-Stage-4 (אפשרי)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import spearmanr
import pandas as pd

# ==============================
#    Utilities (NEW)
# ==============================
def dbg_stats(name, X):
    """מדפיס סטטוס finite ושונות עבור טנסור 2D/3D."""
    X = np.asarray(X)
    fin = np.isfinite(X)
    print(f"[DBG] {name}: shape={X.shape}, finite%={fin.mean()*100:.2f}%")
    if X.ndim == 2:  # [N, C]
        row_std = X.std(axis=1)
        col_std = X.std(axis=0)
        print(f"[DBG] {name}: rows std>0%={(row_std>0).mean()*100:.2f}%, cols std>0%={(col_std>0).mean()*100:.2f}%")
    elif X.ndim == 3:  # [N, T, C]
        N,T,C = X.shape
        per_t_fin = [np.isfinite(X[:,t,:]).mean()*100 for t in range(T)]
        per_t_cols_std = [(X[:,t,:].std(axis=0)>0).mean()*100 for t in range(T)]
        print(f"[DBG] {name}: finite% per t: {[round(p,2) for p in per_t_fin]}")
        print(f"[DBG] {name}: %cols std>0 per t: {[round(p,2) for p in per_t_cols_std]}")

def safe_zscore(X, eps=1e-8):
    """Z-Score בטוח פר-עמודה עם ε כדי למנוע חלוקה באפס; NaN/Inf נהפכים ל-0 בסוף."""
    X = np.asarray(X, dtype=float)
    X = np.where(np.isfinite(X), X, np.nan)
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd = np.where(sd < eps, eps, sd)
    Z = (X - mu) / sd
    return np.where(np.isfinite(Z), Z, 0.0)

def drop_constant_cols(X, eps=1e-12):
    """מוריד עמודות עם שונות ~0; מחזיר מטריצה מסוננת ומסכת עמודות."""
    sd = X.std(axis=0)
    keep = sd > eps
    return (X[:, keep], keep)

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
    p.add_argument("--norm-scale", type=float, default=1.0, help="מקדם סקייל אחרי F.normalize (במקום *0.1)")

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
    """
    רשת ביטחון: ממלא דיפולטים אם חסר ארגומנט שה-Dynhat עשוי לבקש.
    לא דורסת ערכים שכבר קיימים.
    """
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
#   כלי עזר לפלטים
# ==============================
def _summ(name, arr):
    arr = np.asarray(arr).ravel()
    q = np.percentile(arr, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    print(f"\n{name}:")
    print(f"  mean={arr.mean():.6f}  std={arr.std():.6f}  min={q[0]:.6f}  p1={q[1]:.6f}  p5={q[2]:.6f}")
    print(f"  p25={q[3]:.6f}  p50={q[4]:.6f}  p75={q[5]:.6f}  p95={q[6]:.6f}  p99={q[7]:.6f}  max={q[8]:.6f}")

def _print_topk(title, scores, k=10):
    k = min(k, len(scores))
    order = np.argsort(-scores)[:k]
    print(f"\n{title} (Top-{k}):")
    for rank, idx in enumerate(order, 1):
        print(f"  {rank:>2}. node={int(idx):>6}  score={float(scores[idx]):.6f}")
    return order

def _save_hist(arr, fname, bins=40, title=None):
    plt.figure(figsize=(8,4.5))
    plt.hist(np.asarray(arr).ravel(), bins=bins)
    if title: plt.title(title)
    plt.xlabel("score"); plt.ylabel("count"); plt.tight_layout()
    plt.savefig(fname); plt.close()
    print(f"💾 saved: {fname}")

# ========================================================
#  Stage 4 unified: validate_with_noise_injection (IF בלבד)
# ========================================================
def validate_with_noise_injection(
    G_original,
    embedding_matrix,           # Tensor [N, T, F] – אמבדינגים קיימים (לא מחשבים מחדש)
    model,                      # Dynhat במצב מאומן
    T,                          # מספר חותמות זמן
    k_percent=0.05,             # יחס רעש מכלל הצמתים
    connect_prob=0.5,           # הסתברות לקשת רעש↔מקוריים
    n_iters=30,                 # מספר איטרציות
    contamination=0.05,         # פרמ' IF
    random_state=42,            # לרפליקציה
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
            base_t = embedding_matrix[:, t, :]  # [N, F]
            if t == t_i:
                feats_t = torch.cat([base_t, noise_tensor], dim=0)  # [N+noise, F]
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
                h_t = model(edge_index_noisy, x=node_features_over_time_noisy[:, t, :])  # [N', C]
                outputs_t.append(h_t)
            X_noisy = torch.stack(outputs_t, dim=1)                 # [N', T, C]
            att_output_noisy = model.seq_model(X_noisy)             # [N', T, C] או [N', C]
            if att_output_noisy.ndim == 2:
                att_output_noisy = att_output_noisy.unsqueeze(1)    # [N', 1, C]
            Np, Tp, Fp = att_output_noisy.shape
            assert Tp == T, "שכבת הרצף צריכה לשמר את ציר הזמן"
            X_flat_noisy = att_output_noisy.reshape(Np, Tp * Fp).detach().cpu().numpy()  # [N', T*C]

        # אפשר להשאיר MinMax כאן — זה שלב ולידציה נפרד
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_flat_noisy)
        clf = IsolationForest(n_estimators=100, contamination=contamination, random_state=random_state)
        clf.fit(X_scaled)
        anomaly_scores = -clf.decision_function(X_scaled)  # גדול → חריג יותר

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
                _ = model(edge_index, x=probe_x)  # צעד בדיקה יחיד
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
    print("✅ embedding_matrix shape:", tuple(embedding_matrix.shape))  # [N, T, F]

    # דיבוג: סטטוס אמבדינגים גולמיים
    dbg_stats("embedding_matrix raw", embedding_matrix.detach().cpu().numpy())

    # קיבוע nfeat & num_nodes לפי אמבדינגים
    args.nfeat = int(embedding_matrix.shape[-1])     # F
    args.num_nodes = int(embedding_matrix.shape[0])  # N

    # -----------------------------------------------------------
    # שלב 2: טעינת גרף/פיצ'רים/לייבלים + ספליטים
    # -----------------------------------------------------------
    adj, features_sp, labels_np, idx_train, idx_val, idx_test = load_citation_data(
        dataset_str="dblpv13",
        use_feats=True,
        data_path=args.data_root
    )

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

    edge_index, _ = from_scipy_sparse_matrix(adj)
    features = torch.FloatTensor(features_sp.todense()).to(device)
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
    # שלב 3: מודל Dynhat + אימון עם ראש סיווג
    # -----------------------------------------------------------
    probe_x = embedding_matrix[:, 0, :].to(device)
    model = _auto_fit_nhid_and_rebuild(Dynhat, args, device, edge_index, probe_x, time_length=T_bins).to(device)

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

                # ניקוי קודם ל-normalize
                x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1e6, neginf=-1e6)
                x_t = torch.nn.functional.normalize(x_t, p=2, dim=1) * float(args.norm_scale)

                h_t = model(edge_index, x=x_t)           # [N, C]
                h_t = torch.nan_to_num(h_t, nan=0.0, posinf=1e6, neginf=-1e6)
                temporal_outputs.append(h_t)

                # דיבוג חד-פעמי לפריים ראשון
                if epoch == 0 and t == 0:
                    dbg_stats("x_t after normalize (*norm-scale)", x_t.detach().cpu().numpy())
                    dbg_stats("h_t from Dynhat (t=0)", h_t.detach().cpu().numpy())

            X = torch.stack(temporal_outputs, dim=1)     # [N, T, C]
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
            optimizer.step()

            # דפוס גראד
            total_norm = 0.0
            for p in list(model.parameters()) + list(cls_head.parameters()):
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item()

            train_losses.append(loss.item())
            print(f"[Epoch {epoch}] Train Loss: {loss.item():.4f} | grad_norm={total_norm:.6f}")
    elif not single_class_problem and args.max_epoch is None:
        print("ℹ Training skipped because --max-epoch=None (set a value to train).")
    else:
        print("⚠ Detected a single class in labels. Skipping supervised cross-entropy training.")
        model.eval(); cls_head.eval()

    # -----------------------------------------------------------
    # === חישוב att_output (Eval) עבור IF/LOF ===
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
        att_output = model.seq_model(X_eval)            # [N, T, C] או [N, C]
        if att_output.ndim == 2:
            att_output = att_output.unsqueeze(1)        # [N, 1, C]

    # דיבוג: סטטוס att_output
    dbg_stats("att_output eval", att_output.detach().cpu().numpy())

    # -----------------------------------------------------------
    # שלב 5: IF+LOF טמפורלי (רובוסטי)
    # -----------------------------------------------------------
    N_eval, T_eval, C_eval = att_output.shape
    lof_k = max(2, min(args.lof_n_neighbors, N_eval - 1))
    contam = min(max(args.contamination, max(1, int(0.02 * N_eval)) / N_eval), 0.20)
    print(f"\n📏 Shapes: N={N_eval}, T={T_eval}, C={C_eval}  |  LOF.k={lof_k}, IF.contamination={contam:.4f}")

    AS_if  = np.full((N_eval, T_eval), np.nan, dtype=np.float32)  # NaN כדי שלא יזהם ממוצעים
    AS_lof = np.full((N_eval, T_eval), np.nan, dtype=np.float32)

    n_valid_t = 0
    for t in range(T_eval):
        X_t = att_output[:, t, :].detach().cpu().numpy()

        # ניקוי ראשוני
        if not np.isfinite(X_t).all():
            n_bad = np.size(X_t) - np.isfinite(X_t).sum()
            print(f"[WARN] Found {n_bad} non-finite entries at t={t}. Fixing with zscore guards.")
        # Zscore בטוח + הסרת עמודות קבועות
        X_t = safe_zscore(X_t)
        X_t, keep = drop_constant_cols(X_t)

        if X_t.shape[1] < 2:
            print(f"[INFO] t={t}: not enough informative features after filtering (C'={X_t.shape[1]}). Skipping.")
            continue

        # IF
        if_clf_t = IsolationForest(
            n_estimators=100,
            contamination=contam,
            random_state=args.random_state,
            n_jobs=-1
        )
        s_if = -if_clf_t.fit(X_t).decision_function(X_t)
        rng_if = s_if.max() - s_if.min()
        if rng_if > 1e-12:
            s_if = (s_if - s_if.min()) / rng_if
        else:
            s_if = np.full_like(s_if, np.nan)

        # LOF
        k_here = min(lof_k, max(2, X_t.shape[0]-1))
        lof_t = LocalOutlierFactor(
            n_neighbors=k_here,
            contamination=contam,
            novelty=False,
            n_jobs=-1
        )
        _ = lof_t.fit_predict(X_t)
        s_lof = -(lof_t.negative_outlier_factor_)
        rng_lof = s_lof.max() - s_lof.min()
        if rng_lof > 1e-12:
            s_lof = (s_lof - s_lof.min()) / rng_lof
        else:
            s_lof = np.full_like(s_lof, np.nan)

        AS_if[:, t]  = s_if
        AS_lof[:, t] = s_lof
        n_valid_t += 1

    if n_valid_t == 0:
        print("⚠ No valid timesteps for anomaly scoring (all t were constant/invalid).")
    else:
        print(f"✅ Valid timesteps for anomaly scoring: {n_valid_t}/{T_eval}")

    # פרופילים סטטיסטיים פר-צומת (להתעלם מ-NaN)
    mu_if   = np.nanmean(AS_if,  axis=1)
    std_if  = np.nanstd( AS_if,  axis=1)
    mu_lof  = np.nanmean(AS_lof, axis=1)
    std_lof = np.nanstd( AS_lof, axis=1)

    # כמה t תקפים לכל צומת
    valid_counts = np.sum(np.isfinite(AS_if) | np.isfinite(AS_lof), axis=1)
    print(f"[INFO] mean valid t per node: {valid_counts.mean():.2f} / {T_eval}")

    # Top-K
    K = max(1, min(args.topk, N_eval))
    _print_topk("IF by mean (μ)",  mu_if,  k=min(20, N_eval))
    _print_topk("IF by std (σ)",   std_if, k=min(20, N_eval))
    _print_topk("LOF by mean (μ)", mu_lof, k=min(20, N_eval))
    _print_topk("LOF by std (σ)",  std_lof, k=min(20, N_eval))

    # קורלציה בין IF ל-LOF (בממוצעים) — רק אם יש שונות
    try:
        if np.all(~np.isfinite(mu_if)) or np.all(~np.isfinite(mu_lof)):
            print("\n🔗 Spearman skipped: non-finite vectors.")
        elif (np.nanstd(mu_if) < 1e-12) or (np.nanstd(mu_lof) < 1e-12):
            print("\n🔗 Spearman skipped: constant vectors.")
        else:
            rho, p = spearmanr(mu_if, mu_lof, nan_policy='omit')
            print(f"\n🔗 Spearman(IF.mean, LOF.mean) = {rho:.4f}  (p={p:.2e})")
    except Exception as e:
        print("Spearman failed:", e)

    # סטטיסטיקות התפלגות
    _summ("IF.mean over nodes",  np.nan_to_num(mu_if, nan=0.0))
    _summ("IF.std  over nodes",  np.nan_to_num(std_if, nan=0.0))
    _summ("LOF.mean over nodes", np.nan_to_num(mu_lof, nan=0.0))
    _summ("LOF.std  over nodes", np.nan_to_num(std_lof, nan=0.0))

    # שמירה לקבצים (CSV + היסטוגרמות)
    df_summary = pd.DataFrame({
        "mu_if":  mu_if,  "std_if": std_if,
        "mu_lof": mu_lof, "std_lof": std_lof,
        "valid_t": valid_counts
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
