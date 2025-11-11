# -*- coding: utf-8 -*-
"""
newMain.py
==========
זרימה מלאה:
1) טוען גרפים סנאפשוטים ומטא-דאטה
2) מפיק אמבדינגים דינמיים עם Node2Vec (N, T, F)
3) טוען adjacency + labels ל-PyG
4) בונה Dynhat (בלי לשנות את הקובץ שלו) + ראש סיווג חיצוני Linear(C -> num_classes)
5) מאמן עם CrossEntropy על הלוגיטים (לא על תכונות גאומטריות)
6) ולידציה
7) מחשב ציוני אנומליה אחרי האימון (IF)
"""

import os
import sys
import math
import copy
import json
import ast
import pickle
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from torch_geometric.utils import from_networkx, from_scipy_sparse_matrix

# ========= קונפיג =========
try:
    from config import args  # אצלך בפרויקט
except Exception as e:
    # גיבוי מינימלי אם אין config.args
    class _Obj: pass
    args = _Obj()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.Time_stamps = 10
    args.nfeat = 128
    args.max_epoch = 32
    args.dropout = 0.0
    args.num_walks = 100
    args.workers = 2
    args.heads = 1
    args.temporal_attention_layer_heads = 1
    args.manifold = "Hyperboloid"
    args.nhid = 32
    args.nout = 32
    args.seq_model = "TemporalAttention"
    args.fix_curvature = True
    args.validation_iteration = 0
    args.graph_type = "None"

# ========= ייבוא Dynhat (נתיב גמיש) =========
_dynhat_import_ok = False
for candidate in ["models.Dynhat", "model.Dynhat", "script.models.Dynhat"]:
    try:
        Dynhat = __import__(candidate, fromlist=["Dynhat"]).Dynhat
        _dynhat_import_ok = True
        break
    except Exception:
        pass

if not _dynhat_import_ok:
    raise ImportError("Could not import Dynhat from models.Dynhat/model.Dynhat/script.models.Dynhat")

# ========= ייבוא דאטה =========
try:
    # אצלך: utilis.data_utilis או script.utils.dataUtils (תומך בשניהם)
    try:
        from utilis.data_utilis import load_citation_data
    except Exception:
        from script.utils.dataUtils import load_citation_data
except Exception as e:
    raise ImportError("Cannot import load_citation_data from your project") from e

try:
    from node2vec import Node2Vec
except Exception as e:
    raise ImportError("Missing node2vec package in environment") from e


# ========= עזר =========
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def finite_ratio(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.isfinite(x).sum()) * 100.0 / x.size

def ensure_tensor_cpu_np(x: torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ========= ארגומנטים נוספים מה-CLI (לא חובה) =========
def parse_extra_cli():
    p = argparse.ArgumentParser(description="newMain.py - Dynhat + classifier head")
    p.add_argument("--snapshots_pickle", type=str,
                   default="src/data/generate_custom_output/ind.dblpv13.snapshot_graphs",
                   help="Path to snapshot_graphs pickle")
    p.add_argument("--meta_csv", type=str,
                   default="src/data/final_filtered_by_fos_and_reference.csv",
                   help="Path to metadata CSV")
    p.add_argument("--data_path", type=str,
                   default="src/data/generate_custom_output",
                   help="Path for load_citation_data")
    p.add_argument("--save_dir", type=str,
                   default="src/plots/anomaly_score_plots",
                   help="Where to save plots & summaries")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    return p.parse_args([])  # ברירת מחדל, בלי לשבור קוד אצלך


EX = parse_extra_cli()
SAVE_DIR = Path(EX.save_dir)
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ========= שלב 1: טעינת סנאפשוטים + מטא =========
print("== Load metadata and snapshots ==")
df_meta = pd.read_csv(EX.meta_csv)
with open(EX.snapshots_pickle, "rb") as f:
    snapshot_graphs = pickle.load(f)

# ========= שלב 2: אמבדינגים דינמיים Node2Vec =========
print("== Build dynamic Node2Vec embeddings ==")
T = int(args.Time_stamps)
node2vec_embeddings = {}
for i, year in enumerate(sorted(snapshot_graphs.keys())[:T]):
    G_year = snapshot_graphs[year]

    # Node2Vec ל-פר-גרף
    node2vec = Node2Vec(G_year,
                        dimensions=args.nfeat,
                        walk_length=30,
                        num_walks=args.num_walks,
                        workers=args.workers)
    model_n2v = node2vec.fit(window=10, min_count=1)

    # טנזור [N, F] — מבוסס על אינדקסים של df_meta
    N = len(df_meta)
    emb = torch.zeros((N, args.nfeat), dtype=torch.float32)
    for node in G_year.nodes():
        try:
            emb[node] = torch.tensor(model_n2v.wv[str(node)], dtype=torch.float32)
        except KeyError:
            continue
    node2vec_embeddings[i] = emb

embedding_matrix = torch.stack([node2vec_embeddings[t] for t in range(T)], dim=1)  # [N, T, F]
print(f"✅ embedding_matrix shape: {tuple(embedding_matrix.shape)}")
_EMB = ensure_tensor_cpu_np(embedding_matrix)
print(f"[DBG] embedding_matrix finite%={finite_ratio(_EMB):.2f}%")

# ========= שלב 3: טעינת adjacency + labels =========
print("== Load citation data (adj/features/labels) ==")
adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
    dataset_str="dblpv13", use_feats=True, data_path=EX.data_path
)
edge_index, _ = from_scipy_sparse_matrix(adj)

# מעבירים לטנסורים/דיווייס
features = torch.FloatTensor(features.todense()).to(args.device)
labels = torch.LongTensor(labels).to(args.device)
edge_index = edge_index.to(args.device)
embedding_matrix = embedding_matrix.to(args.device)

# לוודא ש-labels זה אינדקסים (לא one-hot)
if labels.ndim == 2 and labels.shape[1] > 1:
    labels = labels.argmax(dim=1)

args.num_nodes = features.shape[0]
args.num_classes = int(labels.max().item()) + 1

print(f"[DEBUG] labels_np shape: {labels.shape} dtype: {labels.dtype}")
# (לא מחלק לפי TRAIN/VAL כאן; אם תרצי, הדפיסי histogram עם numpy.bincount)

# ========= שלב 4: מודל Dynhat =========
print("== Build Dynhat model ==")
model = Dynhat(args, time_length=T).to(args.device)
print(f"🔧 Dynhat init params: num_nodes={args.num_nodes}, nfeat={args.nfeat}, nclass={args.num_classes}, nout={args.nout}")
print(f"✅ Dynhat ready with nhid={args.nhid} (att heads={args.temporal_attention_layer_heads}, struct heads={args.heads}).")

# ראש סיווג ייבנה דינמית בפעם הראשונה שנדע את C (ממד הפיצ'רים אחרי הקשב)
clf_head = None
params_for_optim = list(model.parameters())  # נוסיף את הראש כשייבנה
optimizer = torch.optim.AdamW(params_for_optim, lr=EX.lr, weight_decay=EX.weight_decay)

# ========= שלב 5: אימון =========
print("== Training ==")
train_losses = []
val_accuracies = []

def _temporal_forward_all_times(m: torch.nn.Module,
                                edge_index: torch.Tensor,
                                emb_mat: torch.Tensor,
                                T_steps: int):
    """ מחזיר X=[N,T,C] על ידי הפעלת Dynhat לכל t ואז seq_model. """
    outs = []
    for t in range(T_steps):
        x_t = emb_mat[:, t, :]
        h_t = m(edge_index, x=x_t)
        outs.append(h_t)
    X = torch.stack(outs, dim=1)  # [N, T, C]
    # אצל Dynhat, שכבת הקשב היא model.seq_model
    X_att = m.seq_model(X)        # [N, T, C] (לרוב שומר ממד)
    return X_att

for epoch in range(int(args.max_epoch)):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # קדימה דרך כל הזמנים + קשב
    X_att = _temporal_forward_all_times(model, edge_index, embedding_matrix, T)  # [N,T,C]
    feat_last = X_att[:, -1, :]  # [N,C] — זמן אחרון (כמו ב-AdiHS)

    # בונים ראש סיווג בפעם הראשונה: C -> num_classes
    if clf_head is None:
        C = int(feat_last.shape[-1])
        num_classes = int(args.num_classes)
        clf_head = torch.nn.Linear(C, num_classes).to(args.device)
        # לעדכן אופטימייזר שיכלול גם את הראש
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(clf_head.parameters()),
            lr=EX.lr, weight_decay=EX.weight_decay
        )

    logits = clf_head(feat_last)  # [N,num_classes]
    loss = F.cross_entropy(logits[idx_train], labels[idx_train])
    loss.backward()

    # יציבות
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(clf_head.parameters()), max_norm=EX.grad_clip)
    optimizer.step()

    # ולידציה
    model.eval()
    with torch.no_grad():
        X_att_val = _temporal_forward_all_times(model, edge_index, embedding_matrix, T)
        feat_last_val = X_att_val[:, -1, :]
        val_logits = clf_head(feat_last_val)
        if len(idx_val) > 0:
            val_pred = val_logits[idx_val].argmax(dim=1)
            acc_val = (val_pred == labels[idx_val]).float().mean().item()
        else:
            acc_val = float("nan")

    train_losses.append(float(loss.item()))
    val_accuracies.append(float(acc_val))
    print(f"[Epoch {epoch}] Train Loss: {loss.item():.4f} | Val Acc: {acc_val:.4f}")

# ========= גרף Overfitting =========
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss", linewidth=2)
plt.plot(val_accuracies, label="Validation Accuracy", linewidth=2)
plt.xlabel("Epoch"); plt.ylabel("Value"); plt.title("Train Loss vs Validation Accuracy")
plt.grid(True); plt.legend()
plot_path = SAVE_DIR / "loss_vs_val_acc.png"
plt.savefig(plot_path); plt.close()
print(f"✅ Saved plot: {plot_path}")

# ========= שלב 6: חישוב פלט קשב מלא לצורך אנומליה =========
print("== Compute attention outputs for anomaly scoring ==")
model.eval()
with torch.no_grad():
    X_att_all = _temporal_forward_all_times(model, edge_index, embedding_matrix, T)  # [N,T,C]

X_np = ensure_tensor_cpu_np(X_att_all)  # [N,T,C]
if not np.isfinite(X_np).any():
    print("⚠️ No finite values in attention output — skipping anomaly scoring.")
    # שמירה מינימלית ויציאה
    pd.DataFrame({"train_loss": train_losses, "val_acc": val_accuracies}).to_csv(SAVE_DIR / "train_val_curve.csv", index=False)
    sys.exit(0)

# ========= שלב 7: אנומליה (IF) =========
print("== Anomaly detection (IsolationForest) on [N, T*C] ==")
N, T_eff, C_eff = X_np.shape
X_flat = X_np.reshape(N, T_eff * C_eff)

# guard מינימלי: החלפת non-finite ב-0
mask_finite = np.isfinite(X_flat)
if not mask_finite.all():
    X_flat = np.where(mask_finite, X_flat, 0.0)

# MinMax
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_flat)

clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
clf.fit(X_scaled)
anomaly_scores = -clf.decision_function(X_scaled)  # גדול=יותר אנומלי
anomaly_ranks = np.argsort(anomaly_scores)[::-1]   # מהכי אנומלי ומטה

# שמירת סיכום
summary_df = pd.DataFrame({
    "node_id": np.arange(N, dtype=int),
    "anomaly_score": anomaly_scores
}).sort_values("anomaly_score", ascending=False)

summary_path = SAVE_DIR / "temporal_anomaly_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"💾 saved: {summary_path}")

# היסטוגרמה ידידותית (Guard no-data)
if np.isfinite(anomaly_scores).any():
    plt.figure(figsize=(8, 4))
    plt.hist(anomaly_scores, bins=50)
    plt.title("IsolationForest anomaly scores (flattened over time)")
    plt.xlabel("score"); plt.ylabel("count"); plt.grid(True)
    fig_path = SAVE_DIR / "hist_anomaly_scores.png"
    plt.savefig(fig_path); plt.close()
    print(f"💾 saved: {fig_path}")
else:
    print("ℹ️ skipped histogram: no finite data.")

print("\nTop-10 anomalous nodes (IF):")
for rank, nid in enumerate(anomaly_ranks[:10], start=1):
    print(f"{rank:2d}. node={nid:5d}  score={anomaly_scores[nid]:.6f}")

# ========= סיום =========
print("\n✅ Done.")
