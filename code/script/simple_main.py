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

# מודלים/כלים פנימיים
from models.Dynhat import Dynhat
from script.utils.dynamic_node2vec import load_manifest_and_snapshots, build_dynamic_node2vec
from script.utils.dataUtils import load_citation_data  # מחזיר adj, features_sp, labels, idx_train, idx_val, idx_test


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


if __name__ == "__main__":
    main()
