# -*- coding: utf-8 -*-

""" מריצים את הקובץ הזה לאחר שטענו את הדאטה סט ועשינו לו עיבוד
כאן כבר מתחילים לעבוד איתו
השלב הנוכחי כרגע הוא שטוענים את הקבצים שעיבדנו ומעבירים אותם שלב של NODE2VEC"""


import os
import sys
import argparse
import torch
import numpy as np
from torch_geometric.utils import from_scipy_sparse_matrix
from model.Dynhat import Dynhat

from script.utils.dynamic_node2vec import (
    load_manifest_and_snapshots,   # מצפה ל-manifest.json + snapshots.npz
    build_dynamic_node2vec         # בונה Tensor [N, T, F]
)
from script.utils.dataUtils import load_citation_data  # המימוש המינימלי שכתבנו

def parse_args():
    p = argparse.ArgumentParser(description="Unified main: Dynamic Node2Vec + data loading.")
    # נתיבי קלט
    p.add_argument("--data-root", type=str, default="src/data/custom_out",
                   help="תיקייה עם קבצי הפלט של prepareData (manifest.json, snapshots.npz, graph.json, ...)")
    # פרמטרי Node2Vec
    p.add_argument("--emb-dim", type=int, default=128, help="F: ממד האמבדינג")
    p.add_argument("--walk-length", type=int, default=30, help="Node2Vec: אורך הליכה")
    p.add_argument("--num-walks", type=int, default=10, help="Node2Vec: מספר הליכות לצומת")
    p.add_argument("--workers", type=int, default=4, help="Node2Vec: מספר תהליכי רקע")
    p.add_argument("--window", type=int, default=10, help="Word2Vec window")
    p.add_argument("--t-max", type=int, default=None,
                   help="אופציונלי: שימוש רק ב-T הראשונים (לבדיקות/קיצור)")
    # מכשיר ושמירה
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                   help="cuda / cpu")
    p.add_argument("--save-bundle", type=str, default="",
                   help="נתיב לשמירת חבילה אחת בסוף (embedding_matrix + graph tensors). ריק = לא שומר.")
    return p.parse_args()

def main():
    # תמיכה ב-UTF-8 למסופים מסוימים
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args()
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
    )  # torch.Tensor [N, T, F] על CPU כברירת מחדל (בהתאם למימוש שלך)


    print (embedding_matrix)
    # -----------------------------------------------------------
    # שלב 2: טעינת גרף/פיצ'רים/לייבלים + ספליטים (פורמט custom_out)
    # -----------------------------------------------------------
    # --- שלב 2: טעינת גרף/פיצ'רים/לייבלים (פורמט custom_out) ---
    adj, features_sp, labels_np, idx_train, idx_val, idx_test = load_citation_data(
        dataset_str="dblpv13",   # נשאר לחתימה
        use_feats=True,          # אם תרצי מטריצת זהות: False
        data_path=cli.data_root
    )

    # === ההמרות הקצרות בדיוק כמו בקוד המקורי ===
    edge_index, _ = from_scipy_sparse_matrix(adj)
    features = torch.FloatTensor(features_sp.todense()).to(args.device)
    labels = torch.LongTensor(labels_np).to(args.device)
    edge_index = edge_index.to(args.device)

    args.num_nodes = features.shape[0]
    args.num_classes = labels.max().item() + 1
    model = Dynhat(args, time_length=T).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # === מעקב אימון + יישור labels לפורמט הנדרש ===
    train_losses: list[float] = []

    # אם labels הגיעו בטעות כ-one-hot ([N, C]) → ממירים לאינדקסים [N]
    if labels.dim() == 2 and labels.size(1) > 1:
        labels = labels.argmax(1)

    # הבטחת טיפוס/דבייס (בדרך כלל כבר תקין אצלך, שמרנו את זה מינימלי)
    if labels.dtype is not torch.long:
        labels = labels.long()
    if labels.device.type != torch.device(args.device).type:
        labels = labels.to(args.device)

    train_losses: list[float] = []

    for epoch in range(args.max_epoch):
        model.train()
        optimizer.zero_grad()

        # 1) Forward על כל ה-T טיימסטמפים
        temporal_outputs = []
        for t in range(T):
            x_t = embedding_matrix[:, t, :].to(args.device)  # [N, F]
            h_t = model(edge_index, x=x_t)                   # [N, C] (לוגיטים/ייצוג)
            temporal_outputs.append(h_t)

        # 2) ערימה + שכבת קשב → בוחרים את הזמן האחרון
        X = torch.stack(temporal_outputs, dim=1)             # [N, T, C]
        att = model.ddy_attention_layer(X)                   # לרוב [N, T, C]
        logits = att[:, -1, :]                               # [N, C]

        # 3) Loss + עדכון משקולות (train בלבד)
        loss = F.cross_entropy(logits[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        print(f"[Epoch {epoch}] Train Loss: {loss.item():.4f}")
    


    

