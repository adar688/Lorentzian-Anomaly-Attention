# -*- coding: utf-8 -*-

""" מריצים את הקובץ הזה לאחר שטענו את הדאטה סט ועשינו לו עיבוד
כאן כבר מתחילים לעבוד איתו
השלב הנוכחי כרגע הוא שטוענים את הקבצים שעיבדנו ומעבירים אותם שלב של NODE2VEC"""

import os
import sys
import argparse
import copy
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

# === ייבוא למודולי אנומליות והשוואה ===
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import spearmanr
import pandas as pd
import re


# ==============================
#    פרמטרים וארגומנטים
# ==============================
def parse_args():
    p = argparse.ArgumentParser(description="Unified main: Dynamic Node2Vec + data loading.")
    # נתיבי קלט
    p.add_argument("--data-root", type=str, default="script/data/custom_out",
                   help="תיקייה עם קבצי הפלט של prepareData (manifest.json, snapshots.npz, graph.json, ...)")
    # פרמטרי Node2Vec
    p.add_argument("--emb-dim", type=int, default=128, help="F: ממד האמבדינג")
    p.add_argument("--walk-length", type=int, default=30, help="Node2Vec: אורך הליכה")
    p.add_argument("--num-walks", type=int, default=10, help="Node2Vec: מספר הליכות לצומת")
    p.add_argument("--workers", type=int, default=4, help="Node2Vec: מספר תהליכי רקע")
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
    p.add_argument("--nhid", type=int, default=64, help="גודל השכבה החבויה (hidden size)")
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
    p.add_argument("--nfeat", type=int, default=None,
                   help="מספר הפיצ'רים לקלט המודל (ברירת מחדל: ייגזר מ-embedding_matrix)")
    p.add_argument("--nout", type=int, default=None,
                   help="מספר יחידות פלט של המודל (ברירת מחדל: יוגדר לפי num_classes)")
    p.add_argument("--seq-model", dest="seq_model", type=str, default="gru",
                   choices=["gru", "lstm", "transformer", "none"],
                   help="מודל רצף טמפורלי בתוך Dynhat")
    p.add_argument("--seq-hidden", dest="seq_hidden", type=int, default=128,
                   help="גודל החבוי של מודל הרצף")
    p.add_argument("--seq-layers", dest="seq_layers", type=int, default=1,
                   help="מספר שכבות במודל הרצף")
    p.add_argument("--seq-dropout", dest="seq_dropout", type=float, default=0.1,
                   help="Dropout בתוך מודל הרצף")

    # אימון
    p.add_argument("--max-epoch", type=int, default=50, help="מספר אפוקים לאימון Dynhat")
    p.add_argument("--lr", type=float, default=1e-2, help="למידה - Adam LR")
    p.add_argument("--weight-decay", type=float, default=5e-4, help="למידה - Adam weight decay")

    # מכשיר ושמירה
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                   help="cuda / cpu")
    p.add_argument("--save-bundle", type=str, default="",
                   help="נתיב לשמירת חבילה אחת בסוף (embedding_matrix + graph tensors). ריק = לא שומר.")

    # אנומליות
    p.add_argument("--contamination", type=float, default=0.05, help="אחוז אנומליות משוער ל-IF/LOF")
    p.add_argument("--lof-n-neighbors", type=int, default=20, help="k של LOF")
    p.add_argument("--topk", type=int, default=20, help="Top-K להצגה/ניתוח (לא חובה)")

    # רעש (Stage 4)
    p.add_argument("--noise-percent", type=float, default=0.05, help="k% צמתי רעש מכלל הצמתים")
    p.add_argument("--noise-connect-prob", type=float, default=0.5, help="הסתברות חיבור רעש↔מקוריים")
    p.add_argument("--noise-iters", type=int, default=30, help="מספר איטרציות ולידציה עם רעש")
    p.add_argument("--random-state", type=int, default=42, help="זרע רנדומי לשחזוריות")

    return p.parse_args()


def _ensure_dynhat_defaults(args):
    """
    רשת ביטחון: ממלא דיפולטים אם חסר ארגומנט שה- Dynhat עשוי לבקש.
    לא דורסת ערכים שכבר קיימים.
    """
    defaults = {
        "nhid": 64,
        "dropout": 0.5,
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
        # ערך c לארגומנטים, נשתמש בו גם כ-fallback למחלקה אם צריך
        args.c = float(getattr(args, "curvature", getattr(args, "c0", 1.0)))

    return args


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
        # 1) זמן הזרקה אקראי + כמות רעש לפי k%
        t_i = int(rng.integers(low=0, high=T))
        n_noise_nodes = max(1, int(round(k_percent * N)))

        # 2) גרף רועש: הוספת צמתי רעש + קשתות אקראיות
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

        # 3) פיצ'רים: רעש “מופיע” רק ב-t_i (בשאר הזמנים 0)
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

        # 4) edge_index_noisy מ-NX
        G_noisy_simple = nx.DiGraph()
        G_noisy_simple.add_nodes_from(G_noisy.nodes())
        G_noisy_simple.add_edges_from(G_noisy.edges())
        edge_index_noisy = from_networkx(G_noisy_simple).edge_index.to(device)

        # 5) Forward לאורך T → קשב → השטחה
        with torch.no_grad():
            outputs_t = []
            for t in range(T):
                h_t = model(edge_index_noisy, x=node_features_over_time_noisy[:, t, :])  # [N', F']
                outputs_t.append(h_t)
            X_noisy = torch.stack(outputs_t, dim=1)                 # [N', T, F']
            att_output_noisy = model.ddy_attention_layer(X_noisy)   # רצוי [N', T, F_att]
            if att_output_noisy.ndim == 2:
                att_output_noisy = att_output_noisy.unsqueeze(1)    # [N', 1, F_att]
            Np, Tp, Fp = att_output_noisy.shape
            assert Tp == T, "שכבת הקשב צריכה לשמר את ציר הזמן (או החזרה טמפורלית עקבית)"
            X_flat_noisy = att_output_noisy.reshape(Np, Tp * Fp).detach().cpu().numpy()  # [N', T*F_att]

        # 6) IF על וקטור משוטח (ציון יחיד לצומת)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_flat_noisy)
        clf = IsolationForest(n_estimators=100, contamination=contamination, random_state=random_state)
        clf.fit(X_scaled)
        anomaly_scores = -clf.decision_function(X_scaled)  # גדול → חריג יותר

        # פיצול ציונים
        real_scores = anomaly_scores[:original_N]
        fake_scores = anomaly_scores[original_N:]

        # 7) סף על בסיס המקוריים (אחוזון 95) → TPR/FPR
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
        "fpr_mean": float(np.nanmean(fpr_list)) if len(fpr_list) else float("nan"),
        "fpr_std": float(np.nanstd(fpr_list)) if len(fpr_list) else float("nan"),
    }
    return {"summary": summary, "results_per_iter": results_per_iter}


def plot_mean_std(mu_if: np.ndarray, std_if: np.ndarray, top_k: int = 10):
    N = len(mu_if)
    x = np.arange(N)

    plt.figure(figsize=(11, 6))
    plt.bar(x - 0.2, mu_if, width=0.4, color='skyblue', alpha=0.7, label='Mean (μ)')
    plt.bar(x + 0.2, std_if, width=0.4, color='orange', alpha=0.7, label='Std Dev (σ)')

    top_mean_idx = np.argsort(-mu_if)[:top_k]
    plt.scatter(top_mean_idx, mu_if[top_mean_idx], color='red', s=80, label=f'Top {top_k} Mean')

    top_std_idx = np.argsort(-std_if)[:top_k]
    plt.scatter(top_std_idx, std_if[top_std_idx], color='purple', s=80, label=f'Top {top_k} Std')

    plt.title('Isolation Forest — Mean & Std per Node', fontsize=14)
    plt.xlabel('Node index (i)', fontsize=12)
    plt.ylabel('Score value', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()




def _auto_fit_nhid_and_rebuild(model_cls, args, device, edge_index, probe_x, time_length, max_tries=2):
    """
    מנסה להריץ צעד forward אחד. אם מתקבלת שגיאת כפל מטריצות (mobius_matvec),
    מפענח את הממדים מהשגיאה, מתאים args.nhid = U - 1, ובונה מודל מחדש.
    """
    for _ in range(max_tries):
        model = model_cls(args, time_length=time_length).to(device)
        try:
            with torch.no_grad():
                _ = model(edge_index, x=probe_x)  # צעד בדיקה יחיד
            return model  # הצליח, אין התאמות נדרשות
        except RuntimeError as e:
            msg = str(e)
            # מחפשים תבנית כמו: "mat1 and mat2 shapes cannot be multiplied (5x2 and 65x65)"
            m = re.search(r"mat1 and mat2 shapes cannot be multiplied \(\d+x(\d+) and (\d+)x(\d+)\)", msg)
            if m is None:
                raise  # שגיאה אחרת – מעבירים הלאה
            u_dim = int(m.group(1))  # ה-U מהצורה (N x U)
            w_in = int(m.group(2))   # ה-in_features של המשקל (W_in x W_out או ההפך, תלוי במימוש)
            # ב-HGCN לרוב הקלט הוא (nhid + 1). אם אנחנו רואים U=2, צריך nhid=1, וכו'.
            suggested_nhid = max(1, u_dim - 1)
            if getattr(args, "nhid", None) == suggested_nhid:
                # כבר ניסינו את הערך הזה – אין טעם לחזור
                raise
            print(f"🔧 Adjusting args.nhid from {getattr(args,'nhid',None)} to {suggested_nhid} based on probe (U={u_dim}).")
            args.nhid = suggested_nhid
            # עדכון נגזר: לעיתים יש שכבות שמוגדרות על בסיס nhid+1, אין עוד מה לעשות כאן – נבנה מחדש בלופ
            continue
    # אם לא הצליח בתוך max_tries
    raise RuntimeError("Failed to auto-fit nhid after probe attempts.")


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

    # נקבע תמיד את nfeat & num_nodes לפי ה-embedding_matrix כדי למנוע None
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

    edge_index, _ = from_scipy_sparse_matrix(adj)
    features = torch.FloatTensor(features_sp.todense()).to(device)
    labels = torch.LongTensor(labels_np).to(device)
    edge_index = edge_index.to(device)

    # לייבלים – ודא אינדקסים (לא one-hot)
    if labels.dim() == 2 and labels.size(1) > 1:
        labels = labels.argmax(1)
    if labels.dtype is not torch.long:
        labels = labels.long()

    # num_classes / nclass לטובת כל מימוש
    args.num_classes = int(labels.max().item() + 1)
    if not hasattr(args, "nclass"):
        args.nclass = args.num_classes

    # לקבע nout (מס' יחידות פלט במודל) לפי מספר המחלקות
    if getattr(args, "nout", None) in (None, 0):
        args.nout = args.num_classes

    # אם יש רק מחלקה אחת – אין טעם ב-CrossEntropy (נמשיך בלי אימון מונחה)
    single_class_problem = args.num_classes < 2

    print(f"🔧 Dynhat init params: num_nodes={args.num_nodes}, nfeat={args.nfeat}, nclass={args.nclass}, nout={args.nout}")

    # ---- Fallback קריטי: לספק ערך עקמומיות ברמת המחלקה אם __init__ משתמש ב-self.c לפני שהוגדר ----
    # (Python מחפש קודם באטריביוטי האובייקט ואז במחלקה. זה מונע AttributeError בלי לגעת ב-Dynhat.py)
    Dynhat.c = float(getattr(args, "c", getattr(args, "curvature", getattr(args, "c0", 1.0))))

    # -----------------------------------------------------------
    # שלב 3: מודל Dynhat + אימון (מדלגים על אימון אם יש רק מחלקה אחת)
    # -----------------------------------------------------------
   # נבדוק מראש ממד בפועל באמצעות פריים יחיד כדי לכוון את nhid אם צריך
    probe_x = embedding_matrix[:, 0, :].to(device)  # פריים זמן בודד כבדיקה
    model = _auto_fit_nhid_and_rebuild(Dynhat, args, device, edge_index, probe_x, time_length=T_bins)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"✅ Dynhat ready with nhid={args.nhid} (agg_feat_size={args.nhid + 1}).")

    train_losses: list[float] = []

    if not single_class_problem:
        for epoch in range(args.max_epoch):
            model.train()
            optimizer.zero_grad()

            # 1) Forward על כל ה-T טיימסטמפים
            temporal_outputs = []
            for t in range(T_bins):
                x_t = embedding_matrix[:, t, :].to(device)  # [N, F]
                h_t = model(edge_index, x=x_t)              # [N, C]
                temporal_outputs.append(h_t)

            # 2) ערימה + שכבת קשב → בחירת זמן אחרון ללוס
            X = torch.stack(temporal_outputs, dim=1)        # [N, T, C]
            att = model.seq_model(X)              # [N, T, C] או [N, C]
            logits = att[:, -1, :] if att.ndim == 3 else att  # [N, C]

            # 3) Loss + עדכון משקולות
            loss = F.cross_entropy(logits[idx_train], labels[idx_train])
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            print(f"[Epoch {epoch}] Train Loss: {loss.item():.4f}")
    else:
        print("⚠ Detected a single class in labels. Skipping supervised cross-entropy training.")
        model.eval()

    # -----------------------------------------------------------
    # === חישוב att_output במצב eval כדי שישמש ל-IF + LOF שלך ===
    # -----------------------------------------------------------
    model.eval()
    with torch.no_grad():
        outs = []
        for t in range(T_bins):
            x_t = embedding_matrix[:, t, :].to(device)
            h_t = model(edge_index, x=x_t)              # [N, C]
            outs.append(h_t)
        X_eval = torch.stack(outs, dim=1)               # [N, T, C]
        att_output = model.seq_model(X_eval)  # [N, T, C] או [N, C]
        if att_output.ndim == 2:
            att_output = att_output.unsqueeze(1)        # [N, 1, C]

    # -----------------------------------------------------------
    # שלב 5: IF+LOF על וקטור מאוחד (שיטוח [T,C] לכל צומת)
    # -----------------------------------------------------------
    N_eval, T_eval = att_output.shape[0], att_output.shape[1]  # [N, T, C]
    C_eval = att_output.shape[2]
    assert att_output.ndim == 3, "att_output חייב להיות [N, T, C] לניתוח טמפורלי פר-חותמת זמן"

    AS_if  = np.zeros((N_eval, T_eval), dtype=np.float32)  # AS_i^IF(t)
    AS_lof = np.zeros((N_eval, T_eval), dtype=np.float32)  # AS_i^LOF(t)

    contam = args.contamination
    lof_k  = args.lof_n_neighbors

    for t in range(T_eval):
        X_t = att_output[:, t, :].detach().cpu().numpy()

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
        AS_lof[:, t]  = -(lof_t.negative_outlier_factor_)  # גדול = יותר חריג

    mu_if  = AS_if.mean(axis=1)
    std_if = AS_if.std(axis=1, ddof=0)
    mu_lof  = AS_lof.mean(axis=1)
    std_lof = AS_lof.std(axis=1, ddof=0)

    K = max(1, min(args.topk, N_eval))
    top_if_idx  = np.argsort(-mu_if)[:K]
    top_lof_idx = np.argsort(-mu_lof)[:K]

    print("✅ Temporal IF/LOF computed per time step.")
    print("Top IF (by mean over time):", top_if_idx.tolist())
    print("Top LOF (by mean over time):", top_lof_idx.tolist())

    temporal_anomaly_package = {
        "AS_if": AS_if,        # shape [N, T]
        "AS_lof": AS_lof,      # shape [N, T]
        "mu_if": mu_if,        # shape [N]
        "std_if": std_if,      # shape [N]
        "mu_lof": mu_lof,      # shape [N]
        "std_lof": std_lof,    # shape [N]
    }

    plot_mean_std(mu_if, std_if, top_k=20)
    plot_mean_std(mu_lof, std_lof, top_k=20)

    # -----------------------------------------------------------
    # שלב 6: Stage 4 — Noise Injection Validation (TPR/FPR) אחרי IF+LOF
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
        contamination=args.contamination,
        random_state=args.random_state,
    )

    print("=== Stage-4 Validation Summary ===")
    print(noise_res["summary"])

    # -----------------------------------------------------------
    # אופציונלי: שמירת bundle
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
