######################## Adar & Shoval ##################################


# -*- coding: utf-8 -*-
"""
Dynamic citation dataset builder (ריפקטור מתועד).
-------------------------------------------------
תפקיד הסקריפט:
1) לקרוא CSV של מאמרים (id, year, abstract, fos, references).
2) למפות מזהי מאמרים לאינדקסים פנימיים (node_idx).
3) לחלק את הדאטה ל-train/val/test לפי סדר כרונולוגי (ספירת מאמרים לפי שנה).
4) להפיק מטריצת פיצ'רים (Bag-of-Words על abstract).
5) לקודד תוויות (fos.name → one-vs-all).
6) לבנות רשימת שכנויות (adjacency list) מתוך references (ציטוטים).
7) לבנות סנאפשוטים דינמיים (גרף לכל time-bin) לפי חלוקה אחידה על ציר הזמן.
8) לשמור את הכל בפורמט נוח: NPZ (מטריצות) + JSON (אינדקסים/מטא).

הערות:
- זה מימוש חדש (לא העתקה) שמייצר אותם ארטיפקטים לוגיים בפורמט שונה ונוח.
- ניתן להרחיב/להחליף רכיבים (למשל TF-IDF במקום CountVectorizer) בלי לגעת בשאר הפייפליין.
"""

from __future__ import annotations

import os
import sys
import json
import ast
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer

# --- פרמטרי קונפיג חיצוניים (אם יש) ---
try:
    # אם יש config.args חיצוני — נשתמש בו לטיימסטמפס
    from config import args
    TIME_STAMPS = args.Time_stamps
except Exception:
    TIME_STAMPS = 12  # ברירת מחדל אם אין args

# תמיכה בהדפסה ב-Unicode במסופים מסוימים
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- נתיבי קלט/פלט ---
INPUT_CSV = Path("script/data/final_filtered_by_fos_and_reference.csv")
OUTPUT_DIR = Path("script/data/custom_out")  # תיקיית יעד (ניתן לשנות)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ===========================
#           HELPERS
# ===========================

def load_dataframe(csv_path: Path) -> pd.DataFrame:
    """
    טוען DataFrame מה-CSV ומוודא שקיימות עמודות בסיס הכרחיות.

    Parameters
    ----------
    csv_path : Path
        נתיב לקובץ ה-CSV.

    Returns
    -------
    pd.DataFrame
        טבלת המאמרים הגולמית.

    Raises
    ------
    FileNotFoundError
        אם הקובץ לא קיים.
    ValueError
        אם חסרות עמודות 'id' או 'year'.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "id" not in df.columns or "year" not in df.columns:
        raise ValueError("CSV must include 'id' and 'year' columns.")
    return df


def build_id_index(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    בונה מיפוי מזהי מאמרים (מחרוזות) לאינדקסים פנימיים עוקבים, ומוסיף עמודת node_idx.

    Parameters
    ----------
    df : pd.DataFrame
        טבלת המאמרים (נדרש לכלול עמודת 'id').

    Returns
    -------
    df_out : pd.DataFrame
        עותק של df עם עמודת 'node_idx'.
    id2idx : Dict[str, int]
        מיפוי 'paper_id' (str) -> אינדקס (int) לשימוש פנימי.
    """
    paper_ids = df["id"].astype(str).tolist()
    id2idx = {pid: i for i, pid in enumerate(paper_ids)}
    df_out = df.copy()
    df_out["node_idx"] = df_out["id"].astype(str).map(id2idx)
    return df_out, id2idx


def split_by_year_counts(df: pd.DataFrame, train_ratio=0.6, val_ratio=0.2):
    """
    מחלק ל-train/val/test לפי מיון כרונולוגי (שנים) ואז חתך 60/20/20 בהיקף מאמרים.

    Parameters
    ----------
    df : pd.DataFrame
        חייב לכלול 'year' ו-'node_idx'.
    train_ratio : float
        חלק ל-train.
    val_ratio : float
        חלק ל-val מתוך הכלל.

    Returns
    -------
    (train_idx, val_idx, test_idx) : Tuple[List[int], List[int], List[int]]
        שלושה וקטורים של אינדקסי צמתים לכל חלוקה.
    """
    df_sorted = df.sort_values(by="year", ascending=True).reset_index(drop=True)
    total = len(df_sorted)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_idx = df_sorted.iloc[:train_end]["node_idx"].tolist()
    val_idx = df_sorted.iloc[train_end:val_end]["node_idx"].tolist()
    test_idx = df_sorted.iloc[val_end:]["node_idx"].tolist()
    return train_idx, val_idx, test_idx


def build_bow_features(df: pd.DataFrame):
    """
    מפיק מטריצת Bag-of-Words על טקסט התקציר (abstract).

    Parameters
    ----------
    df : pd.DataFrame
        חייב לכלול עמודה 'abstract' (אם חסר — יוחלף בריק).

    Returns
    -------
    X : scipy.sparse.csr_matrix
        מטריצת BoW ספרסה בגודל [N, |V|].
    vectorizer : CountVectorizer
        הוקטורייזר המאומן (כולל vocabulary) — נשמר למטא.
    """
    abstracts = df.get("abstract", pd.Series([], dtype=str)).fillna("")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(abstracts)
    return X, vectorizer


def encode_labels(df: pd.DataFrame):
    """
    מקודד תוויות טקסטואליות ('fos.name') לייצוג one-vs-all באמצעות LabelBinarizer.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    Y : np.ndarray
        מטריצת תוויות צפופה [N, C].
    lb : LabelBinarizer
        המקודד המאומן (classes_ במטא).
    """
    fos = df.get("fos.name", pd.Series([], dtype=str)).fillna("")
    lb = LabelBinarizer()
    Y = lb.fit_transform(fos)
    return Y, lb


def safe_parse_list(s: str) -> List[str]:
    """
    ממיר מחרוזת המייצגת רשימה (למשל '["id1","id2"]') לרשימת מחרוזות.
    משתמש ב-ast.literal_eval בבטיחות יחסית. במקרה כשל — מחזיר [].

    Parameters
    ----------
    s : str

    Returns
    -------
    List[str]
    """
    if not isinstance(s, str):
        return []
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        return []
    return []


def build_adjacency(df: pd.DataFrame, id2idx: Dict[str, int]) -> Dict[int, List[int]]:
    """
    בונה adjacency list (רשימת שכנויות) לפי עמודת 'references'.

    Parameters
    ----------
    df : pd.DataFrame
        חייב לכלול 'node_idx' ו-'references'.
    id2idx : Dict[str, int]
        מיפוי מזהה מאמר -> אינדקס.

    Returns
    -------
    Dict[int, List[int]]
        מילון: node_idx -> רשימת node_idx של שכנים שמצוטטים על ידו.
    """
    adj: Dict[int, List[int]] = {}
    for _, row in df.iterrows():
        u = int(row["node_idx"])
        refs = safe_parse_list(row.get("references", "[]"))
        neigh = [id2idx[r] for r in refs if r in id2idx]
        adj[u] = neigh
    return adj


def time_binning(df: pd.DataFrame, T: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    מחלק את ציר השנים ל-T סלילים (bins) אחידים, ומחזיר df עם עמודת time_bin.

    Parameters
    ----------
    df : pd.DataFrame
        חייב לכלול 'year'.
    T : int
        מספר סלילי הזמן.

    Returns
    -------
    df_out : pd.DataFrame
        עותק עם עמודת 'time_bin' (אינדקס סליל).
    edges : np.ndarray
        גבולות הסלילים על ציר השנים (כולל הקצה העליון).
    """
    if df["year"].isna().all():
        raise ValueError("Column 'year' contains only NaNs.")
    df2 = df.dropna(subset=["year"]).copy()
    df2["year"] = df2["year"].astype(int)
    y_min, y_max = df2["year"].min(), df2["year"].max()
    edges = np.linspace(y_min, y_max + 1, T + 1, dtype=int)
    df2["time_bin"] = pd.cut(df2["year"], bins=edges, labels=False, include_lowest=True)
    return df2, edges


def build_dynamic_snapshots(df_time_binned: pd.DataFrame, id2idx: Dict[str, int], T: int):
    """
    בונה גרפים מכוונים לכל סל זמן ומחזיר ייצוג קומפקטי לשמירה (nodes/edges לכל t).

    Parameters
    ----------
    df_time_binned : pd.DataFrame
        חייב לכלול 'id', 'references', 'time_bin'.
    id2idx : Dict[str, int]
        מיפוי מזהה מאמר -> אינדקס.
    T : int
        מספר סלילי הזמן.

    Returns
    -------
    Dict[int, Dict[str, np.ndarray]]
        מילון לכל t: {"nodes": np.ndarray[int], "edges": np.ndarray[int, int](E,2)}
        edges מייצג קשתות מכוונות (u -> v) לפי references.
    """
    snapshots: Dict[int, nx.DiGraph] = defaultdict(nx.DiGraph)

    # לוודא שיש רשימות רפרנסים מפוענחות
    if "parsed_refs" not in df_time_binned.columns:
        df_time_binned = df_time_binned.copy()
        df_time_binned["parsed_refs"] = df_time_binned["references"].apply(safe_parse_list)

    for _, r in df_time_binned.iterrows():
        t = r.get("time_bin")
        if pd.isna(t):
            continue
        t = int(t)
        u = id2idx.get(str(r["id"]))
        if u is None:
            continue
        snapshots[t].add_node(u)
        for rid in r["parsed_refs"]:
            v = id2idx.get(str(rid))
            if v is not None:
                snapshots[t].add_edge(u, v)

    # המרה למבני np לצורך שמירה מהירה וקריאה קלה
    snap_arrays = {}
    for t in range(T):
        G = snapshots.get(t, nx.DiGraph())
        nodes = np.fromiter(G.nodes(), dtype=np.int64)
        edges = np.array(list(G.edges()), dtype=np.int64) if G.number_of_edges() > 0 else np.empty((0, 2), np.int64)
        snap_arrays[t] = {"nodes": nodes, "edges": edges}
    return snap_arrays


def save_matrix_npz(path: Path, **arrays):
    """
    שומר מספר מערכים (numpy arrays) לקובץ NPZ דחוס.

    Parameters
    ----------
    path : Path
        נתיב קובץ יעד.
    arrays : dict
        מפת שם->מערך לשמירה.

    Notes
    -----
    הפונקציה יוצרת את התיקייה אם אינה קיימת.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def save_json(path: Path, obj):
    """
    שומר אובייקט JSON לקובץ בצורה יפה (indent=2).

    Parameters
    ----------
    path : Path
        נתיב קובץ יעד.
    obj : Any
        אובייקט סיריאלי ל-JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ===========================
#            MAIN
# ===========================

def main():
    """
    נקודת הכניסה הראשית:
    - טוענת CSV
    - בונה id->idx + node_idx
    - מחלקת ל-train/val/test
    - מפיקה פיצ'רים ותוויות
    - בונה adjacency list
    - מבצעת time-binning ובונה snapshots
    - שומרת את כל הארטיפקטים ל-OUTPUT_DIR
    """
    # === LOAD ===
    df = load_dataframe(INPUT_CSV)
    df, id2idx = build_id_index(df)

    # === SPLIT === (לפי counts after chronological sort)
    train_idx, val_idx, test_idx = split_by_year_counts(df)
    print("✅ Split by paper counts")
    print(f"train: {len(train_idx)} | val: {len(val_idx)} | test: {len(test_idx)}")

    # === FEATURES === (BoW על abstract)
    X_all, vec = build_bow_features(df)
    # סאב-מטריצות לפי הספליטים
    x_train = X_all[train_idx]
    x_test = X_all[test_idx]
    x_allx = X_all[train_idx + val_idx]

    # === LABELS === (LabelBinarizer על fos.name)
    Y_all, lb = encode_labels(df)
    y_train = Y_all[train_idx]
    y_test = Y_all[test_idx]
    y_ally = Y_all[train_idx + val_idx]

    # === GRAPH (adjacency list לפי references) ===
    adjacency = build_adjacency(df, id2idx)

    # === DYNAMIC SNAPSHOTS (חלוקת שנים ל-T bins) ===
    df_tb, bin_edges = time_binning(df, T=TIME_STAMPS)
    snapshots = build_dynamic_snapshots(df_tb, id2idx, T=TIME_STAMPS)

    # === SAVE (NPZ/JSON) ===
    # שמירת מטא: כמה time-bins, גבולות, וכמה צמתים כלליים
    save_json(OUTPUT_DIR / "manifest.json",
              {"time_bins": TIME_STAMPS, "bin_edges": bin_edges.tolist(), "num_nodes": len(id2idx)})

    # מילון פיצ'רים (vocabulary) ותוויות (classes) לשימוש עתידי
    save_json(OUTPUT_DIR / "vocab.json", {"feature_names": getattr(vec, "get_feature_names_out")().tolist()})
    save_json(OUTPUT_DIR / "label_binarizer.json", {"classes": getattr(lb, "classes_", []).tolist()})

    # ספליטים לשחזור ניסויים
    save_json(OUTPUT_DIR / "splits.json", {"train": train_idx, "val": val_idx, "test": test_idx})

    # מפת מזהים — חשוב לשמור עקביות בין שלבים
    save_json(OUTPUT_DIR / "id_map.json", id2idx)

    # נשמור את המטריצות הספרסיות (CSR) בפורמט קומפקטי בתוך NPZ
    from scipy.sparse import csr_matrix

    def csr_to_dict(mat: csr_matrix):
        """
        ממיר מטריצת CSR למילון של data/indices/indptr/shape כדי לשמור כ-NPZ.
        מבטיח טיפוסי float32 לצמצום משקל.
        """
        mat = mat.tocsr().astype(np.float32)
        return dict(data=mat.data, indices=mat.indices, indptr=mat.indptr, shape=mat.shape)

    save_matrix_npz(OUTPUT_DIR / "features_train.npz", **csr_to_dict(x_train))
    save_matrix_npz(OUTPUT_DIR / "features_test.npz", **csr_to_dict(x_test))
    save_matrix_npz(OUTPUT_DIR / "features_allx.npz", **csr_to_dict(x_allx))

    # תוויות (צפופות) — שמירה כ-NPZ פשוט
    save_matrix_npz(OUTPUT_DIR / "labels_train.npz", arr=y_train.astype(np.int32))
    save_matrix_npz(OUTPUT_DIR / "labels_test.npz", arr=y_test.astype(np.int32))
    save_matrix_npz(OUTPUT_DIR / "labels_ally.npz", arr=y_ally.astype(np.int32))

    # adjacency כ-JSON (מפת int→List[int]); מפתחות JSON חייבים להיות מחרוזות
    adjacency_json = {str(k): v for k, v in adjacency.items()}
    save_json(OUTPUT_DIR / "graph.json", adjacency_json)

    # snapshots: edges/nodes לכל t — נארוז לקובץ NPZ אחד נוח
    snap_pack = {}
    for t, payload in snapshots.items():
        snap_pack[f"nodes_t{t}"] = payload["nodes"]
        snap_pack[f"edges_t{t}"] = payload["edges"]
    save_matrix_npz(OUTPUT_DIR / "snapshots.npz", **snap_pack)

    print(f"✅ Done. Files written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
