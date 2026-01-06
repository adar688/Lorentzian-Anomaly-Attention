# -*- coding: utf-8 -*-
"""
Dynamic citation dataset builder.

This script:
1) Loads a filtered citation CSV with columns like: id, year, fos.name, abstract, references.
2) Builds a node index (id -> node_idx).
3) Creates chronological, class-stratified train/val/test splits.
4) Builds Bag-of-Words features (CountVectorizer).
5) Encodes labels using LabelEncoder (1D int vector).
6) Builds a static adjacency dictionary and dynamic snapshots over time.
7) Saves all artifacts (features, labels, splits, graph, snapshots, vocab, label classes).
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
from sklearn.preprocessing import LabelEncoder

# --- external config (optional) ---
try:
    from config import args
    TIME_STAMPS = args.Time_stamps
except Exception:
    TIME_STAMPS = 10

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- IO paths ---
INPUT_CSV = Path("script/data/final_filtered_by_fos_and_reference.csv")
OUTPUT_DIR = Path("script/data/custom_out")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ===========================
#           HELPERS
# ===========================

def load_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "id" not in df.columns or "year" not in df.columns:
        raise ValueError("CSV must include 'id' and 'year' columns.")
    return df


def build_id_index(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    paper_ids = df["id"].astype(str).tolist()
    id2idx = {pid: i for i, pid in enumerate(paper_ids)}
    df_out = df.copy()
    df_out["node_idx"] = df_out["id"].astype(str).map(id2idx)
    return df_out, id2idx


def _split_one_class_chrono(
    idx_by_year: np.ndarray,
    ratios: Tuple[float, float, float]
) -> Tuple[List[int], List[int], List[int]]:
    """Split a single-class index array chronologically into train/val/test."""
    train_ratio, val_ratio, _ = ratios
    n = len(idx_by_year)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (
        idx_by_year[:train_end].tolist(),
        idx_by_year[train_end:val_end].tolist(),
        idx_by_year[val_end:].tolist(),
    )


def split_chrono_stratified(
    df: pd.DataFrame,
    label_col: str = "fos.name",
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    min_per_class: Tuple[int, int, int] = (1, 1, 1),
) -> Tuple[List[int], List[int], List[int]]:
    """
    Stratified-by-class chronological split.

    For each class separately:
      1) Sort samples by year.
      2) Split indices according to the given ratios.
      3) Merge per-class splits into global train/val/test index lists.
    """
    df = df.copy()
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DF.")

    df = df.sort_values(by="year", ascending=True).reset_index(drop=True)

    classes = df[label_col].fillna("").astype(str).unique().tolist()
    train_idx, val_idx, test_idx = [], [], []

    for cls in classes:
        sub = df[df[label_col].fillna("").astype(str) == cls].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(by="year", ascending=True)
        idx_arr = sub["node_idx"].to_numpy(dtype=int)
        tr, va, te = _split_one_class_chrono(idx_arr, ratios)
        train_idx += tr
        val_idx += va
        test_idx += te

    # Ensure no overlap between splits and keep global chronological order in each split
    train_idx = sorted(
        set(train_idx),
        key=lambda i: int(df.loc[df["node_idx"] == i, "year"].values[0]),
    )
    val_idx = sorted(
        set(val_idx),
        key=lambda i: int(df.loc[df["node_idx"] == i, "year"].values[0]),
    )
    test_idx = sorted(
        set(test_idx),
        key=lambda i: int(df.loc[df["node_idx"] == i, "year"].values[0]),
    )

    # Soft fallback: if some split is missing certain classes,
    # move a few samples from other splits to guarantee minimum count per class.
    def _enforce_min_per_class(target_idx: List[int], name: str):
        if not target_idx:
            return
        need = dict(zip(classes, list(min_per_class)))
        counts = pd.Series(
            df.set_index("node_idx").loc[target_idx, label_col].values
        ).value_counts().to_dict()

        for c in classes:
            have = counts.get(c, 0)
            want = need.get(c, 0)
            if have >= want:
                continue

            # Try to move samples of class c from other splits into the target split
            def _steal(from_list: List[int]) -> bool:
                for k in range(len(from_list)):
                    cand = from_list[k]
                    if str(
                        df.loc[df["node_idx"] == cand, label_col].values[0]
                    ) == c:
                        target_idx.append(cand)
                        del from_list[k]
                        return True
                return False

            # Priority of where to steal from depends on the target split
            if name == "train":
                if not _steal(val_idx):
                    _steal(test_idx)
            elif name == "val":
                if not _steal(train_idx):
                    _steal(test_idx)
            else:  # test
                if not _steal(val_idx):
                    _steal(train_idx)

    _enforce_min_per_class(train_idx, "train")
    _enforce_min_per_class(val_idx, "val")
    _enforce_min_per_class(test_idx, "test")

    return train_idx, val_idx, test_idx


def build_bow_features(df: pd.DataFrame):
    """
    Build Bag-of-Words features from the 'abstract' column.

    Returns:
      X:   scipy.sparse matrix (N x V)
      vec: fitted CountVectorizer
    """
    abstracts = df.get("abstract", pd.Series([], dtype=str)).fillna("")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(abstracts)
    return X, vectorizer


def encode_labels_as_int(df: pd.DataFrame, label_col: str = "fos.name"):
    """
    Encode labels as a 1D int vector using LabelEncoder (0..C-1).

    Returns:
      y_int: np.ndarray, shape [N], integer class IDs
      le:    LabelEncoder instance (with .classes_)
      y_raw: np.ndarray of original string labels (for debugging / analysis)
    """
    fos = df.get(label_col, pd.Series([], dtype=str)).fillna("").astype(str)
    y_raw = fos.to_numpy()
    le = LabelEncoder()
    y_int = le.fit_transform(y_raw)
    return y_int, le, y_raw


def safe_parse_list(s: str) -> List[str]:
    """Safely parse a list-like string using ast.literal_eval, return list of strings."""
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
    Build a directed adjacency list from 'references' column.

    Returns:
      dict: node_idx -> list of neighbor node_idx (outgoing edges).
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
    Assign each paper to a time bin based on its year.

    Returns:
      df2:      filtered dataframe with 'time_bin' column
      bin_edges: np.ndarray of bin edges used in pd.cut
    """
    if df["year"].isna().all():
        raise ValueError("Column 'year' contains only NaNs.")
    df2 = df.dropna(subset=["year"]).copy()
    df2["year"] = df2["year"].astype(int)
    y_min, y_max = df2["year"].min(), df2["year"].max()
    edges = np.linspace(y_min, y_max + 1, T + 1)
    df2["time_bin"] = pd.cut(
        df2["year"], bins=edges, labels=False, include_lowest=True
    )
    return df2, edges


def build_dynamic_snapshots(
    df_time_binned: pd.DataFrame,
    id2idx: Dict[str, int],
    T: int,
):
    """
    Build dynamic directed graphs for each time bin.

    Returns:
      snapshots: dict[t] -> {'nodes': np.ndarray, 'edges': np.ndarray of shape (E, 2)}
    """
    snapshots: Dict[int, nx.DiGraph] = defaultdict(nx.DiGraph)
    if "parsed_refs" not in df_time_binned.columns:
        df_time_binned = df_time_binned.copy()
        df_time_binned["parsed_refs"] = df_time_binned["references"].apply(
            safe_parse_list
        )

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

    snap_arrays = {}
    for t in range(T):
        G = snapshots.get(t, nx.DiGraph())
        nodes = np.fromiter(G.nodes(), dtype=np.int64)
        edges = (
            np.array(list(G.edges()), dtype=np.int64)
            if G.number_of_edges() > 0
            else np.empty((0, 2), np.int64)
        )
        snap_arrays[t] = {"nodes": nodes, "edges": edges}
    return snap_arrays


def save_matrix_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ===========================
#            MAIN
# ===========================

def main():
    df = load_dataframe(INPUT_CSV)
    df, id2idx = build_id_index(df)

    # === STRATIFIED CHRONOLOGICAL SPLIT ===
    train_idx, val_idx, test_idx = split_chrono_stratified(df, label_col="fos.name")
    print("✅ Split by paper counts (chronological, stratified by class)")
    print(f"train: {len(train_idx)} | val: {len(val_idx)} | test: {len(test_idx)}")



    # === FEATURES (BoW) ===
    X_all, vec = build_bow_features(df)
    x_train = X_all[train_idx]
    x_val = X_all[val_idx]
    x_test = X_all[test_idx]
    # Concatenation of train + val, for legacy GCN-style loaders if needed
    x_allx = X_all[train_idx + val_idx]

    # === LABELS (LabelEncoder → 1D ints) ===
    y_int, le, y_raw = encode_labels_as_int(df, label_col="fos.name")
    y_train = y_int[train_idx]
    y_val = y_int[val_idx]
    y_test = y_int[test_idx]
    y_ally = y_int[train_idx + val_idx]


    # === GRAPH (static adjacency) ===
    adjacency = build_adjacency(df, id2idx)

    # === DYNAMIC SNAPSHOTS ===
    df_tb, bin_edges = time_binning(df, T=TIME_STAMPS)
    snapshots = build_dynamic_snapshots(df_tb, id2idx, T=TIME_STAMPS)

    # === SAVE META ===
    save_json(
        OUTPUT_DIR / "manifest.json",
        {
            "time_bins": TIME_STAMPS,
            "bin_edges": bin_edges.tolist(),
            "num_nodes": len(id2idx),
        },
    )

    # Feature names – use the fitted vectorizer
    try:
        feature_names = vec.get_feature_names_out().tolist()
    except Exception:
        feature_names = list(getattr(vec, "vocabulary_", {}).keys())
    save_json(OUTPUT_DIR / "vocab.json", {"feature_names": feature_names})

    # Class names from the LabelEncoder
    save_json(OUTPUT_DIR / "label_classes.json", {"classes": le.classes_.tolist()})
    np.save(OUTPUT_DIR / "labels_raw.npy", y_raw)

    # Splits for reproducibility
    save_json(OUTPUT_DIR / "splits.json", {"train": train_idx, "val": val_idx, "test": test_idx})
    save_json(OUTPUT_DIR / "id_map.json", id2idx)

    # === SAVE MATRICES (CSR) ===
    from scipy.sparse import csr_matrix

    def csr_to_dict(mat: csr_matrix):
        mat = mat.tocsr().astype(np.float32)
        return dict(
            data=mat.data,
            indices=mat.indices,
            indptr=mat.indptr,
            shape=mat.shape,
        )

    save_matrix_npz(OUTPUT_DIR / "features_train.npz", **csr_to_dict(x_train))
    save_matrix_npz(OUTPUT_DIR / "features_val.npz", **csr_to_dict(x_val))
    save_matrix_npz(OUTPUT_DIR / "features_test.npz", **csr_to_dict(x_test))
    save_matrix_npz(OUTPUT_DIR / "features_allx.npz", **csr_to_dict(x_allx))

    # Labels as 1D int arrays (what your main model expects)
    save_matrix_npz(OUTPUT_DIR / "labels_train.npz", arr=y_train.astype(np.int32))
    save_matrix_npz(OUTPUT_DIR / "labels_val.npz", arr=y_val.astype(np.int32))
    save_matrix_npz(OUTPUT_DIR / "labels_test.npz", arr=y_test.astype(np.int32))
    save_matrix_npz(OUTPUT_DIR / "labels_ally.npz", arr=y_ally.astype(np.int32))

    # Adjacency JSON
    adjacency_json = {str(k): v for k, v in adjacency.items()}
    save_json(OUTPUT_DIR / "graph.json", adjacency_json)

    # Snapshots NPZ
    snap_pack = {}
    for t, payload in snapshots.items():
        snap_pack[f"nodes_t{t}"] = payload["nodes"]
        snap_pack[f"edges_t{t}"] = payload["edges"]
    save_matrix_npz(OUTPUT_DIR / "snapshots.npz", **snap_pack)

    print(f"✅ Done. Files written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
