# script/utils/dynamic_node2vec.py
import json
import numpy as np
import torch
import networkx as nx
from pathlib import Path
from node2vec import Node2Vec  # ודאי שמותקן: pip install node2vec

def load_manifest_and_snapshots(root_dir: str):
    """
    root_dir: תיקייה עם manifest.json ו-snapshots.npz (כמו ששמרנו בריפקטור)
    מחזיר: num_nodes, T, dict {t: (nodes_np, edges_np)}
    """
    root = Path(root_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    num_nodes = int(manifest["num_nodes"])
    T = int(manifest["time_bins"])

    snaps = np.load(root / "snapshots.npz", allow_pickle=False)
    snapshots = {}
    for t in range(T):
        nodes = snaps[f"nodes_t{t}"]
        edges = snaps[f"edges_t{t}"]  # shape (E,2) אולי ריק
        snapshots[t] = (nodes, edges)
    return num_nodes, T, snapshots

def build_dynamic_node2vec(
    snapshots: dict,
    num_nodes: int,
    T: int,
    emb_dim: int,
    walk_length: int = 30,
    num_walks: int = 10,
    workers: int = 4,
    window: int = 10,
) -> torch.Tensor:
    """
    בונה embedding_matrix [N, T, F] באמצעות Node2Vec לכל t.
    אם צומת לא קיים בזמן t – נשאר וקטור אפס.

    snapshots: {t: (nodes_np, edges_np)}
    """
    per_t = []
    for t in range(T):
        nodes_np, edges_np = snapshots[t]
        G = nx.DiGraph()
        # חשוב לשמור על זהות האינדקסים כפי ששמרנו
        G.add_nodes_from(nodes_np.tolist())
        if edges_np.size > 0:
            G.add_edges_from(edges_np.tolist())

        n2v = Node2Vec(
            G,
            dimensions=emb_dim,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers,
        )
        w2v = n2v.fit(window=window, min_count=1)

        E_t = torch.zeros((num_nodes, emb_dim), dtype=torch.float32)
        for u in G.nodes():
            key = str(u)
            if key in w2v.wv:
                E_t[u] = torch.tensor(w2v.wv[key])
        per_t.append(E_t)
    return torch.stack(per_t, dim=1)  # [N, T, F]
