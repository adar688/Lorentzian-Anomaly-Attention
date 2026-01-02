import os, json
import numpy as np
import scipy.sparse as sp

def _load_csr_from_npz(path: str) -> sp.csr_matrix:
    blob = np.load(path, allow_pickle=False)
    return sp.csr_matrix((blob["data"], blob["indices"], blob["indptr"]), shape=tuple(blob["shape"]))

def _to_class_idx(arr: np.ndarray) -> np.ndarray:
    return arr.argmax(axis=1).astype(np.int64) if arr.ndim == 2 else arr.astype(np.int64)

def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    man_path    = os.path.join(data_path, "manifest.json")
    splits_path = os.path.join(data_path, "splits.json")
    graph_path  = os.path.join(data_path, "graph.json")
    feats_tr    = os.path.join(data_path, "features_train.npz")
    feats_ax    = os.path.join(data_path, "features_allx.npz")  # train+val
    feats_te    = os.path.join(data_path, "features_test.npz")
    labs_tr     = os.path.join(data_path, "labels_train.npz")
    labs_ally   = os.path.join(data_path, "labels_ally.npz")    # train+val
    labs_te     = os.path.join(data_path, "labels_test.npz")

    required = [man_path, splits_path, graph_path, feats_tr, feats_ax, feats_te, labs_tr, labs_ally, labs_te]
    for p in required:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    with open(man_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    N = int(manifest["num_nodes"])
    idx_train = list(map(int, splits["train"]))
    idx_val   = list(map(int, splits["val"]))
    idx_test  = list(map(int, splits["test"]))

    if use_feats:
        X_tr = _load_csr_from_npz(feats_tr)
        X_ax = _load_csr_from_npz(feats_ax)
        X_te = _load_csr_from_npz(feats_te)
        d = X_ax.shape[1]
        features = sp.csr_matrix((N, d), dtype=X_ax.dtype)
        features[idx_train] = X_tr
        features[idx_val]   = X_ax[len(idx_train): len(idx_train)+len(idx_val)]
        features[idx_test]  = X_te
    else:
        features = sp.eye(N, dtype=np.float32, format="csr")


    y_tr   = np.load(labs_tr,   allow_pickle=False)["arr"]
    y_ally = np.load(labs_ally, allow_pickle=False)["arr"]
    y_te   = np.load(labs_te,   allow_pickle=False)["arr"]
    labels = np.full((N,), -1, dtype=np.int64)
    labels[idx_train] = _to_class_idx(y_tr)
    labels[idx_val]   = _to_class_idx(y_ally[len(idx_train): len(idx_train)+len(idx_val)])
    labels[idx_test]  = _to_class_idx(y_te)


    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)  # {"0":[1,2], "1":[...], ...}
    rows, cols = [], []
    for u_str, neigh in graph.items():
        u = int(u_str)
        for v in neigh:
            rows.append(u); cols.append(int(v))
    if rows:
        data = np.ones(len(rows), dtype=np.float32)
        adj = sp.coo_matrix((data, (np.array(rows), np.array(cols))), shape=(N, N)).tocsr()
    else:
        adj = sp.csr_matrix((N, N), dtype=np.float32)

    return adj, features, labels, idx_train, idx_val, idx_test
