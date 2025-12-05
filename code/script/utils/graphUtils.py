import matplotlib.pyplot as plt
import numpy as np
import os

# ======================
#        PLOTS
# ======================
def plot_mean_std(mu: np.ndarray, std: np.ndarray, top_k: int = 10, save_dir: str = "plots"):
    """
    Two separate bar charts:
      1) Mean (μ) per node, highlighting Top-K
      2) Std (σ) per node, highlighting Top-K
    """
    os.makedirs(save_dir, exist_ok=True)

    N = len(mu)
    x = np.arange(N)

    mu = np.nan_to_num(
        mu,
        nan=0.0,
        posinf=np.max(mu[np.isfinite(mu)]) if np.any(np.isfinite(mu)) else 0.0,
        neginf=0.0,
    )
    std = np.nan_to_num(
        std,
        nan=0.0,
        posinf=np.max(std[np.isfinite(std)]) if np.any(np.isfinite(std)) else 0.0,
        neginf=0.0,
    )

    # Mean
    plt.figure(figsize=(12, 6))
    plt.bar(x, mu, color='skyblue', alpha=0.7, label='Mean (μ)')
    top_mean_idx = np.argsort(-mu)[:top_k]
    plt.scatter(top_mean_idx, mu[top_mean_idx], color='red', s=80, label=f'Top {top_k} Mean')
    plt.title('Isolation Forest — Mean (μ) per Node', fontsize=14)
    plt.xlabel('Node index (i)', fontsize=12)
    plt.ylabel('Mean anomaly score', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    mean_path = os.path.join(save_dir, "plot_mean.png")
    plt.savefig(mean_path, dpi=150)
    plt.close()
    print(f"💾 Saved: {mean_path}")

    # Std
    plt.figure(figsize=(12, 6))
    plt.bar(x, std, color='orange', alpha=0.7, label='Std Dev (σ)')
    top_std_idx = np.argsort(-std)[:top_k]
    plt.scatter(top_std_idx, std[top_std_idx], color='purple', s=80, label=f'Top {top_k} Std')
    plt.title('Isolation Forest — Standard Deviation (σ) per Node', fontsize=14)
    plt.xlabel('Node index (i)', fontsize=12)
    plt.ylabel('Standard deviation of anomaly score', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    std_path = os.path.join(save_dir, "plot_std.png")
    plt.savefig(std_path, dpi=150)
    plt.close()
    print(f"💾 Saved: {std_path}")


def plot_nodes_hist_mean(mu: np.ndarray, method: str = "IF", save_dir: str = "plots"):
    """
    Bar chart over all nodes: X = node id, Y = mean anomaly score (μ).
    """
    os.makedirs(save_dir, exist_ok=True)
    N = len(mu)
    x = np.arange(N)

    mu = np.nan_to_num(
        mu,
        nan=0.0,
        posinf=np.max(mu[np.isfinite(mu)]) if np.any(np.isfinite(mu)) else 0.0,
        neginf=0.0,
    )

    plt.figure(figsize=(14, 6))
    plt.bar(x, mu, alpha=0.8)
    plt.title(f"{method} — Mean (μ) per node")
    plt.xlabel("Node ID")
    plt.ylabel("Mean anomaly score")
    plt.tight_layout()
    out = os.path.join(save_dir, f"{method.lower()}_nodes_mean_bar.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"💾 Saved: {out}")

def plot_nodes_hist_std(std: np.ndarray, method: str = "IF", save_dir: str = "plots"):
    """
    Bar chart over all nodes: X = node id, Y = std of anomaly score (σ).
    """
    os.makedirs(save_dir, exist_ok=True)
    N = len(std)
    x = np.arange(N)

    std = np.nan_to_num(
        std,
        nan=0.0,
        posinf=np.max(std[np.isfinite(std)]) if np.any(np.isfinite(std)) else 0.0,
        neginf=0.0,
    )

    plt.figure(figsize=(14, 6))
    plt.bar(x, std, alpha=0.8)
    plt.title(f"{method} — Std (σ) per node")
    plt.xlabel("Node ID")
    plt.ylabel("Std of anomaly score")
    plt.tight_layout()
    out = os.path.join(save_dir, f"{method.lower()}_nodes_std_bar.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"💾 Saved: {out}")

def plot_top_node_timeseries_by_method(
    AS: np.ndarray,
    mu: np.ndarray,
    method_name: str,
    save_dir: str = "plots",
) -> int:
    """
    Plot time-series for the most anomalous node according to the given method.
    - AS: [N, T] anomaly matrix for the method (already normalized to [0,1])
    - mu: [N] per-node mean anomaly scores for the method
    - method_name: "IF" or "LOF" (used for labels and filename)
    Returns: selected node index.
    """
    os.makedirs(save_dir, exist_ok=True)
    if AS is None or mu is None or len(mu) == 0:
        print(f"[{method_name}] No data to plot.")
        return -1

    node_idx = int(np.argmax(np.nan_to_num(mu, nan=-1e30)))
    y = AS[node_idx, :]
    t = np.arange(y.shape[0])

    plt.figure(figsize=(12, 5))
    plt.plot(t, y, marker='o', linewidth=1.5, label=method_name)
    plt.title(f"Most anomalous node time-series by {method_name} (node {node_idx})")
    plt.xlabel("Time (snapshot index)")
    plt.ylabel("Anomaly score")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(save_dir, f"top_node_{method_name.lower()}_{node_idx}_timeseries.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"💾 Saved: {out}")
    return node_idx
def plot_if_lof_timeseries_for_nodes(
    AS_if: np.ndarray,
    AS_lof: np.ndarray,
    node_indices,
    save_dir: str = "plots/common_if_lof_timeseries",
):
    """
    For each node in node_indices, plot a time-series with two lines:
      - IF anomaly score over time
      - LOF anomaly score over time

    AS_if, AS_lof: [N, T] matrices (already normalized to [0,1]).
    node_indices: iterable of node indices (ints).
    """
    os.makedirs(save_dir, exist_ok=True)

    if AS_if is None or AS_lof is None:
        print("[IF+LOF] Missing anomaly matrices, skipping.")
        return

    if AS_if.shape != AS_lof.shape:
        print(f"[IF+LOF] Shape mismatch: IF {AS_if.shape}, LOF {AS_lof.shape}")
        return

    N, T = AS_if.shape

    for idx in sorted(node_indices):
        if idx < 0 or idx >= N:
            print(f"[IF+LOF] Node index {idx} out of range, skipping.")
            continue

        y_if = np.nan_to_num(AS_if[idx, :], nan=0.0)
        y_lof = np.nan_to_num(AS_lof[idx, :], nan=0.0)
        t = np.arange(T)

        plt.figure(figsize=(12, 5))
        plt.plot(t, y_if, marker='o', linewidth=1.5, label='IF')
        plt.plot(t, y_lof, marker='x', linewidth=1.5, label='LOF')

        plt.title(f"Node {idx} anomaly time-series (IF vs LOF)")
        plt.xlabel("Time (snapshot index)")
        plt.ylabel("Anomaly score")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(save_dir, f"node_{idx}_if_lof_timeseries.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"💾 Saved: {out_path}")

def plot_hist_distribution(values: np.ndarray, title: str, xlabel: str, save_path: str):
    """
    Statistical histogram of anomaly values across nodes.
    """
    values = np.nan_to_num(values, nan=0.0)
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of nodes")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"💾 Saved: {save_path}")

def plot_noise_validation_tpr(
    tpr_if,
    tpr_lof=None,
    save_path: str = "plots/noise_validation_tpr.png",
):
    """
    Plot TPR over noise-injection validation iterations for IF (and optionally LOF).

    tpr_if: list or array of TPR values for Isolation Forest, length = num_iterations
    tpr_lof: optional list or array of TPR values for LOF (same length)
    save_path: path to save the PNG plot
    """
    if tpr_if is None or len(tpr_if) == 0:
        print("[plot_noise_validation_tpr] No TPR values for IF, skipping plot.")
        return

    iterations = list(range(1, len(tpr_if) + 1))

    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    plt.figure(figsize=(8, 5))

    # Plot IF TPR
    plt.plot(iterations, tpr_if, marker="o", linestyle="-", label="IF TPR")

    # Optionally plot LOF TPR
    if tpr_lof is not None and len(tpr_lof) == len(tpr_if):
        plt.plot(iterations, tpr_lof, marker="s", linestyle="--", label="LOF TPR")

    plt.xlabel("Validation iteration")
    plt.ylabel("TPR")
    plt.title("Noise-injection validation: TPR over iterations")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(iterations)
    plt.ylim(0.0, 1.0)

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[plot_noise_validation_tpr] Saved TPR plot to: {save_path}")

