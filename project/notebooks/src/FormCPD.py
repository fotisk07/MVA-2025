import numpy as np
from scipy.optimize import linear_sum_assignment 
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
import pandas as pd
rng = np.random.default_rng(0)


def simulate_swap(T=300, sigma=0.5):
    # 6-role formation
    mu = np.array([
        [-8,  2], [-2, 2], [4, 2],
        [-4, -2], [2, -2],
        [0, -6]
    ])

    N = len(mu)
    X = np.zeros((T, N, 2))

    # players assigned fixed roles initially
    roles = np.arange(N)

    # choose two roles to swap
    k1, k2 = 3, 4

    for t in range(T):
        alpha = 0
        if T//3 <= t <= 2*T//3:
            alpha = (t - T//3) / (T//3)

        for i in range(N):
            k = roles[i]
            center = mu[k]

            # interpolate swap
            if k == k1:
                center = (1-alpha)*mu[k1] + alpha*mu[k2]
            if k == k2:
                center = (1-alpha)*mu[k2] + alpha*mu[k1]

            X[t, i] = center + sigma * rng.standard_normal(2)

    return X, mu


def em_hungarian(X, n_iter=20):
    T, N, _ = X.shape

    # init role means as mean of each player trajectory
    mu = X.mean(axis=0).copy()  # (N, 2)
    Sigma = np.array([np.eye(2) for _ in range(N)])

    assignments = np.zeros((T, N), dtype=int)

    for it in tqdm(range(n_iter), desc="EM-Hungarian"):
        # E-step: Hungarian per frame
        for t in range(T):
            cost = np.zeros((N, N))
            for i in range(N):
                for k in range(N):
                    diff = X[t, i] - mu[k]
                    cost[i, k] = diff @ diff  # squared distance
            row_ind, col_ind = linear_sum_assignment(cost)
            # row_ind should be [0..N-1] in order, but be safe
            assignments[t, row_ind] = col_ind

        # M-step: update mu
        for k in range(N):
            mask = assignments == k
            pts = X[mask]
            if len(pts) == 0:
                continue
            mu[k] = pts.mean(axis=0)

    return mu, assignments


def delaunay_adjacency(points):
    """
    Build symmetric adjacency matrix from Delaunay triangulation.
    """
    tri = Delaunay(points)
    N = len(points)
    A = np.zeros((N, N), dtype=int)

    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                u, v = simplex[i], simplex[j]
                A[u, v] = 1
                A[v, u] = 1

    return A


def simulate_roles(F1, F2, T=400, cp=200, sigma=2.0):
    """
    Simulates role trajectories with a single formation change.
    """
    N = len(F1)
    V = np.zeros((T, N, 2))

    for t in range(T):
        mu = F1 if t < cp else F2
        V[t] = mu + sigma * np.random.randn(N,2)

    return V


def plot_delaunay(ax, V_frame, title):
    tri = Delaunay(V_frame)
    ax.triplot(V_frame[:, 0],V_frame[:, 1],tri.simplices,lw=1)
    ax.scatter(V_frame[:, 0],V_frame[:, 1],s=80)

    for i, (x, y) in enumerate(V_frame):
        ax.text(x + 0.8, y + 0.8, str(i+1), fontsize=10)

    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)

    ax.set_xlim(-60, 60)
    ax.set_ylim(-35, 35)

    ax.set_aspect("equal")
    ax.set_title(title)


def plot_cmap(ax, A, title, cmap="Greys"):
    im = ax.imshow(
        A,
        vmin=0,
        vmax=1,
        cmap=cmap,
        interpolation="nearest"
    )
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def manhattan_matrix_distance(A, B):
    """Manhattan distance between two (10x10) adjacency matrices."""
    return np.abs(A - B).sum()


def mean_matrix(A):
    """Mean adjacency matrix over a segment."""
    return A.mean(axis=0)


def build_r_graph_from_matrices(A):
    """
    Prepare all R-side inputs from matrix sequence A.

    Parameters
    ----------
    A : ndarray (T, 10, 10)
    
    Returns
    -------
    X_uni : ndarray (U,100)   unique flattened matrices
    id_vec : ndarray (T,)     mapping frames -> unique nodes (1-based)
    """
    T = A.shape[0]
    X = A.reshape(T, -1)

    # handle repeated observations
    X_uni, inv = np.unique(X, axis=0, return_inverse=True)
    id_vec = inv + 1  # R uses 1-based indexing

    return X_uni, id_vec


def run_gseg_discrete(X_uni, id_vec):
    """
    Calls R gseg1_discrete on a segment and returns CP candidates.

    Returns dictionary:
      {stat_name: (tau, pval)}
    """
    ro.globalenv["dat_uni"] = X_uni
    ro.globalenv["id"] = id_vec

    ro.r("""
    dmat <- dist(dat_uni)
    E <- gSeg::nnl(dmat, 1)

    res <- gSeg::gseg1_discrete(
        n = length(id),
        E = E,
        id = id,
        statistics = "all"
    )
    """)

    # ----------- Extract statistics -----------------
    stats = {}
    families = ["ori", "weighted", "generalized", "max.type"]
    variants = ["a", "u"]
    name_map = {
        "ori": "Original",
        "weighted": "Weighted",
        "generalized": "Generalized",
        "max.type": "Max"
    }
    for fam in families:
        for var in variants:
            label = f"{name_map[fam]}_{var}"

            tau = int(ro.r(f"res$scanZ${fam}$tauhat_{var}")[0])
            p   = float(ro.r(f"res$pval.appr${fam}_{var}")[0])

            stats[label] = {
                "tau": tau,
                "pval": p
            }

    return stats


def is_significant_cp(A_seg, tau,
                      pval,
                      fps,
                      alpha=0.01,
                      min_minutes=5,
                      min_dist=7.0):
    """
    Determines if a CP satisfies ALL paper constraints.
    """

    # --- criterion 1: statistical
    if pval >= alpha:
        return False

    # --- criterion 2: duration
    min_frames = int(min_minutes * 60 * fps)

    if min(tau, A_seg.shape[0] - tau) < min_frames:
        return False

    # --- criterion 3: matrix distance
    M1 = mean_matrix(A_seg[:tau])
    M2 = mean_matrix(A_seg[tau:])

    d = manhattan_matrix_distance(M1, M2)

    if d < min_dist:
        return False

    return True


def recursive_cp(A,
                 start_idx,
                 fps,
                 alpha,
                 min_minutes,
                 min_dist):

    results = []

    T = A.shape[0]

    min_frames = int(min_minutes * 60 * fps)

    # Too small to segment
    if T < 2 * min_frames:
        return results

    # -----------------
    # Run gseg
    # -----------------
    X_uni, id_vec = build_r_graph_from_matrices(A)
    stats = run_gseg_discrete(X_uni, id_vec)

    # -----------------
    # Select best statistic (Generalized union)
    # -----------------
    best = stats["Generalized_u"]

    tau = best["tau"]
    pval = best["pval"]

    # -----------------
    # Significance test
    # -----------------
    if not is_significant_cp(A, tau, pval,
                             fps=fps,
                             alpha=alpha,
                             min_minutes=min_minutes,
                             min_dist=min_dist):
        return results

    # Global index of CP
    global_tau = start_idx + tau

    # ------------------
    # Recursive split
    # ------------------

    results.append(global_tau)

    left  = recursive_cp(A[:tau], start_idx,
                          fps, alpha, min_minutes, min_dist)

    right = recursive_cp(A[tau:], global_tau,
                          fps, alpha, min_minutes, min_dist)

    return left + results + right


def detect_formation_changes(A,
                              fps=1,
                              alpha=0.01,
                              min_minutes=5,
                              min_dist=7.0):
    """
    Full SoccerCPD-style detection.

    Parameters
    ----------
    A : ndarray (T,10,10)
        Role adjacency matrix sequence.
    fps : float
        Sampling rate in frames/second.
    alpha : float
        Statistical threshold.
    min_minutes : int
        Minimum segment duration in minutes.
    min_dist : float
        Manhattan distance threshold between mean formations.
    Returns
    -------
    dict with:
      - 'change_points' : sorted list of indices
      - 'num_phases'    : count of segments
      - 'segments'     : list of (start, end)
    """

    cps = recursive_cp(A,
                        start_idx=0,
                        fps=fps,
                        alpha=alpha,
                        min_minutes=min_minutes,
                        min_dist=min_dist)

    cps = sorted(cps)

    segments = []
    prev = 0

    for cp in cps:
        segments.append((prev, cp))
        prev = cp

    segments.append((prev, A.shape[0]))

    return {
        "change_points": cps,
        "num_phases": len(segments),
        "segments": segments
    }

