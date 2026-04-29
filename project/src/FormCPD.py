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




def manhattan_matrix_distance(A, B):
    return np.abs(A - B).sum()


def mean_matrix(A):
    return A.mean(axis=0)


def build_r_graph_from_matrices(A):
    T = A.shape[0]
    X = A.reshape(T, -1)

    # handle repeated observations
    X_uni, inv = np.unique(X, axis=0, return_inverse=True)
    id_vec = inv + 1  # R uses 1-based indexing

    return X_uni, id_vec


def run_gseg_discrete(X_uni, id_vec):
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

# We also define a pure Python version of the formation change detection in case of R/gSeg issues.
def detect_formation_changes_python(adjacency_sequence, min_minutes=5, n_bkps=4):
    """
    Detects formation changes using the Kernel Change Point Detection (Python only).
    Replaces the unstable R/gSeg implementation.
    """
    # Flatten (T, 10, 10) matrices into (T, 100) vectors
    T_sub = adjacency_sequence.shape[0]
    data_flat = adjacency_sequence.reshape(T_sub, -1)
    
    # Calculate minimum segment size in frames (1Hz assumption)
    min_size = min_minutes * 60 
    
    # Use Binary Segmentation
    try:
        algo = rpt.Binseg(model="l2", min_size=min_size).fit(data_flat)
        change_points = algo.predict(n_bkps=n_bkps)
    except Exception as e:
        print(f"Warning: Segmentation failed - {e}.")
        change_points = [T_sub]

    # Format results like the original output
    bkps = sorted(list(set([0] + change_points + [T_sub])))
    
    # Create segments list [(start, end), (start, end)...]
    segments = []
    for i in range(len(bkps) - 1):
        segments.append((bkps[i], bkps[i+1]))
    internal_cps = [cp for cp in bkps if cp != 0 and cp != T_sub]
    
    return {
        'num_phases': len(segments),
        'change_points': internal_cps,
        'segments': segments
    }