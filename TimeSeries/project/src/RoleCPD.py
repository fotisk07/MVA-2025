import numpy as np
from matplotlib import pyplot as plt


# try : 
#     import os
#     os.environ["R_HOME"] = r"C:\Program Files\R\R-4.3.1"
#     os.add_dll_directory(r"C:\Program Files\R\R-4.3.1\bin\x64")
# except:
#     pass

try : 
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr

    numpy2ri.activate()
    gSeg = importr("gSeg")
except:
    raise ValueError


def generate_synthetic_role_data(T=450, N=10, switch_times=[150, 300], noise_level=0.1):
    """
    Generates a sequence of role permutations.
    Returns: Array of shape (T, N) where row t is the permutation at time t.
    """
    # Phase 1
    # Base permutation: Identity
    base_roles = np.arange(N)
    
    # Phase 2
    # Tactical Change: Swap roles 7 and 9
    phase2_roles = base_roles.copy()
    phase2_roles[7], phase2_roles[9] = 9, 7

    # Phase 3
    # Tactical Change: Swap roles 3 and 4
    phase3_roles = base_roles.copy()
    phase3_roles[3], phase3_roles[4] = 4, 3

    permutations = []
    
    for t in range(T):
        # Determine dominant tactic
        if t < switch_times[0]:
            current_perm = base_roles.copy()
        elif t < switch_times[1]:
            current_perm = phase2_roles.copy()
        else:
            current_perm = phase3_roles.copy()
            
        # Add temporary noise (random temporary swaps)
        if np.random.rand() < noise_level:
            # Randomly swap two players to simulate temporary cover/overlap
            i, j = np.random.choice(N, 2, replace=False)
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            
        permutations.append(current_perm)
        
    return np.array(permutations)


def hamming_distance(perm_a, perm_b):
    """
    Calculates normalized Hamming distance (Switch Rate) between two permutations.
    """
    N = len(perm_a)
    diffs = np.sum(perm_a != perm_b)
    return diffs / N


def preprocess_permutations(perms, N=10):
    """
    1. Identifies the global dominant permutation (Identity).
    2. Calculates Switch Rate for every frame.
    3. Filters out frames with Switch Rate > 0.7.
    """
    vals, counts = np.unique(perms, axis=0, return_counts=True)
    dominant_perm = vals[np.argmax(counts)]
    
    valid_indices = []
    switch_rates = []
    
    for t in range(len(perms)):
        sr = hamming_distance(perms[t], dominant_perm)
        switch_rates.append(sr)
        
        if sr <= 0.7:  # Threshold from paper
            valid_indices.append(t)
            
    return np.array(valid_indices), np.array(switch_rates)


def run_role_gseg(perms):
    """
    Runs discrete g-segmentation on a sequence of permutations using Hamming Distance.
    """
    # unique permutations and mapping
    uni_perms, inverse_indices = np.unique(perms, axis=0, return_inverse=True)
    y = inverse_indices + 1  # R uses 1-based indexing
    
    n = len(perms)
    n_unique = len(uni_perms)
    
    if n_unique < 2:
        print(f"Warning: Not enough unique permutations ({n_unique}) for g-segmentation.")
        return 0, 1.0  # No change point, p-value = 1

    # Compute Hamming distance matrix
    dist_matrix = np.zeros((n_unique, n_unique))
    for i in range(n_unique):
        for j in range(i, n_unique):
            d = hamming_distance(uni_perms[i], uni_perms[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            
    # 3. Pass to R
    numpy2ri.activate()
    robjects.globalenv['dist_mat'] = dist_matrix
    robjects.globalenv['y'] = y
    robjects.globalenv['n_len'] = n
    # SoccerCPD framework uses K=1 and strict boundary windows (n0, n1)
    # n0 = 0.1 * n, n1 = 0.9 * n
    
    r_script = """
    library(gSeg)
    E <- gSeg::nnl(as.dist(dist_mat), K=1)
    res <- gSeg::gseg1_discrete(n_len, E, y, statistics="generalized", n0=0.1*n_len, n1=0.9*n_len)
    tau_hat <- res$scanZ$generalized$tauhat_a
    pval <- res$pval.appr$generalized_a
    """
    try:
        robjects.r(r_script)

        tau = robjects.globalenv['tau_hat'][0]
        pval = robjects.globalenv['pval'][0]
    except Exception as e:
        print(f"Error running g-segmentation in R: {e}")
        return 0, 1.0  # Default return on error
    
    return int(tau), float(pval)


def get_dominant_perm(perms):
    vals, counts = np.unique(perms, axis=0, return_counts=True)
    return vals[np.argmax(counts)]


def recursive_role_segmentation(perms, original_indices, alpha=0.01, min_seg_len=50):
    """
    Recursively detects change points in a sequence of role permutations.

    Parameters:
        perms: The array of permutations (subset of valid frames).
        original_indices: Mapping from current subset indices to original video frame indices.
        alpha: Significance level (p-value threshold).
        min_seg_len: Minimum number of frames required to attempt segmentation.

    Returns:
        List of detected change points (in original frame indices).
    """
    n = len(perms)

    # Base Case: Segment is too short to split
    if n < min_seg_len:
        return []

    # Run G-Segmentation on the current segment
    tau_local, pval = run_role_gseg(perms)

    # Significance Condition 1: P-value, if p-value is too high, we assume no change in this segment
    if pval >= alpha:
        return []

    # Define split point
    split_idx = int(tau_local)
    
    # Boundary check
    if split_idx < 5 or split_idx > n - 5:
        return []

    # Significance Condition 2: Dominant Permutation Change
    # As per the SoccerCPD framework, we reject changes where the dominant role assignment is identical before and after the cut.
    seg1 = perms[:split_idx]
    seg2 = perms[split_idx:]
    
    dom1 = get_dominant_perm(seg1)
    dom2 = get_dominant_perm(seg2)
    
    if hamming_distance(dom1, dom2) == 0:
        # The tactic (most frequent permutation) didn't actually change
        return []

    # If Significant: Record Global Timestamp and Recurse
    global_cp = original_indices[split_idx]
    
    print(f"Significant Change Detected at Frame {global_cp} (p={pval:.2e})")
    print(f"Left Dominant: {dom1}")
    print(f"Right Dominant: {dom2}")

    # Recursive calls
    left_cps = recursive_role_segmentation(seg1, original_indices[:split_idx], alpha, min_seg_len)
    right_cps = recursive_role_segmentation(seg2, original_indices[split_idx:], alpha, min_seg_len)

    return sorted(left_cps + [global_cp] + right_cps)