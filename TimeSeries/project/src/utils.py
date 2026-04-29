import pandas as pd
import pickle
from pathlib import Path

def save_results(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(results, f)
    print(f"[saved] {path}")

def load_results(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def match_cps(cps_a, cps_b, tol_seconds=30):
    matched = 0
    used = set()

    for t in cps_a:
        for j, s in enumerate(cps_b):
            if j in used:
                continue
            if abs((t - s).total_seconds()) <= tol_seconds:
                matched += 1
                used.add(j)
                break

    precision = matched / max(len(cps_a), 1)
    recall = matched / max(len(cps_b), 1)

    return precision, recall

def pairwise_agreement(results, tol_seconds=30):
    methods = list(results.keys())
    rows = []

    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if j <= i:
                continue

            cps1 = results[m1]["change_points"]
            cps2 = results[m2]["change_points"]

            p, r = match_cps(cps1, cps2, tol_seconds)

            rows.append({
                "method_1": m1,
                "method_2": m2,
                "precision": p,
                "recall": r,
                "f1": 2*p*r / (p+r) if (p+r) > 0 else 0.0,
            })

    return pd.DataFrame(rows)