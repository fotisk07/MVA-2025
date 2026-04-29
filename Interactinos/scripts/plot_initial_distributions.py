import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

ROOT = pathlib.Path(__file__).parent.parent
OUT_DIR = ROOT / "results" / "figures" / "baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)
N = 200_000

Y  = RNG.standard_normal(N)
Yb = RNG.standard_normal(N)
P  = RNG.integers(0, 2, size=N)
U  = RNG.uniform(0, 1, size=N)

k1 = np.sqrt(3) / 10; theta1 = 0.1
k5 = 0.25; theta5 = 0.1
k6 = 0.3; theta6 = 0.07

samples = {
    "Case 1\n$X_0=0.2Y$\n$\\sigma{=}0.20$": 0.2 * Y,
    "Case 2\n$X_0=0.3+0.05Y$\n$\\sigma{=}0.05$, $\\mu{=}0.3$": 0.3 + 0.05 * Y,
    "Case 3\n$X_0=0.05Y$\n$\\sigma{=}0.05$": 0.05 * Y,
    "Case 4\n(bimodal sym.)": P * (-k1 + theta1 * Y) + (1 - P) * (k1 + theta1 * Yb),
    "Case 5\n(bimodal sym., $k{=}0.25$)": P * (-k5 + theta5 * Y) + (1 - P) * (k5 + theta5 * Yb),
    "Case 6\n(trimodal)":
        (-np.where(np.floor(3 * U) == 0, k6, 0)
         + np.where(np.floor(3 * U) == 1, k6, 0)) + theta6 * Y,
}

# encoder grid bounds
GRID_LO, GRID_HI = -2.0, 2.0

fig, axes = plt.subplots(2, 3, figsize=(11, 5.5))
axes = axes.flatten()

for ax, (title, smp) in zip(axes, samples.items()):
    x_range = max(abs(smp.min()), abs(smp.max())) * 1.3
    x_plot = np.linspace(-x_range, x_range, 600)
    kde = gaussian_kde(smp, bw_method="scott")
    density = kde(x_plot)

    ax.plot(x_plot, density, color="steelblue", lw=1.8)
    ax.fill_between(x_plot, density, alpha=0.25, color="steelblue")

    outside_mask = (x_plot < GRID_LO) | (x_plot > GRID_HI)
    if outside_mask.any():
        ax.fill_between(x_plot, density, where=outside_mask,
                        alpha=0.55, color="firebrick",
                        label="outside grid $[-2,2]$")

    ymax = density.max()
    ax.axvline(GRID_LO, color="firebrick", lw=1.0, ls="--", alpha=0.7)
    ax.axvline(GRID_HI, color="firebrick", lw=1.0, ls="--", alpha=0.7)

    ax.set_title(title, fontsize=8.5)
    ax.set_yticks([])
    ax.set_xlabel("$X_0$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right", "left"]].set_visible(False)

    frac_out = np.mean((smp < GRID_LO) | (smp > GRID_HI))
    if frac_out > 1e-4:
        ax.text(0.97, 0.92, f"{100*frac_out:.1f}% outside grid",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="firebrick")

handles = [
    plt.Line2D([0], [0], color="firebrick", lw=1.0, ls="--", alpha=0.7,
               label="Bin-encoder grid boundary $[-2,2]$"),
    plt.Rectangle((0, 0), 1, 1, fc="firebrick", alpha=0.55,
                  label="Mass outside grid"),
]
fig.legend(handles=handles, loc="lower center", ncol=2,
           fontsize=8, bbox_to_anchor=(0.5, -0.01))

fig.suptitle("Initial distributions for Cases 1–6 (Pham-Warin benchmark)",
             fontsize=11, fontweight="bold", y=1.01)
plt.tight_layout()

out = OUT_DIR / "initial_distributions.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
