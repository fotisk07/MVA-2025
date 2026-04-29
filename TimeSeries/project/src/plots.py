from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np

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

def plot_adjacency_cmap(ax, A, title, cmap="Greys"):
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


def plot_method_cmap(ax, methods,M):
        
    im = ax.imshow(M.values, vmin=0, vmax=1, cmap="inferno")

    ax.set_xticks(range(len(methods)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_yticklabels(methods)

    # annotate values
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(
                j, i,
                f"{M.values[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if M.values[i, j] > 0.6 else "black",
                fontsize=9
            )

    ax.set_title("Pairwise agreement (F1) between formation CPD methods")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("F1 score")

def plot_n_segments(results, xlabel, title):
    """
    results: dict[param_value -> out]
    """
    xs = list(results.keys())
    ys = [results[x]["n_segments"] for x in xs]

    plt.figure(figsize=(5, 3))
    plt.plot(xs, ys, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("# formation segments")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()

def plot_cp_timeline(results, xlabel, title):
    """
    Each row = one parameter value
    """
    plt.figure(figsize=(8, 4))

    for y, (param, out) in enumerate(results.items()):
        cps = out["change_points"]
        plt.vlines(
            cps,
            y - 0.4,
            y + 0.4,
            linewidth=2,
        )

    plt.yticks(
        range(len(results)),
        [str(k) for k in results.keys()],
    )
    plt.xlabel("time")
    plt.ylabel(xlabel)
    plt.title(title)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

def plot_method_cp_timeline(results, title):
    plt.figure(figsize=(8, 4))

    for y, (method, out) in enumerate(results.items()):
        cps = out["change_points"]
        plt.vlines(cps, y - 0.4, y + 0.4, linewidth=2)

    plt.yticks(
        range(len(results)),
        list(results.keys()),
    )
    plt.xlabel("time")
    plt.ylabel("CPD method")
    plt.title(title)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

def generate_heatmap_comparison(ax, gps_data, sb_events, METRIC_X_RANGE,METRIC_Y_RANGE):    
    # Plot 1: GPS Data (Continuous)
    ax1 = ax[0]
    h1 = ax1.hist2d(gps_data['x'], gps_data['y'], bins=50, cmap='inferno', 
                    range=[METRIC_X_RANGE, METRIC_Y_RANGE], cmin=1)
    ax1.set_title("Fitogether GPS Data\n(Continuous 10Hz Signal)")
    ax1.set_aspect('equal')
    ax1.set_xlabel("Length (m)")
    ax1.set_ylabel("Width (m)")
    
    # Plot 2: Event Data (Sparse)
    ax2 = ax[1]
    if sb_events is not None:
        h2 = ax2.hist2d(sb_events['x_metric'], sb_events['y_metric'], bins=50, cmap='inferno', 
                        range=[METRIC_X_RANGE, METRIC_Y_RANGE], cmin=1)
        ax2.set_title("StatsBomb Event Data\n(Sparse Event Stream)")
    else:
        ax2.text(0.5, 0.5, "Data Unavailable", ha='center')
        ax2.set_title("StatsBomb Event Data (Unavailable)")
        
    ax2.set_aspect('equal')
    ax2.set_xlabel("Length (m)")
    
    plt.suptitle("Spatial Density and Signal Structure Comparison", fontsize=14)

    return ax

# Helper function to plot formations
def plot_formation(ax, mu, title, color):
    # Safety check: if data has NaNs or Infs, skip plotting to prevent crash
    if np.isnan(mu).any() or np.isinf(mu).any():
        ax.text(0, 0, "Invalid Data (NaN/Inf)", ha='center', color='red')
        ax.axis('off')
        return

    try:
        tri = Delaunay(mu)
        ax.triplot(mu[:, 0], mu[:, 1], tri.simplices, color='gray', linestyle=':', alpha=0.5)
        ax.scatter(mu[:, 0], mu[:, 1], s=200, c=color, edgecolors='black', zorder=5)
        for i, (x, y) in enumerate(mu):
            ax.text(x, y, str(i), fontsize=12, ha='center', va='center', color='white', fontweight='bold')
    except Exception as e:
        ax.text(0, 0, "Plot Error", ha='center', color='red')
    
    # Pitch Layout
    pitch_length = 105
    pitch_width = 68
    ax.axvline(-pitch_length/2, color='k', lw=2)
    ax.axvline(pitch_length/2, color='k', lw=2)
    ax.axhline(-pitch_width/2, color='k', lw=2)
    ax.axhline(pitch_width/2, color='k', lw=2)
    ax.set_xlim(-(pitch_length/2 + 5), (pitch_length/2 + 5))
    ax.set_ylim(-(pitch_width/2 + 5), (pitch_width/2 + 5))
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Length (m)")
    ax.set_ylabel("Width (m)")