"""Publication figures generated exclusively from saved, complete condition metrics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from ..lofo import FAMILIES
from ..utils import atomic_write, save_json
from .checkpoints import MODELS
from .data import CORRUPTIONS
from .metrics import aggregate

LABELS = {"clean_baseline": "Clean baseline", **{f"lofo_{f}": f"LOFO-{f.title()}" for f in FAMILIES}}
COLORS = ("#414141", "#0072B2", "#D55E00", "#009E73", "#CC79A7")


def plot_saved(output):
    output = Path(output)
    saved = json.loads((output / "metrics/evaluation.json").read_text())
    if not saved["summary"]["benchmark_complete"] or saved["summary"]["mode"] != "full":
        raise ValueError("Publication figures require a complete scientific full evaluation.")
    result = aggregate(saved["conditions"])
    for key, value in result.items():
        if value != saved["aggregates"][key]:
            raise ValueError("Saved aggregates differ from saved condition metrics.")
    destination = output / "plots"
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    produced = []
    def save(fig, name, rect=None):
        fig.tight_layout(rect=rect)
        for extension in ("svg", "png"):
            path = destination / f"{name}.{extension}"
            atomic_write(path, lambda p: fig.savefig(p, format=extension, dpi=300, bbox_inches="tight"))
            produced.append(str(path.relative_to(output)))
        plt.close(fig)

    matrix = np.array([[next(r["accuracy_pct"] for r in result["corruptions"]
                             if r["model"] == m and r["corruption"] == c) for c in CORRUPTIONS] for m in MODELS])
    fig, ax = plt.subplots(figsize=(15, 5.2))
    im = ax.imshow(matrix, vmin=0, vmax=100, cmap="viridis", aspect="auto")
    ax.set_xticks(range(15), [c.replace("_", " ") for c in CORRUPTIONS], rotation=45, ha="right")
    ax.set_yticks(range(5), [LABELS[m] for m in MODELS])
    for i in range(5):
        for j in range(15):
            ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if matrix[i,j] < 48 else "black")
    left = 0
    for family, names in FAMILIES.items():
        ax.text(left+(len(names)-1)/2, 1.03, family.title(), transform=ax.get_xaxis_transform(), ha="center")
        left += len(names)
        if left < 15:
            ax.axvline(left-0.5, color="white", lw=2)
    ax.set_title("Corruption accuracy averaged over severities 1–5", pad=35)
    fig.colorbar(im, ax=ax, label="Accuracy (%)", shrink=.75)
    save(fig, "model_corruption_heatmap")

    held = [r for r in result["held_out_transfer"] if r["scope"] == "family"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(4)
    ax.bar(x-.19, [r["baseline_accuracy_pct"] for r in held], .38, color=COLORS[0], label="Clean baseline")
    ax.bar(x+.19, [r["held_out_accuracy_pct"] for r in held], .38, color=COLORS[1], label="Corresponding held-out LOFO")
    for i, r in enumerate(held):
        ax.text(i, max(r["held_out_accuracy_pct"], r["baseline_accuracy_pct"])+2,
                f"{r['difference_pp']:+.2f} pp", ha="center")
    ax.set(xticks=x, xticklabels=[r["family"].title() for r in held], ylim=(0,100), ylabel="Family accuracy (%)",
           title="Held-out family transfer relative to clean-only training")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.1), ncol=2, frameon=False)
    save(fig, "held_out_transfer")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for ax, family in zip(axes.flat, FAMILIES):
        for m, color in zip(MODELS, COLORS):
            points = [r["accuracy_pct"] for r in result["severity"] if r["model"] == m and r["family"] == family]
            ax.plot(range(1,6), points, marker="o", color=color, label=LABELS[m], linewidth=1.7)
        ax.set(title=family.title(), xticks=range(1,6), ylim=(0,100), xlabel="Severity", ylabel="Accuracy (%)")
        ax.grid(alpha=.2)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0,.075,1,1))
    # Keep lower legend margin when save invokes tight_layout again.
    fig.subplots_adjust(bottom=.15)
    save(fig, "severity_curves", rect=(0,.09,1,1))

    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(5)
    ax.bar(x-.19, [r["clean_accuracy_pct"] for r in result["overall"]], .38, color=COLORS[0], label="Clean test")
    ax.bar(x+.19, [r["mean_corrupted_accuracy_pct"] for r in result["overall"]], .38,
           color=COLORS[1], label="75-condition macro mean")
    ax.set(xticks=x, xticklabels=[LABELS[m] for m in MODELS], ylim=(0,100), ylabel="Accuracy (%)",
           title="Clean accuracy and overall corruption accuracy")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "overall_robustness")

    delta = np.array([[next(r["baseline_difference_pp"] for r in result["families"]
                            if r["model"] == m and r["family"] == f) for m in MODELS] for f in FAMILIES])
    limit = max(1, float(np.abs(delta).max()))
    fig, ax = plt.subplots(figsize=(9,5.5))
    im = ax.imshow(delta, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(5), [LABELS[m] for m in MODELS], rotation=20, ha="right")
    ax.set_yticks(range(4), [f.title() for f in FAMILIES])
    for i, f in enumerate(FAMILIES):
        for j, m in enumerate(MODELS):
            held_out = m == f"lofo_{f}"
            ax.text(j, i, f"{delta[i,j]:+.2f}"+("\nheld out" if held_out else ""), ha="center", va="center",
                    color="white" if abs(delta[i,j]) > .6*limit else "black")
            if held_out:
                ax.add_patch(Rectangle((j-.48,i-.48),.96,.96,fill=False,edgecolor="black",linestyle="--",linewidth=2))
    ax.set_title("Family accuracy improvement relative to clean baseline")
    fig.colorbar(im, ax=ax, label="Difference (percentage points)")
    save(fig, "family_transfer_matrix")
    save_json(destination / "index.json", {"source": "metrics/evaluation.json", "files": produced})
