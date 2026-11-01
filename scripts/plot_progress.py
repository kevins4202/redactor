"""Plot progress across development iterations from checked benchmark JSON.

Run after the occlusion-recovery evaluations described in docs/occlusion-recovery.md:
    python scripts/plot_progress.py
"""

import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs/images"

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NEW, OLD, BEFORE = "#2a78d6", "#b8b7b0", "#eb6834"  # validated pair: NEW/BEFORE
HIDDEN = "#ecebe7"

# (iteration label, result file prefix, method). Iterations 1-2 come from the
# final benchmark; 3-4 are the paired SAM 2 rerun on one machine.
ITERATIONS = [("1 · Translation\nbaseline", "final", "lasso"),
              ("2 · Classical\n+ recovery", "final", "classical"),
              ("3 · SAM 2.1\nTiny", "occlusion-recovery", "sam2"),
              ("4 · SAM 2\n+ recovery", "occlusion-recovery", "sam2-recover")]
DATASETS = {"synthetic": "Synthetic stress tests\n(4 scenarios × 3 prompt slacks)",
            "davis": "DAVIS real video\n(8 sequences, 12 objects)",
            "crosswalk": "Crosswalk clip\n(6 hand-labeled frames)"}
METRICS = [("mean_iou", "Mask IoU  (higher is better)", 1, "{:.2f}"),
           ("mean_leak_rate", "Target pixels left uncovered\n(%, lower is better)", 100, "{:.1f}%"),
           ("review_frame_fraction", "Frames flagged for review\n(%, lower is better)", 100, "{:.0f}%")]


def load(prefix, dataset):
    report = json.loads((ROOT / "results" / f"{prefix}-{dataset}.json").read_text())
    if not report.get("complete"):
        raise ValueError(f"{prefix}-{dataset} has not finished")
    return report["runs"]


def runs_for(prefix, dataset, method):
    runs = [run for run in load(prefix, dataset) if run["method"] == method]
    if not runs:
        raise ValueError(f"No {method} runs in {prefix}-{dataset}")
    return runs


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=.8)
    ax.set_axisbelow(True)


def progress_figure():
    fig, axes = plt.subplots(len(METRICS), len(DATASETS), figsize=(13, 10.5), layout="constrained",
                             facecolor=SURFACE)
    for column, (dataset, title) in enumerate(DATASETS.items()):
        groups = [runs_for(prefix, dataset, method) for _, prefix, method in ITERATIONS]
        for row, (key, label, scale, fmt) in enumerate(METRICS):
            ax = axes[row, column]
            style(ax)
            values = [statistics.mean(run[key] for run in group) * scale for group in groups]
            colors = [OLD] * (len(values) - 1) + [NEW]
            ax.bar(np.arange(len(values)), values, color=colors, width=.62)
            top = 1.0 if scale == 1 else max(100, max(values) * 1.15)
            ax.set_ylim(0, top * 1.12)
            for x, value in enumerate(values):
                ax.text(x, value + top * .02, fmt.format(value), ha="center", va="bottom", fontsize=10,
                        color=INK, fontweight="bold" if x == len(values) - 1 else "normal")
            ax.set_xticks(np.arange(len(values)), [name for name, *_ in ITERATIONS] if row == len(METRICS) - 1 else [],
                          fontsize=9, color=INK_2)
            if column == 0:
                ax.set_ylabel(label, color=INK, fontsize=10)
        axes[0, column].set_title(title, color=INK, fontsize=11)
    fig.suptitle("Tracking quality across development iterations", color=INK, fontsize=16, x=.01, ha="left")
    fig.text(.01, -.015, "Raw tracking masks, before the full-frame review guard. Iterations 3 and 4 were rerun together "
             "on one machine so the comparison is paired. Blue = current iteration.", color=INK_2, fontsize=9)
    fig.savefig(IMAGES / "iteration-progress.png", dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def occlusion_figure():
    runs = load("occlusion-recovery", "synthetic")
    pick = {run["method"]: run for run in runs if run["sequence"] == "occlusion" and run["slack"] == 5}
    fig, ax = plt.subplots(figsize=(11, 4.6), layout="constrained", facecolor=SURFACE)
    style(ax)
    frames = [row["frame"] for row in pick["sam2"]["per_frame"]]
    hidden = [row["frame"] for row in pick["sam2"]["per_frame"] if row["visible_pixels"] == 0]
    if hidden:
        ax.axvspan(min(hidden) - .5, max(hidden) + .5, color=HIDDEN, zorder=0)
        ax.text((min(hidden) + max(hidden)) / 2, 104, "fully behind\npillar", ha="center", va="bottom",
                fontsize=9, color=INK_2)
    for method, color, name in (("sam2", BEFORE, "Iteration 3 · SAM 2"),
                                ("sam2-recover", NEW, "Iteration 4 · SAM 2 + recovery")):
        leak = [np.nan if row["leak_rate"] is None else row["leak_rate"] * 100 for row in pick[method]["per_frame"]]
        ax.plot(frames, leak, color=color, linewidth=2, drawstyle="steps-mid", label=name)
        summary = pick[method]
        ax.text(frames[-1] + 1, leak[-1], f"{name}\n{summary['mean_leak_rate']:.1%} mean leak · "
                f"longest exposure {summary['longest_leak_frames']} frames",
                color=INK, fontsize=9, va="center")
    rows = pick["sam2-recover"]["per_frame"]
    for row in rows:
        if row["state"] == "recovered":
            y = row["leak_rate"] * 100
            ax.scatter([row["frame"]], [y], s=70, color=NEW, edgecolor=SURFACE, linewidth=2, zorder=4)
            ax.annotate("re-detected here;\nSAM 2 re-prompted", (row["frame"], y), xytext=(row["frame"] - 1.5, 30),
                        ha="right", fontsize=9, color=INK,
                        arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": .8, "shrinkB": 5})
    if hidden:
        ax.text(min(hidden) / 2, 8, "Both runs identical until the target reappears", ha="center",
                fontsize=9, color=INK_2)
    ax.set_xlim(0, frames[-1] + 1)
    ax.set_ylim(-3, 118)
    ax.set_xlabel("Frame", color=INK_2)
    ax.set_ylabel("Target pixels left uncovered (%)", color=INK)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title("Synthetic occlusion clip: SAM 2 never re-acquires the target until recovery re-prompts it",
                 color=INK, fontsize=12, loc="left")
    fig.savefig(IMAGES / "occlusion-timeline.png", dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    IMAGES.mkdir(exist_ok=True)
    progress_figure()
    occlusion_figure()
    print("Wrote docs/images/{iteration-progress,occlusion-timeline}.png")


if __name__ == "__main__":
    main()
