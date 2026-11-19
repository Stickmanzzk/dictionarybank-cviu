import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


PAPER = Path(__file__).resolve().parent
RESULTS = PAPER / "experiments" / "results"
OUT = PAPER / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLORS = {
    "blue": "#3778B8",
    "orange": "#E17B2E",
    "green": "#3A9A72",
    "purple": "#7962B8",
    "gray": "#6E7781",
    "light": "#D8DEE7",
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rounded_box(ax, xy, width, height, title, subtitle, color):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=0.8,
        edgecolor=color,
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.64,
        title,
        ha="center",
        va="center",
        color="white",
        fontsize=8,
        fontweight="bold",
    )
    ax.text(
        x + width / 2,
        y + height * 0.27,
        subtitle,
        ha="center",
        va="center",
        color="white",
        fontsize=6.5,
    )


def arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "linewidth": 0.9,
            "color": COLORS["gray"],
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )


def overview():
    fig, ax = plt.subplots(figsize=(7.15, 2.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rounded_box(
        ax,
        (0.01, 0.37),
        0.14,
        0.29,
        "Frozen features",
        "CLIP | SigLIP | DINOv2",
        COLORS["blue"],
    )
    rounded_box(
        ax,
        (0.19, 0.43),
        0.17,
        0.17,
        "Dictionary 1",
        "4,096 candidates",
        COLORS["purple"],
    )
    rounded_box(
        ax,
        (0.19, 0.22),
        0.17,
        0.17,
        "Dictionary K",
        "independent training",
        COLORS["purple"],
    )
    rounded_box(
        ax,
        (0.41, 0.35),
        0.17,
        0.32,
        "Global sparse bank",
        "K x 4,096 candidates",
        COLORS["green"],
    )
    rounded_box(
        ax,
        (0.63, 0.35),
        0.15,
        0.32,
        "Top-64",
        "one active support",
        COLORS["orange"],
    )
    rounded_box(
        ax,
        (0.83, 0.35),
        0.15,
        0.32,
        "Prototype",
        "source-centered",
        COLORS["blue"],
    )

    for y in (0.515, 0.305):
        arrow(ax, (0.15, 0.515), (0.185, y))
    arrow(ax, (0.36, 0.515), (0.405, 0.515))
    arrow(ax, (0.36, 0.305), (0.405, 0.305))
    arrow(ax, (0.58, 0.51), (0.625, 0.51))
    arrow(ax, (0.78, 0.51), (0.825, 0.51))

    ax.text(
        0.54,
        0.12,
        "candidate width grows with K; active support remains 64",
        ha="center",
        va="center",
        color=COLORS["gray"],
        fontsize=6.6,
    )
    ax.text(
        0.90,
        0.23,
        "average three streams",
        ha="center",
        va="center",
        color=COLORS["gray"],
        fontsize=6.2,
    )
    fig.tight_layout(pad=0.08)
    fig.savefig(OUT / "dictionary_bank_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def domainnet_gains(aggregate, reviewer):
    domains = [
        "Clipart",
        "Infograph",
        "Painting",
        "Quickdraw",
        "Real",
        "Sketch",
    ]
    single = 100.0 * np.array([
        np.mean([
            reviewer["per_target"][f"single_seed{seed}"][str(target)]
            for seed in range(3)
        ])
        for target in range(6)
    ])
    ensemble = 100.0 * np.array([
        reviewer["per_target"]["single_ensemble"][str(target)]
        for target in range(6)
    ])
    wide = 100.0 * np.array([
        reviewer["per_target"]["wide_k64"][str(i)]
        for i in range(6)
    ])
    bank = 100.0 * np.array([
        reviewer["per_target"]["bank_full"][str(i)]
        for i in range(6)
    ])
    x = np.arange(len(domains))
    width = 0.20

    fig, ax = plt.subplots(figsize=(7.1, 2.55))
    ax.bar(
        x - 1.5 * width,
        single,
        width,
        label="Single mean",
        color=COLORS["gray"],
        alpha=0.9,
    )
    ax.bar(
        x - 0.5 * width,
        ensemble,
        width,
        label="Ensemble",
        color=COLORS["orange"],
        alpha=0.9,
    )
    ax.bar(
        x + 0.5 * width,
        wide,
        width,
        label="Wide 192/64",
        color=COLORS["green"],
        alpha=0.9,
    )
    ax.bar(
        x + 1.5 * width,
        bank,
        width,
        label="DictionaryBank",
        color=COLORS["blue"],
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylim(15, 92)
    ax.grid(axis="y", color=COLORS["light"], linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(pad=0.15)
    fig.savefig(OUT / "domainnet_gains.pdf", bbox_inches="tight")
    plt.close(fig)


def budget_controls(aggregate, reviewer):
    scaling = aggregate["domainnet_content_only_scaling"]
    values = 100.0 * np.array([
        scaling["k1_single"],
        scaling["k2_bank"],
        scaling["k3_bank"],
    ])
    controls = {
        "Single seed 0\n(k=64)": 100.0 * scaling["k1_single"],
        "Ensemble\n(3 paths)": 100.0 * reviewer["macro"]["single_ensemble"],
        "Bank\n(k=64)": 100.0 * scaling["k3_bank"],
        "Wide\ntrain 192 / infer 64": 100.0 * reviewer["macro"]["wide_k64"],
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.25),
        gridspec_kw={"width_ratios": [1.0, 1.3]},
    )
    ax = axes[0]
    ax.plot(
        np.arange(1, 4),
        values,
        marker="o",
        markersize=4,
        linewidth=1.3,
        color=COLORS["green"],
    )
    for k, value in zip((1, 2, 3), values):
        ax.text(k, value + 0.06, f"{value:.2f}", ha="center", va="bottom")
    ax.set_xlabel("Number of dictionaries")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks((1, 2, 3))
    ax.set_ylim(67.3, 68.1)
    ax.grid(color=COLORS["light"], linewidth=0.5)
    ax.set_axisbelow(True)

    ax = axes[1]
    labels = list(controls)
    values = np.array(list(controls.values()))
    colors = [
        COLORS["gray"],
        COLORS["orange"],
        COLORS["blue"],
        COLORS["purple"],
    ]
    bars = ax.barh(labels, values - 67.3, left=67.3, color=colors)
    ax.set_xlim(67.3, 68.05)
    ax.set_xlabel("Accuracy (%)")
    ax.grid(axis="x", color=COLORS["light"], linewidth=0.5)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(
            value + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=7,
        )

    fig.tight_layout(pad=0.25)
    fig.savefig(OUT / "budget_controls.pdf", bbox_inches="tight")
    plt.close(fig)


def support_allocation(stage11_support):
    domains = [
        "Clipart",
        "Infograph",
        "Painting",
        "Quickdraw",
        "Real",
        "Sketch",
    ]
    streams = ("clip", "siglip", "dino")
    fractions = np.zeros((3, 6))
    entropy = np.zeros((3, 6))
    for target in range(6):
        record = stage11_support[str(target)]["streams"]
        for member in range(3):
            fractions[member, target] = np.mean([
                record[stream]["member_selection_fraction"][member]
                for stream in streams
            ])
        for stream_index, stream in enumerate(streams):
            entropy[stream_index, target] = record[stream][
                "mean_support_entropy"
            ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.1, 2.6),
        gridspec_kw={"width_ratios": [1.1, 1.0]},
    )
    x = np.arange(6)
    bottom = np.zeros(6)
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    for member in range(3):
        axes[0].bar(
            x,
            fractions[member],
            bottom=bottom,
            color=colors[member],
            label=f"Dictionary {member + 1}",
            width=0.68,
        )
        bottom += fractions[member]
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(domains, rotation=25, ha="right")
    axes[0].set_ylabel("Selected-atom fraction")
    axes[0].set_ylim(0, 1)
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=3,
        frameon=False,
    )
    axes[0].grid(axis="y", color=COLORS["light"], linewidth=0.5)
    axes[0].set_axisbelow(True)

    for stream_index, stream in enumerate(streams):
        axes[1].plot(
            x,
            entropy[stream_index],
            marker="o",
            markersize=3.5,
            linewidth=1.1,
            color=colors[stream_index],
            label=stream.upper() if stream != "siglip" else "SigLIP",
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(domains, rotation=25, ha="right")
    axes[1].set_ylabel("Normalized support entropy")
    axes[1].set_ylim(0.94, 1.0)
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].grid(axis="y", color=COLORS["light"], linewidth=0.5)
    axes[1].set_axisbelow(True)
    fig.tight_layout(pad=0.25)
    fig.savefig(OUT / "support_allocation.pdf", bbox_inches="tight")
    plt.close(fig)


def rvl_ci(stage10):
    order = (
        ("single_seed0", "vs seed 0"),
        ("single_seed1", "vs seed 1"),
        ("single_seed2", "vs seed 2"),
        ("wide192", "vs wide SAE (k=192)"),
        ("repeated_seed0_top192", "vs repeated"),
    )
    rows = [
        (
            label,
            stage10["comparisons"][f"bank_full_vs_{key}"]["delta_points"],
            stage10["comparisons"][f"bank_full_vs_{key}"]["bootstrap_95_ci_points"],
        )
        for key, label in order
    ]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(3.5, 2.15))
    for index, (_, delta, (low, high)) in enumerate(rows):
        ax.plot([low, high], [index, index], color=COLORS["blue"], linewidth=1.4)
        ax.plot(delta, index, marker="o", markersize=4, color=COLORS["blue"])
        ax.text(
            high + 0.06,
            index,
            f"{delta:+.2f}",
            va="center",
            fontsize=6.5,
        )
    ax.axvline(0.0, color=COLORS["gray"], linestyle="--", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy difference (points)")
    ax.set_xlim(-1.15, 1.95)
    ax.grid(axis="x", color=COLORS["light"], linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.15)
    fig.savefig(OUT / "rvl_ci.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    aggregate = load_json(
        RESULTS
        / "09_cviu_controls"
        / "stage09_aggregate_v2.json"
    )
    stage10 = load_json(
        RESULTS / "10_rvl_q1_controls" / "summary.json"
    )
    reviewer = load_json(
        RESULTS / "11_reviewer_controls" / "ensemble_analysis.json"
    )
    support = load_json(
        RESULTS / "11_reviewer_controls" / "support_diagnostics.json"
    )
    domainnet_gains(aggregate, reviewer)
    budget_controls(aggregate, reviewer)
    support_allocation(support)
    rvl_ci(stage10)
