import argparse
import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_pareto(df, x_col, y_col):
    """
    Standard Pareto: For a given recall, what is the maximum QPS?
    As Recall decreases, QPS should increase.
    """
    if df.empty:
        return df

    # 1. Sort by Recall DESCENDING.
    # For a fixed Recall, higher QPS comes first.
    sorted_df = df.sort_values(by=[x_col, y_col], ascending=[False, False])

    pareto_front = []
    curr_max_y = -1

    # 2. Iterate from right to left (high recall to low recall)
    # We only keep a point if its QPS is better than anything seen
    # at a HIGHER recall level.
    for _, row in sorted_df.iterrows():
        if row[y_col] > curr_max_y:
            pareto_front.append(row)
            curr_max_y = row[y_col]

    # Sort back to ascending X for proper line plotting
    return pd.DataFrame(pareto_front).sort_values(by=x_col)


def _make_label(method, build_name, variants):
    """Generates a clean legend label."""
    base = build_name if build_name.startswith(method) else f"{method}_{build_name}"
    # If it's a list of multiple variants, just label the base
    if isinstance(variants, list) and len(variants) > 1:
        return f"{base} (Pareto)"
    # If it's a single variant (string or list of 1)
    v = variants[0] if isinstance(variants, list) else variants
    return f"{base}_{v}" if v else base


def _join_affixes(*parts):
    return "_".join([p for p in parts if p])


def _normalize_plot_groups(plots_field):
    if not isinstance(plots_field, list) or len(plots_field) == 0:
        raise ValueError("'plots' must be a non-empty list.")
    if all(isinstance(x, (list, tuple)) for x in plots_field):
        return [{"name": "", "results": plots_field, "k": None}]
    groups = []
    for i, g in enumerate(plots_field):
        if not isinstance(g, dict):
            raise ValueError(f"Items must be dicts; item {i} is {type(g)}")
        groups.append(
            {
                "name": g.get("name", ""),
                "results": g["results"],
                "k": g.get("k", None),
            }
        )
    return groups


def generate_pareto_plot(dataset_config, k, results_to_plot, experiment_name, affix=""):
    dataset_name = dataset_config["name"]
    base_results_dir = dataset_config["results"]

    print(f"--- Generating Pareto plots for: {dataset_name} (k={k}) ---")

    # --- Scientific Styling ---
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelsize": 18,
            "font.size": 16,
            "legend.fontsize": 13,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
            "figure.titlesize": 20,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle(f"Pareto Optimized QPS vs. Recall\n{dataset_name} ($k={k}$)", fontweight="bold")

    plot_configs = [
        {"ax": axes[0, 0], "x": "recall_1_k", "y": "QPS_seq", "title": f"Seq QPS vs. Recall 1@{k}"},
        {"ax": axes[0, 1], "x": "recall_k_k", "y": "QPS_seq", "title": f"Seq QPS vs. Recall {k}@{k}"},
        {"ax": axes[1, 0], "x": "recall_1_k", "y": "QPS_par", "title": f"Par QPS vs. Recall 1@{k}"},
        {"ax": axes[1, 1], "x": "recall_k_k", "y": "QPS_par", "title": f"Par QPS vs. Recall {k}@{k}"},
    ]

    for p in plot_configs:
        ax = p["ax"]
        ax.set_xlabel("Recall")
        ax.set_ylabel("Queries Per Second (Log Scale)")
        ax.set_title(p["title"])
        ax.set_yscale("log")
        ax.grid(True, which="major", linestyle="-", alpha=0.4)
        ax.grid(True, which="minor", linestyle=":", alpha=0.2)

    palette = sns.color_palette("bright", len(results_to_plot))
    lines, labels = [], []

    for i, entry in enumerate(results_to_plot):
        method, build_name = entry[0], entry[1]
        # Variants are all elements from index 2 onwards
        variants = entry[2:] if len(entry) > 2 else [""]

        search_dir = os.path.join(base_results_dir, method, build_name, f"k={k}")
        all_data_frames = []

        for var in variants:
            fname = f"{var}_results.csv" if var else "results.csv"
            path = os.path.join(search_dir, fname)
            if os.path.exists(path):
                df_part = pd.read_csv(path)
                # Keep track of which variant this point came from
                df_part['variant_origin'] = var if var else "base"
                all_data_frames.append(df_part)
            else:
                print(f"  Warning: File not found: {path}")

        if not all_data_frames:
            continue

        full_df = pd.concat(all_data_frames, ignore_index=True)
        label = _make_label(method, build_name, variants)
        color = palette[i]

        for p_idx, p_config in enumerate(plot_configs):
            ax = p_config["ax"]
            x_col, y_col = p_config["x"], p_config["y"]

            if x_col in full_df.columns and y_col in full_df.columns:
                # Filter Pareto points
                pareto_df = calculate_pareto(full_df, x_col, y_col)

                # Plot the background "cloud" of all points (lightly)
                ax.scatter(full_df[x_col], full_df[y_col], color=color, alpha=0.15, s=15, edgecolors='none')

                # Plot the Pareto Frontier line
                (line,) = ax.plot(
                    pareto_df[x_col],
                    pareto_df[y_col],
                    marker='o',
                    linestyle='-',
                    label=label,
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                )

                if p_idx == 0:
                    lines.append(line)
                    labels.append(label)

    if not lines:
        print("No data found to plot.")
        plt.close(fig)
        return

    fig.legend(lines, labels, loc="lower center", ncol=min(3, len(labels)), bbox_to_anchor=(0.5, 0.02), frameon=True)
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])

    out_dir = f"./results/{experiment_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{affix}_{dataset_name}_k={k}_pareto.pdf")

    print(f"Saving Pareto plot to: {out_file}")
    fig.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_pareto_main(config_path, affix=""):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    experiment_name = config.get("name", "pareto_study")
    dataset_configs = config.get("datasets") or [config.get("dataset")]
    default_k = config.get("k", 10)

    plot_groups = _normalize_plot_groups(config.get("plots", []))

    for dataset_cfg in dataset_configs:
        for group in plot_groups:
            group_k = group["k"] if group.get("k") is not None else default_k
            group_affix = _join_affixes(affix, group.get("name", ""))
            generate_pareto_plot(dataset_cfg, group_k, group["results"], experiment_name, group_affix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Pareto Frontier QPS vs. Recall.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--affix", type=str, default="", help="Optional filename affix.")
    args = parser.parse_args()
    plot_pareto_main(args.config, args.affix)
