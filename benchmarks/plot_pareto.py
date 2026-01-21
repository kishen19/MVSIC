import argparse
import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_pareto(df, x_col, y_col, min_recall_spacing=0.01):
    """
    Calculates a simplified Pareto front.
    1. De-duplicates by keeping the max QPS for each recall value.
    2. Calculates the true Pareto front.
    3. Simplifies the front by enforcing a minimum recall spacing.
    """
    if df.empty:
        return df

    # 1. De-duplicate: For each unique recall, keep only the row with the max QPS.
    df = df.loc[df.groupby(x_col)[y_col].idxmax()]

    # 2. Sort by Recall DESC, then QPS DESC for Pareto calculation.
    sorted_df = df.sort_values(by=[x_col, y_col], ascending=[False, False])

    # 3. Calculate the "true" Pareto front.
    true_pareto_front = []
    max_qps_so_far = -1
    for _, row in sorted_df.iterrows():
        if row[y_col] > max_qps_so_far:
            true_pareto_front.append(row)
            max_qps_so_far = row[y_col]

    if not true_pareto_front:
        return pd.DataFrame([])

    # 4. Simplify the front to reduce jaggedness (smoothing).
    simplified_front = [true_pareto_front[0]]
    for point in true_pareto_front[1:]:
        # Add point only if it's spaced out enough from the last added point.
        if abs(point[x_col] - simplified_front[-1][x_col]) > min_recall_spacing:
            simplified_front.append(point)

    # 5. Ensure the very last point (highest QPS) is always included.
    last_true_point = true_pareto_front[-1]
    if simplified_front[-1][x_col] != last_true_point[x_col]:
        # If the last point is too close to the one before it, replace the last one.
        if abs(last_true_point[x_col] - simplified_front[-1][x_col]) <= min_recall_spacing:
            simplified_front[-1] = last_true_point
        else:
            simplified_front.append(last_true_point)

    return pd.DataFrame(simplified_front).sort_values(by=x_col)


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
            "axes.labelsize": 20,
            "font.size": 18,
            "legend.fontsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "lines.markersize": 10,
            "figure.titlesize": 24,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle(f"{dataset_name} ($k={k}$)", fontweight="bold")

    plot_configs = [
        {"ax": axes[0, 0], "x": "recall_1_k", "y": "QPS_seq", "title": f"QPS vs. Recall 1@{k}"},
        {"ax": axes[0, 1], "x": "recall_k_k", "y": "QPS_seq", "title": f"QPS vs. Recall {k}@{k}"},
        {"ax": axes[1, 0], "x": "recall_1_k", "y": "QPS_par", "title": f"QPS (Batched) vs. Recall 1@{k}"},
        {"ax": axes[1, 1], "x": "recall_k_k", "y": "QPS_par", "title": f"QPS (Batched) vs. Recall {k}@{k}"},
    ]

    color_map = {
        "mvivf": "#023eff",
        "svh_ivf": "#1ac939",
        "muvera": "#8b2be2",
        "vamana": "#e8050a",
        "fastplaid": "#ffc402",
        "mvivf_flat": "#00d7ff",
        "svh": "#9f4801",
    }
    marker_map = {
        "mvivf": "o",
        "svh_ivf": "X",
        "muvera": "s",
        "vamana": "D",
        "fastplaid": "^",
        "mvivf_flat": "v",
        "svh": "<",
    }
    fallback_palette = sns.color_palette("bright", 10)
    next_color_idx = 0
    available_markers = ["o", "s", "X", "D", "^", "v", "<", ">", "P", "*"]
    next_marker_idx = 7

    lines_for_legend, labels_for_legend = [], []

    # Iterate over subplots first
    for p_config in plot_configs:
        ax = p_config["ax"]
        x_col, y_col = p_config["x"], p_config["y"]

        # Set specific X and Y labels
        if x_col == "recall_1_k":
            ax.set_xlabel(f"Recall 1@{k}")
        else:
            ax.set_xlabel(f"Recall {k}@{k}")

        if y_col == "QPS_seq":
            ax.set_ylabel("QPS (Log Scale)")
        else:
            ax.set_ylabel("QPS (Batched, Log Scale)")

        ax.set_title(p_config["title"], fontsize=20)
        ax.set_yscale("log")
        ax.grid(True, which="major", linestyle="-", alpha=0.6)
        ax.grid(True, which="minor", linestyle=":", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        all_min_recalls_for_subplot = []

        # Now, loop over methods to plot them on the current subplot (ax)
        for i, entry in enumerate(results_to_plot):
            method, build_name = entry[0], entry[1]
            variants = entry[2:] if len(entry) > 2 else [""]
            base_method = entry[0]
            label = _make_label(method, build_name, variants)
            label_lower = label.lower()
            if base_method not in marker_map:
                marker_map[base_method] = available_markers[next_marker_idx % len(available_markers)]
                next_marker_idx += 1
            marker = marker_map[base_method]
            if base_method not in color_map:
                found_color = False
                for key in color_map:
                    if key in base_method:
                        color_map[base_method] = color_map[key]
                        found_color = True
                        break
                if not found_color:
                    color_map[base_method] = fallback_palette[next_color_idx % len(fallback_palette)]
                    next_color_idx += 1
            color = color_map[base_method]
            if "fastscan" in label_lower:
                linestyle = "--"
                linewidth = 4.0
            elif "rabitq" in label_lower:
                linestyle = ":"
                linewidth = 5.0
            elif "pq" in label_lower:
                linestyle = "-."
                linewidth = 4.0
            else:
                linestyle = "-"
                linewidth = 4.0

            search_dir = os.path.join(base_results_dir, method, build_name, f"k={k}")
            all_data_frames = []
            for var in variants:
                fname = f"{var}_results.csv" if var else "results.csv"
                path = os.path.join(search_dir, fname)
                if os.path.exists(path):
                    df_part = pd.read_csv(path)
                    df_part['variant_origin'] = var if var else "base"
                    all_data_frames.append(df_part)
                else:
                    print(f"  Warning: File not found: {path}")
            if not all_data_frames:
                continue
            full_df = pd.concat(all_data_frames, ignore_index=True)

            # Plotting and data collection for xlim
            if x_col in full_df.columns and y_col in full_df.columns:
                pareto_df = calculate_pareto(full_df, x_col, y_col)
                if pareto_df.empty:
                    continue

                # Collect min recall for this line
                all_min_recalls_for_subplot.append(pareto_df[x_col].min())

                (line,) = ax.plot(
                    pareto_df[x_col],
                    pareto_df[y_col],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=label,
                    color=color,
                    markeredgecolor="black",
                    markeredgewidth=0.3,
                    alpha=0.9,
                )
                # Only collect legend items from the first subplot's iteration
                if p_config["ax"] == axes[0, 0]:
                    lines_for_legend.append(line)
                    labels_for_legend.append(label)

        # 3. DYNAMIC XLIM LOGIC: After plotting all lines on the subplot
        if len(all_min_recalls_for_subplot) > 1:
            all_min_recalls_for_subplot.sort()
            second_lowest_min_recall = all_min_recalls_for_subplot[1]
            lower_bound = max(0, second_lowest_min_recall - 0.05)
            ax.set_xlim(left=lower_bound, right=1.01)

    if not lines_for_legend:
        print("No data found to plot.")
        plt.close(fig)
        return

    fig.legend(lines_for_legend, labels_for_legend, loc="lower center", ncol=min(4, len(labels_for_legend)), bbox_to_anchor=(0.5, 0.01), frameon=True)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    out_dir = f"./results/{experiment_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{str(affix)+"_" if affix else ""}{dataset_name}_k={k}_pareto.pdf")

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
