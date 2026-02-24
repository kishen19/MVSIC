import argparse
import json
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


def _join_affixes(*parts: str) -> str:
    """
    Join non-empty affix parts with underscores.
    Ensures empty strings ("") don't create hanging underscores.
    """
    return "_".join([p for p in parts if p])


def _normalize_plot_groups(plots_field):
    """
    Backward-compatible parsing for plot groups.
    """
    if not isinstance(plots_field, list) or len(plots_field) == 0:
        raise ValueError("'plots' must be a non-empty list.")

    if all(isinstance(x, (list, tuple)) for x in plots_field):
        return [{"name": "", "results": plots_field, "k": None}]

    groups = []
    for i, g in enumerate(plots_field):
        if not isinstance(g, dict):
            raise ValueError(f"New-style 'plots' must be a list of dicts; item {i} is {type(g)}")
        if "results" not in g:
            raise ValueError(f"Plot item {i} missing required key 'results'.")
        groups.append({
            "name": g.get("name", ""),
            "results": g["results"],
            "k": g.get("k", None),
        })
    return groups


def generate_plot(dataset_config, k, results_to_plot, experiment_name, affix="", file_format="png", style="default"):
    """
    Generates a QPS vs. Recall Pareto frontier plot.
    """
    dataset_name = dataset_config["name"]
    base_results_dir = dataset_config["results"]

    print(f"--- Generating Pareto plot for: {dataset_name} (k={k}, style={style}) ---")

    # --- Setup Styling ---
    if style == "advanced":
        plt.rcParams.update({
            "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelsize": 20, "font.size": 18, "legend.fontsize": 16,
            "xtick.labelsize": 16, "ytick.labelsize": 16, "lines.markersize": 10,
            "figure.titlesize": 24,
        })
    else: # default style
        plt.rcParams.update({
            "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelsize": 18, "font.size": 16, "legend.fontsize": 14,
            "xtick.labelsize": 14, "ytick.labelsize": 14, "lines.linewidth": 2.5,
            "lines.markersize": 10, "figure.titlesize": 22,
        })

    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle(f"{dataset_name} ($k={k}$)", fontweight="bold")

    plot_configs = [
        {"ax": axes[0, 0], "x": "recall_1_k", "y": "QPS_seq", "title": f"QPS vs. Recall 1@{k}"},
        {"ax": axes[0, 1], "x": "recall_k_k", "y": "QPS_seq", "title": f"QPS vs. Recall {k}@{k}"},
        {"ax": axes[1, 0], "x": "recall_1_k", "y": "QPS_par", "title": f"QPS (Batched) vs. Recall 1@{k}"},
        {"ax": axes[1, 1], "x": "recall_k_k", "y": "QPS_par", "title": f"QPS (Batched) vs. Recall {k}@{k}"},
    ]

    # --- Style-dependent color and marker setup ---
    color_map = {}
    marker_map = {}
    if style == "advanced":
        color_map = { "mvivf": "#023eff", "svh_ivf": "#1ac939", "muvera": "#8b2be2", "vamana": "#e8050a",
                       "fastplaid": "#ffc402", "mvivf_flat": "#00d7ff", "svh": "#9f4801" }
        marker_map = { "mvivf": "o", "svh_ivf": "X", "muvera": "s", "vamana": "D", "fastplaid": "^",
                        "mvivf_flat": "v", "svh": "<" }
    
    fallback_palette = sns.color_palette("bright", len(results_to_plot))
    available_markers = ["o", "s", "X", "D", "^", "v", "<", ">", "P", "*"]
    next_color_idx = 0
    next_marker_idx = 0
    
    lines_for_legend, labels_for_legend = [], []

    for p_config in plot_configs:
        ax, x_col, y_col = p_config["ax"], p_config["x"], p_config["y"]

        ax.set_xlabel(f"Recall 1@{k}" if x_col == "recall_1_k" else f"Recall {k}@{k}")
        ax.set_ylabel("QPS (Log Scale)" if "QPS" in y_col else y_col)
        ax.set_title(p_config["title"], fontsize=20 if style == "advanced" else 15)
        ax.set_yscale("log")
        ax.grid(True, which="major", linestyle="-", alpha=0.6)
        ax.grid(True, which="minor", linestyle=":", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        all_min_recalls_for_subplot = []
        for i, entry in enumerate(results_to_plot):
            method, build_name = entry[0], entry[1]
            variants = entry[2:] if len(entry) > 2 else (entry[2] if len(entry) > 2 else [""])
            base_method = entry[0]
            label = _make_label(method, build_name, variants)
            
            # --- Determine Color, Marker, and Line Style ---
            color, marker, linestyle, linewidth = None, None, None, None
            if style == "advanced":
                if base_method not in marker_map:
                    marker_map[base_method] = available_markers[next_marker_idx % len(available_markers)]
                    next_marker_idx += 1
                marker = marker_map[base_method]

                if base_method not in color_map:
                    # Check for partial match
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
                
                label_lower = label.lower()
                linestyle, linewidth = ("-", 4.0)
                if "fastscan" in label_lower: linestyle, linewidth = ("--", 4.0)
                elif "rabitq" in label_lower: linestyle, linewidth = (":", 5.0)
                elif "pq" in label_lower: linestyle, linewidth = ("-.", 4.0)
            else: # default style
                color = fallback_palette[i]
                marker = available_markers[i % len(available_markers)]
                linestyle = ["-", "--", "-.", ":"][i % 4]
                linewidth = 2.5

            # --- Load Data ---
            search_dir = os.path.join(base_results_dir, method, build_name, f"k={k}")
            all_data_frames = []
            
            # Ensure variants is a list for iteration
            variant_list = variants if isinstance(variants, list) else [variants]

            for var in variant_list:
                fname = f"{var}_results.csv" if var else "results.csv"
                path = os.path.join(search_dir, fname)
                if os.path.exists(path):
                    df_part = pd.read_csv(path)
                    df_part['variant_origin'] = var if var else "base"
                    all_data_frames.append(df_part)
                else:
                    print(f"  Warning: File not found: {path}")
            
            if not all_data_frames: continue
            full_df = pd.concat(all_data_frames, ignore_index=True)

            if x_col in full_df.columns and y_col in full_df.columns:
                pareto_df = calculate_pareto(full_df, x_col, y_col)
                if pareto_df.empty: continue

                all_min_recalls_for_subplot.append(pareto_df[x_col].min())
                (line,) = ax.plot(
                    pareto_df[x_col], pareto_df[y_col], marker=marker, linestyle=linestyle,
                    linewidth=linewidth, label=label, color=color, markeredgecolor="black",
                    markeredgewidth=0.3, alpha=0.9,
                )
                if p_config["ax"] == axes[0, 0]:
                    lines_for_legend.append(line)
                    labels_for_legend.append(label)

        # Dynamic XLIM Logic
        if len(all_min_recalls_for_subplot) > 1:
            all_min_recalls_for_subplot.sort()
            second_lowest_min_recall = all_min_recalls_for_subplot[1]
            ax.set_xlim(left=max(0, second_lowest_min_recall - 0.05), right=1.01)

    if not lines_for_legend:
        print("No data found to plot.")
        plt.close(fig)
        return

    fig.legend(lines_for_legend, labels_for_legend, loc="lower center", ncol=min(4, len(labels_for_legend)),
               bbox_to_anchor=(0.5, 0.01), frameon=True)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    out_dir = f"./results/{experiment_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{_join_affixes(affix, style)}_{dataset_name}_k={k}_pareto.{file_format}")

    print(f"Saving Pareto plot to: {out_file}")
    fig.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _interpolate_qps_at_recall(df, recall_col, qps_col, target_recall):
    """Interpolates QPS for a given target recall from a Pareto front."""
    if df.empty or df[recall_col].isna().all() or df[qps_col].isna().all():
        return np.nan

    df = df.sort_values(by=recall_col).reset_index(drop=True)

    # If an exact match exists, return its QPS.
    exact_match = df[df[recall_col] == target_recall]
    if not exact_match.empty:
        return exact_match[qps_col].iloc[0]

    # Find points that bracket the target_recall.
    lower_points = df[df[recall_col] < target_recall]
    higher_points = df[df[recall_col] > target_recall]

    if lower_points.empty or higher_points.empty:
        return np.nan  # Target recall is outside the range of available data.

    p1 = lower_points.iloc[-1]
    p2 = higher_points.iloc[0]

    r1, q1 = p1[recall_col], p1[qps_col]
    r2, q2 = p2[recall_col], p2[qps_col]

    if r2 == r1:
        return q1

    # Linear interpolation.
    interpolated_qps = q1 + (q2 - q1) * (target_recall - r1) / (r2 - r1)
    return interpolated_qps


def generate_qps_at_recall_analysis(dataset_config, k, results_to_plot, experiment_name, affix=""):
    """
    Calculates estimated QPS at 90% recall levels for both QPS_seq and QPS_par,
    and saves to a CSV file.
    """
    dataset_name = dataset_config["name"]
    base_results_dir = dataset_config["results"]
    target_recall = 0.90  # Focus on 90%

    print(f"--- Generating QPS @ 90% Recall analysis for: {dataset_name} (k={k}) ---")

    analysis_results = []

    for entry in results_to_plot:
        method, build_name = entry[0], entry[1]
        variants = entry[2:] if len(entry) > 2 else (entry[2] if len(entry) > 2 else [""])
        label = _make_label(method, build_name, variants)

        # --- Load Data ---
        search_dir = os.path.join(base_results_dir, method, build_name, f"k={k}")
        all_data_frames = []
        variant_list = variants if isinstance(variants, list) else [variants]

        for var in variant_list:
            fname = f"{var}_results.csv" if var else "results.csv"
            path = os.path.join(search_dir, fname)
            if os.path.exists(path):
                df_part = pd.read_csv(path)
                all_data_frames.append(df_part)
            else:
                print(f"  Warning: File not found for analysis: {path}")

        if not all_data_frames:
            continue
        
        full_df = pd.concat(all_data_frames, ignore_index=True)

        # Index build time from build_stats.json (if present)
        build_time_sec = np.nan
        build_stats_path = os.path.join(base_results_dir, method, build_name, "build_stats.json")
        if os.path.exists(build_stats_path):
            try:
                with open(build_stats_path, "r") as f:
                    build_stats = json.load(f)
                build_time_sec = build_stats.get("build_time_sec", np.nan)
            except (json.JSONDecodeError, TypeError):
                pass

        row_data = {"Method": label, "build_time_sec": build_time_sec}
        
        for recall_col in ["recall_1_k", "recall_k_k"]:
            for y_col in ["QPS_seq", "QPS_par"]:
                col_name = f"{y_col}_at_{int(target_recall*100)}_{recall_col}"
                row_data[col_name] = np.nan

                if recall_col not in full_df.columns or y_col not in full_df.columns:
                    continue

                pareto_df = calculate_pareto(full_df, recall_col, y_col)
                if pareto_df.empty:
                    continue
                
                qps_val = _interpolate_qps_at_recall(pareto_df, recall_col, y_col, target_recall)
                row_data[col_name] = qps_val
        
        analysis_results.append(row_data)

    if not analysis_results:
        print("No data found for QPS @ Recall analysis.")
        return

    results_df = pd.DataFrame(analysis_results)

    # QPS columns (exclude 'Method' and 'build_time_sec') for which we add a slowdown multiplier
    qps_cols = [c for c in results_df.columns if c not in ("Method", "build_time_sec") and "QPS" in c]
    best_per_col = results_df[qps_cols].max(skipna=True)

    # Build new column order: Method, build_time_sec, then each QPS column and its multiplier
    new_cols = ["Method", "build_time_sec"]
    for c in qps_cols:
        new_cols.append(c)
        mult_col = f"{c}_mult"
        # multiplier = best / value (1.0 for best, >1 for slower)
        results_df[mult_col] = np.where(
            results_df[c].notna() & (results_df[c] > 0),
            best_per_col[c] / results_df[c],
            np.nan,
        )
        new_cols.append(mult_col)
    results_df = results_df[new_cols]

    out_dir = f"./results/{experiment_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{_join_affixes(affix)}_{dataset_name}_k={k}_qps_at_90_recall.csv")

    print(f"Saving QPS @ 90% Recall analysis to: {out_file}")
    results_df.to_csv(out_file, index=False, float_format="%.2f")




def main():
    parser = argparse.ArgumentParser(description="Plot QPS vs. Recall Pareto frontiers from benchmark results.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--affix", type=str, default="", help="Optional affix for the filename.")
    parser.add_argument("--style", type=str, default="default", choices=["default", "advanced"], 
                        help="Plotting style: 'default' for simple colors, 'advanced' for method-grouped styling.")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf"], help="Output file format.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    experiment_name = config.get("name", "plots")
    dataset_configs = config.get("datasets") or [config.get("dataset")]
    if not dataset_configs or dataset_configs == [None]:
        raise ValueError("No 'datasets' or 'dataset' key found in the config file.")

    default_k = config["k"]
    plots_field = config.get("plots")
    if plots_field is None:
        raise ValueError("No 'plots' key found in the config file.")
    plot_groups = _normalize_plot_groups(plots_field)

    for dataset_config in dataset_configs:
        for group in plot_groups:
            group_k = group["k"] if group.get("k") is not None else default_k
            group_affix = _join_affixes(args.affix, group.get("name", ""))
            
            generate_plot(
                dataset_config, group_k, group["results"],
                experiment_name, group_affix, args.format, args.style
            )
            generate_qps_at_recall_analysis(
                dataset_config, group_k, group["results"],
                experiment_name, group_affix
            )


if __name__ == "__main__":
    main()