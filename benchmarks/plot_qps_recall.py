import argparse
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _make_label(method: str, build_name: str, search_variant: str) -> str:
    """
    Legend label:
      - If build_name already starts with method, don't repeat method.
      - Append search_variant if present.
    """
    base = build_name if build_name.startswith(method) else f"{method}_{build_name}"
    return base + (f"_{search_variant}" if search_variant else "")


def _join_affixes(*parts: str) -> str:
    """
    Join non-empty affix parts with underscores.
    Ensures empty strings ("") don't create hanging underscores.
    """
    return "_".join([p for p in parts if p])


def _normalize_plot_groups(plots_field):
    """
    Backward-compatible parsing for plot groups.

    Supported formats:

    NEW style (recommended):
      plots:
        - name: foo
          results: [...]
        - name: ""         # allowed
          k: 50            # optional override
          results: [...]

    LEGACY style (still supported if you want it):
      plots:
        - [method, build, variant]
        - [method, build, variant]
      -> treated as a single unnamed plot group
    """
    if not isinstance(plots_field, list) or len(plots_field) == 0:
        raise ValueError("'plots' must be a non-empty list.")

    # Legacy style: list of triples/lists
    if all(isinstance(x, (list, tuple)) for x in plots_field):
        return [{"name": "", "results": plots_field, "k": None}]

    # New style: list of dicts
    groups = []
    for i, g in enumerate(plots_field):
        if not isinstance(g, dict):
            raise ValueError(f"New-style 'plots' must be a list of dicts; item {i} is {type(g)}")
        if "results" not in g:
            raise ValueError(f"Plot item {i} missing required key 'results'.")
        groups.append(
            {
                "name": g.get("name", ""),
                "results": g["results"],
                "k": g.get("k", None),  # optional override
            }
        )
    return groups


def generate_plot_for_dataset(dataset_config, k, results_to_plot, experiment_name, affix=""):
    """
    Generates a beautified QPS vs. Recall plot using the original data logic.
    """
    dataset_name = dataset_config["name"]
    base_results_dir = dataset_config["results"]

    print(f"--- Generating plots for dataset: {dataset_name} (k={k}) ---")

    # --- Setup Scientific Styling ---
    # Using a clean serif font and high-resolution settings
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelsize": 18,
            "font.size": 16,
            "legend.fontsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "lines.linewidth": 2.5,
            "lines.markersize": 10,
            "figure.titlesize": 22,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle(f"QPS vs. Recall for {dataset_name} ($k={k}$)", fontweight="bold")

    # Colorblind-friendly high-contrast palette
    palette = sns.color_palette("bright", len(results_to_plot))
    markers = ["o", "s", "X", "D", "^", "v", "<", ">"]
    linestyles = ["-", "--", "-.", ":"]

    plot_configs = [
        {"ax": axes[0, 0], "x": "recall_1_k", "y": "QPS_seq", "title": f"QPS_seq vs. Recall 1@{k}"},
        {"ax": axes[0, 1], "x": "recall_k_k", "y": "QPS_seq", "title": f"QPS_seq vs. Recall {k}@{k}"},
        {"ax": axes[1, 0], "x": "recall_1_k", "y": "QPS_par", "title": f"QPS_par vs. Recall 1@{k}"},
        {"ax": axes[1, 1], "x": "recall_k_k", "y": "QPS_par", "title": f"QPS_par vs. Recall {k}@{k}"},
    ]

    # Initialize subplots with grids and labels
    for p_config in plot_configs:
        ax = p_config["ax"]
        ax.set_xlabel("Recall", labelpad=10)
        ax.set_ylabel("QPS", labelpad=10)
        ax.set_title(p_config["title"], pad=15)
        ax.set_yscale("log")
        ax.grid(True, which="major", linestyle="-", alpha=0.4, color="gray")
        ax.grid(True, which="minor", linestyle=":", alpha=0.2, color="gray")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # --- Load and Plot Data ---
    lines = []
    labels = []

    for i, result_info in enumerate(results_to_plot):
        method, build_name, search_variant = result_info
        search_dir = os.path.join(base_results_dir, method, build_name, f"k={k}")
        result_filename = f"{search_variant}_results.csv" if search_variant else "results.csv"
        result_path = os.path.join(search_dir, result_filename)

        if not os.path.exists(result_path):
            print(f"  Warning: Result file not found at '{result_path}'")
            continue

        print(f"  Loading results from: {result_path}")
        try:
            df = pd.read_csv(result_path)
            label = _make_label(method, build_name, search_variant)

            # Original sorting logic
            if "nprobes" in df.columns:
                df = df.sort_values(by="nprobes").reset_index(drop=True)
            elif "L" in df.columns:
                df = df.sort_values(by="L").reset_index(drop=True)

            color = palette[i]
            marker = markers[i % len(markers)]
            linestyle = linestyles[i % len(linestyles)]

            for p_config in plot_configs:
                ax = p_config["ax"]
                x_col, y_col = p_config["x"], p_config["y"]

                if x_col in df.columns and y_col in df.columns:
                    # Plotting every point exactly as in the original script
                    (line,) = ax.plot(
                        df[x_col],
                        df[y_col],
                        marker=marker,
                        linestyle=linestyle,
                        label=label,
                        color=color,
                        markeredgecolor="white",
                        markeredgewidth=1.0,
                        alpha=0.9,
                    )

                    if p_config["ax"] == axes[0, 0]:
                        lines.append(line)
                        labels.append(label)
                else:
                    print(f"  Warning: Columns '{x_col}' or '{y_col}' not found for '{label}'.")
        except Exception as e:
            print(f"  Error reading {result_path}: {e}")

    if not lines:
        print("No data loaded. Exiting.")
        plt.close(fig)
        return

    # --- Create a single, centralized legend ---
    fig.legend(
        lines,
        labels,
        loc="lower center",
        ncol=min(3, len(labels)),
        bbox_to_anchor=(0.5, 0.02),
        frameon=True,
        edgecolor="0.8",
    )

    # --- Save Plot ---
    affix_str = f"{affix}_" if affix else ""
    output_dir = f"./results/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{output_dir}/{affix_str}{dataset_name}_k={k}_qps_vs_recall.pdf"

    # Adjust layout to prevent clipping (leaving space for legend at bottom)
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])

    print(f"Saving plot to: {output_filename}")
    fig.savefig(output_filename, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_qps_vs_recall(config_path, affix=""):
    with open(config_path, "r") as f:
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
            group_affix = _join_affixes(affix, group.get("name", ""))
            generate_plot_for_dataset(
                dataset_config,
                group_k,
                group["results"],
                experiment_name,
                group_affix,
            )


def main():
    parser = argparse.ArgumentParser(description="Plot QPS vs. Recall from benchmark results.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--affix", type=str, default="", help="Optional affix for the filename.")
    args = parser.parse_args()
    plot_qps_vs_recall(args.config, args.affix)


if __name__ == "__main__":
    main()
