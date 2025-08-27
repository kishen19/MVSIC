import argparse
import csv
import json
import os
import re

import matplotlib.pyplot as plt


def update_json_from_csvs(directory_path, file_type, output_json_path):
    """
    Scans a directory for specific CSV files, parses them, and updates a JSON file.

    It loads data from an existing JSON file, updates it with new data from CSVs,
    and saves it back. If the JSON file doesn't exist, it will be created.

    Args:
        directory_path (str): The path to the directory containing the CSV files.
        file_type (str): The type of files to process (e.g., 'mvivf', 'svh').
        output_json_path (str): The path to the output JSON file to create or update.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return

    # --- Step 1: Load existing data from the JSON file ---
    collected_data = {}
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r') as f:
                if os.path.getsize(output_json_path) > 0:
                    collected_data = json.load(f)
                else:
                    print(f"Info: JSON file '{output_json_path}' is empty. Starting fresh.")
        except json.JSONDecodeError:
            print(
                f"Warning: Could not decode JSON from '{output_json_path}'. Starting with an empty dataset."
            )
        except IOError as e:
            print(f"Error reading JSON file '{output_json_path}': {e}")
            return
    else:
        print(f"Info: JSON file '{output_json_path}' not found. A new file will be created.")

    # --- Step 2: Define regex patterns based on file_type ---
    patterns = {
        "mvivf": re.compile(r"mvivf_maxsize\d+_iters(\d+)_k=(\d+)\.csv"),
        "svh": re.compile(r"svh_maxsize\d+_k=(\d+)_cand=(\d+)\.csv"),
        "vamana": re.compile(r"vamana_R(\d+)_k=(\d+)\.csv"),
        "muvera": re.compile(r"muvera_fde(\d+)_k=(\d+)\.csv"),
        "mpv": re.compile(r"mpv_k=(\d+)\.csv"),
    }

    pattern = patterns.get(file_type)
    if not pattern:
        print(
            f"Error: Unknown file_type '{file_type}'. Supported types are: {list(patterns.keys())}"
        )
        return

    print(f"\nProcessing file type '{file_type}' in directory '{directory_path}'...")

    # --- Step 3: Scan directory, parse CSVs, and update data ---
    for filename in os.listdir(directory_path):
        if not filename.startswith(file_type):
            continue

        match = pattern.match(filename)
        if match:
            k_val, second_level_key, log_info = None, None, ""

            if file_type == 'mvivf':
                second_level_key = int(match.group(1))  # iters
                k_val = int(match.group(2))
                log_info = f"mvivf (k={k_val}, iters={second_level_key})"
            elif file_type == 'svh':
                k_val = int(match.group(1))
                second_level_key = int(match.group(2))  # cand
                log_info = f"svh (k={k_val}, cand={second_level_key})"
            elif file_type == 'vamana':
                second_level_key = int(match.group(1))  # R
                k_val = int(match.group(2))
                log_info = f"vamana (k={k_val}), R={second_level_key})"
            elif file_type == 'muvera':
                second_level_key = int(match.group(1))  # R
                k_val = int(match.group(2))
                log_info = f"muvera (k={k_val}), fde={second_level_key})"
            elif file_type == 'mpv':
                second_level_key = 200  # R
                k_val = int(match.group(1))
                log_info = f"mpv (k={k_val}), R={second_level_key})"

            print(f"  -> Parsing file: {filename} ({log_info})")

            k_val_str, second_level_key_str = str(k_val), str(second_level_key)

            if k_val_str not in collected_data:
                collected_data[k_val_str] = {}
            if second_level_key_str not in collected_data[k_val_str]:
                collected_data[k_val_str][second_level_key_str] = {}

            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, mode='r', newline='') as infile:
                    reader = csv.reader(infile)
                    next(reader, None)  # Skip header
                    for row in reader:
                        if len(row) == 7:
                            nprobes_or_L = int(row[1])
                            data_tuple = (
                                float(row[2]),
                                float(row[3]),
                                float(row[4]),
                                float(row[5]),
                                float(row[6]),
                            )
                            collected_data[k_val_str][second_level_key_str][
                                str(nprobes_or_L)
                            ] = data_tuple
                        else:
                            print(f"     ! Warning: Skipping malformed row in {filename}: {row}")
            except (IOError, ValueError) as e:
                print(f"    ! Error processing file {filename}: {e}")

    # --- Step 4: Write the updated data back to the JSON file ---
    try:
        with open(output_json_path, 'w') as f:
            json.dump(collected_data, f, indent=4)
        print(f"Successfully updated '{output_json_path}'.")
    except IOError as e:
        print(f"Error writing to JSON file '{output_json_path}': {e}")


def plot_data(
    k,
    mvivf_json_path,
    svh_json_path,
    vamana_json_path,
    muvera_json_path,
    mpv_json_path,
    dataset_name,
    output_image_path=None,
):
    """
    Generates and saves a 3x2 grid of plots from the collected data.

    Args:
        k (int): The value of 'k' to filter the data by.
        mvivf_json_path (str): Path to the mvivf JSON data file.
        svh_json_path (str): Path to the svh JSON data file.
        vamana_json_path (str): Path to the vamana JSON data file.
        muvera_json_path (str): Path to the muvera JSON data file.
        mpv_json_path (str): Path to the muvera JSON data file.
        output_image_path (str, optional): Path to save the plot image.
                                           If None, displays the plot instead.
    """
    try:
        with open(mvivf_json_path, 'r') as f:
            mvivf_data = json.load(f)
        with open(svh_json_path, 'r') as f:
            svh_data = json.load(f)
        with open(vamana_json_path, 'r') as f:
            vamana_data = json.load(f)
        with open(muvera_json_path, 'r') as f:
            muvera_data = json.load(f)
        with open(mpv_json_path, 'r') as f:
            mpv_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find data file for plotting: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse JSON data file: {e}")
        return

    k_str = str(k)
    # fig, axs = plt.subplots(3, 2, figsize=(16, 18), sharex='col')
    fig, axs = plt.subplots(2, 2, figsize=(16, 18), sharex='col')
    fig.suptitle(f'{dataset_name}: Performance vs. Recall for k={k}', fontsize=20)

    # y_labels = ['Avg Cmps', 'QPS_seq', 'QPS_par']
    y_labels = ['Avg Cmps', 'QPS_seq']
    # y_indices = [2, 0, 1]  # Indices of y-values in the data tuple
    y_indices = [2, 0]  # Indices of y-values in the data tuple

    # Plot data for mvivf, sorted by the 'iters' value
    if k_str in mvivf_data:
        # Sort by the 'iters' key, converting it to an integer for correct numerical order
        sorted_mvivf_items = sorted(mvivf_data[k_str].items(), key=lambda item: int(item[0]))
        for iters_val, nprobes_data in sorted_mvivf_items:
            points = sorted(nprobes_data.values(), key=lambda x: x[3])
            if not points:
                continue

            x_recall1 = []
            for p in points:
                x_recall1.append(p[3])
                if x_recall1[-1] == 1.0:
                    break

            x_recallk = []
            for p in points:
                x_recallk.append(p[4])
                if x_recallk[-1] == 1.0:
                    break
            # for i in range(3):
            for i in range(2):
                y_vals = [p[y_indices[i]] for p in points]
                axs[i, 0].plot(
                    x_recall1,
                    y_vals[: len(x_recall1)],
                    # marker='o',
                    linestyle='-',
                    label=f'mvivf, iters={iters_val}',
                )
                axs[i, 1].plot(
                    x_recallk,
                    y_vals[: len(x_recallk)],
                    # marker='o',
                    linestyle='-',
                    label=f'mvivf, iters={iters_val}',
                )

    # Plot data for svh, sorted by the 'cand' value
    if k_str in svh_data:
        # Sort by the 'cand' key, converting it to an integer for correct numerical order
        sorted_svh_items = sorted(svh_data[k_str].items(), key=lambda item: int(item[0]))
        for cand_val, nprobes_data in sorted_svh_items:
            points = sorted(nprobes_data.values(), key=lambda x: x[3])
            if not points:
                continue

            x_recall1 = []
            for p in points:
                x_recall1.append(p[3])
                if x_recall1[-1] == 1.0:
                    break

            x_recallk = []
            for p in points:
                x_recallk.append(p[4])
                if x_recallk[-1] == 1.0:
                    break
            # for i in range(3):
            for i in range(2):
                y_vals = [p[y_indices[i]] for p in points]
                axs[i, 0].plot(
                    x_recall1,
                    y_vals[: len(x_recall1)],
                    # marker='s',
                    linestyle='--',
                    label=f'svh, cand={cand_val}',
                )
                axs[i, 1].plot(
                    x_recallk,
                    y_vals[: len(x_recallk)],
                    # marker='s',
                    linestyle='--',
                    label=f'svh, cand={cand_val}',
                )

    # Plot data for vamana, sorted by the 'R' value
    if k_str in vamana_data:
        # Sort by the 'R' key, converting it to an integer for correct numerical order
        sorted_vamana_items = sorted(vamana_data[k_str].items(), key=lambda item: int(item[0]))
        for R_val, Ls_data in sorted_vamana_items:
            points = sorted(Ls_data.values(), key=lambda x: x[3])
            if not points:
                continue

            x_recall1 = []
            for p in points:
                x_recall1.append(p[3])
                if x_recall1[-1] == 1.0:
                    break

            x_recallk = []
            for p in points:
                x_recallk.append(p[4])
                if x_recallk[-1] == 1.0:
                    break
            # for i in range(3):
            for i in range(2):
                y_vals = [p[y_indices[i]] for p in points]
                axs[i, 0].plot(
                    x_recall1,
                    y_vals[: len(x_recall1)],
                    # marker='^',
                    linestyle='dashdot',
                    label=f'vamana, R={R_val}',
                )
                axs[i, 1].plot(
                    x_recallk,
                    y_vals[: len(x_recallk)],
                    # marker='^',
                    linestyle='dashdot',
                    label=f'vamana, R={R_val}',
                )
    # Plot data for muvera, sorted by the 'FDE' value
    if k_str in muvera_data:
        # Sort by the 'fde' key, converting it to an integer for correct numerical order
        sorted_muvera_items = sorted(muvera_data[k_str].items(), key=lambda item: int(item[0]))
        for fde_val, Ls_data in sorted_muvera_items:
            points = sorted(Ls_data.values(), key=lambda x: x[3])
            if not points:
                continue

            x_recall1 = []
            for p in points:
                x_recall1.append(p[3])
                if x_recall1[-1] == 1.0:
                    break

            x_recallk = []
            for p in points:
                x_recallk.append(p[4])
                if x_recallk[-1] == 1.0:
                    break
            # for i in range(3):
            for i in range(2):
                y_vals = [p[y_indices[i]] for p in points]
                axs[i, 0].plot(
                    x_recall1,
                    y_vals[: len(x_recall1)],
                    # marker='*',
                    linestyle='dotted',
                    label=f'muvera, d_fde={fde_val}',
                )
                axs[i, 1].plot(
                    x_recallk,
                    y_vals[: len(x_recallk)],
                    # marker='*',
                    linestyle='dotted',
                    label=f'muvera, d_fde={fde_val}',
                )

    # Plot data for mpv, sorted by the 'R' value
    if k_str in mpv_data:
        # Sort by the 'R' key, converting it to an integer for correct numerical order
        sorted_mpv_items = sorted(mpv_data[k_str].items(), key=lambda item: int(item[0]))
        for R_val, Ls_data in sorted_mpv_items:
            points = sorted(Ls_data.values(), key=lambda x: x[3])
            if not points:
                continue

            x_recall1 = []
            for p in points:
                x_recall1.append(p[3])
                if x_recall1[-1] == 1.0:
                    break

            x_recallk = []
            for p in points:
                x_recallk.append(p[4])
                if x_recallk[-1] == 1.0:
                    break
            # for i in range(3):
            for i in range(2):
                y_vals = [p[y_indices[i]] for p in points]
                axs[i, 0].plot(
                    x_recall1,
                    y_vals[: len(x_recall1)],
                    marker='*',
                    linestyle='dotted',
                    # label=f'mpv, R={R_val}',
                    label=f'mpv',
                )
                axs[i, 1].plot(
                    x_recallk,
                    y_vals[: len(x_recallk)],
                    marker='*',
                    linestyle='dotted',
                    label=f'mpv',
                )

    # --- Final plot styling ---
    # for i in range(3):
    for i in range(2):
        # Set y-axis of the first row (Avg Cmps) to log scale
        if i == 0:
            axs[i, 0].set_yscale('log')
            axs[i, 1].set_yscale('log')
        axs[i, 0].set_ylabel(y_labels[i], fontsize=12)
        axs[i, 1].set_ylabel(y_labels[i], fontsize=12)
        axs[i, 0].legend()
        axs[i, 1].legend()
        axs[i, 0].grid(True, linestyle=':')
        axs[i, 1].grid(True, linestyle=':')

    axs[len(y_indices) - 1, 0].set_xlabel('Recall 1@k', fontsize=12)
    axs[len(y_indices) - 1, 1].set_xlabel(f'Recall {k}@{k}', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Save or show the plot ---
    if output_image_path:
        try:
            plt.savefig(output_image_path, bbox_inches='tight')
            print(f"\nPlot successfully saved to '{output_image_path}'")
        except IOError as e:
            print(f"Error saving plot to '{output_image_path}': {e}")
    else:
        plt.show()

    plt.close(fig)  # Close the figure to free up memory


def main():
    """
    Main function to parse command-line arguments and run the processing and plotting.
    """
    parser = argparse.ArgumentParser(
        description="Process CSV data files and generate plots from a specified dataset directory.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--dataset_name",
        type=str,
        required=True,  # This must be provided to build the path
        help="The name of the dataset (e.g., 'sift1m').\nThe script will look for data in '/ssd2/kishen/MVC/{dataset_name}/stats'.",
    )
    parser.add_argument(
        "-p",
        "--process",
        choices=['mvivf', 'svh', 'vamana', "muvera", "mpv", 'all', 'plot'],
        default='all',
        help="""The type of file to process.
'mvivf': Process only mvivf CSVs.
'svh':   Process only svh CSVs.
'vamana':   Process only vamana CSVs.
'muvera':   Process only muvera CSVs.
'mpv':   Process only mpv CSVs.
'all':   Process all CSVs (default).
'plot':  Only generate plots from existing JSON files.""",
    )
    parser.add_argument(
        "--k", type=int, default=10, help="The value of 'k' to generate plots for (default: 10)."
    )

    args = parser.parse_args()

    # Construct the path dynamically based on the dataset name.
    data_path = f"/ssd2/kishen/MVC/{args.dataset_name}/stats"

    # Ensure the main data directory exists
    if not os.path.exists(data_path):
        print(f"Error: Dataset directory '{data_path}' does not exist.")
        return

    mvivf_json_file = os.path.join(data_path, "mvivf_data.json")
    svh_json_file = os.path.join(data_path, "svh_data.json")
    vamana_json_file = os.path.join(data_path, "vamana_data.json")
    muvera_json_file = os.path.join(data_path, "muvera_data.json")
    mpv_json_file = os.path.join(data_path, "mpv_data.json")

    # --- Process CSVs based on '--process' argument ---
    if args.process in ['mvivf', 'all']:
        update_json_from_csvs(data_path, "mvivf", mvivf_json_file)

    if args.process in ['svh', 'all']:
        update_json_from_csvs(data_path, "svh", svh_json_file)

    if args.process in ['vamana', 'all']:
        update_json_from_csvs(data_path, "vamana", vamana_json_file)

    if args.process in ['muvera', 'all']:
        update_json_from_csvs(data_path, "muvera", muvera_json_file)

    if args.process in ['mpv', 'all']:
        update_json_from_csvs(data_path, "mpv", mpv_json_file)

    # --- Generate plots ---
    if args.process != 'plot':
        print(f"\nGenerating and saving plots for k={args.k}...")
    else:
        print(f"\nGenerating plots for k={args.k} from existing JSON files...")

    plot_output_file = os.path.join(data_path, f"performance_plot_k{args.k}.png")
    plot_data(
        args.k,
        mvivf_json_file,
        svh_json_file,
        vamana_json_file,
        muvera_json_file,
        mpv_json_file,
        args.dataset_name,
        plot_output_file,
    )


if __name__ == "__main__":
    main()
