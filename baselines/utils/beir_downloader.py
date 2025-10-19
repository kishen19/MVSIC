import argparse
import os
import pathlib
import logging

try:
    from beir import util
except ImportError:
    logging.error("BEIR library not found. Please install it using: pip install beir")
    exit(1)

# Configure logging for clear output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def download_datasets(dataset_names: list, output_path: str):
    """
    Downloads and unzips a list of BEIR datasets to a specified output directory.

    Args:
        dataset_names: A list of strings, where each string is the name of a BEIR dataset.
        output_path: The path to the directory where datasets will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    logging.info(f"Ensured output directory exists at: {output_path}")

    for name in dataset_names:
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
        dataset_dir = os.path.join(output_path, name)

        if os.path.exists(dataset_dir):
            logging.info(f"Dataset '{name}' already exists at {dataset_dir}. Skipping.")
            continue

        logging.info(f"--- Downloading dataset: {name} ---")
        try:
            util.download_and_unzip(url, output_path)
            logging.info(f"Successfully downloaded and unzipped '{name}'.")
        except Exception as e:
            logging.error(f"Failed to download dataset '{name}'. Error: {e}")

    logging.info("\nDownload process complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and unzip BEIR datasets.")
    parser.add_argument(
        '--datasets',
        nargs='+',  # This allows for one or more dataset names
        required=True,
        help="A space-separated list of BEIR dataset names to download (e.g., 'scifact' 'msmarco').",
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=str(os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")),
        help="The directory to save the datasets. Defaults to a 'datasets' folder in the current directory.",
    )

    args = parser.parse_args()

    download_datasets(args.datasets, args.output_path)
