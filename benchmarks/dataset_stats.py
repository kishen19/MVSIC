
import os
import csv
import sys
import struct

def get_fast_stats(file_path):
    """
    Quickly calculates statistics for a given .pcs file by reading only its header.

    Args:
        file_path (str): The path to the .pcs file.

    Returns:
        tuple: A tuple containing (num_point_clouds, dim).
               Returns ('N/A', 'N/A') if the file doesn't exist or is invalid.
    """
    if not os.path.exists(file_path):
        return 'N/A', 'N/A'

    try:
        with open(file_path, 'rb') as f:
            # Read the first 8 bytes for dimension
            dim_bytes = f.read(8)
            if len(dim_bytes) < 8:
                return 'Invalid', 'Invalid'
            dim = struct.unpack('Q', dim_bytes)[0]

            # Read the next 8 bytes for number of point clouds
            n_bytes = f.read(8)
            if len(n_bytes) < 8:
                return 'Invalid', 'Invalid'
            num_point_clouds = struct.unpack('Q', n_bytes)[0]
            
            return num_point_clouds, dim
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return 'Error', 'Error'

def check_gt_file(gt_path):
    """Checks if a ground truth file exists."""
    return "Present" if os.path.exists(gt_path) else "Missing"

def main():
    """
    Main function to gather dataset statistics and write them to a CSV file.
    """
    output_file = './results/dataset_stats.csv'
    data_to_write = []
    headers = [
        "Dataset Source", "Dataset Name", "Split", "Data Type",
        "Num Point Clouds", "Dimension", "GT File Status"
    ]

    # --- BEIR and Vidore Datasets ---
    for source in ['beir', 'vidore']:
        base_path = f'./data/{source}'
        if not os.path.isdir(base_path):
            continue
        
        for dataset_name in sorted(os.listdir(base_path)):
            dataset_path = os.path.join(base_path, dataset_name)
            if not os.path.isdir(dataset_path):
                continue

            # Process points
            points_file = os.path.join(dataset_path, f'{dataset_name}_points.pcs')
            num_pcs, dim = get_fast_stats(points_file)
            gt_path = os.path.join(dataset_path, f'{dataset_name}_chamfer_neighbors.gt')
            gt_status = check_gt_file(gt_path)
            data_to_write.append([
                source, dataset_name, "N/A", "points",
                num_pcs, dim, gt_status
            ])

            # Process queries
            queries_file = os.path.join(dataset_path, f'{dataset_name}_queries.pcs')
            num_pcs, dim = get_fast_stats(queries_file)
            # Queries don't have a separate GT file in this structure, linked to points GT
            data_to_write.append([
                source, dataset_name, "N/A", "queries",
                num_pcs, dim, "N/A"
            ])

    # --- Lotte Datasets ---
    lotte_base_path = './data/lotte'
    if os.path.isdir(lotte_base_path):
        for split in ['dev', 'test']:
            split_path = os.path.join(lotte_base_path, split)
            if not os.path.isdir(split_path):
                continue

            for dataset_name in sorted(os.listdir(split_path)):
                dataset_path = os.path.join(split_path, dataset_name)
                if not os.path.isdir(dataset_path):
                    continue

                # Process points
                points_file = os.path.join(dataset_path, f'{dataset_name}_points.pcs')
                num_pcs, dim = get_fast_stats(points_file)
                gt_path = os.path.join(dataset_path, f'{dataset_name}_chamfer_neighbors.gt')
                gt_status = check_gt_file(gt_path)
                data_to_write.append([
                    'lotte', dataset_name, split, "points",
                    num_pcs, dim, gt_status
                ])

                # Process search queries
                search_queries_file = os.path.join(dataset_path, f'{dataset_name}_search_queries.pcs')
                num_pcs, dim = get_fast_stats(search_queries_file)
                gt_path = os.path.join(dataset_path, f'{dataset_name}_search_chamfer_neighbors.gt')
                gt_status = check_gt_file(gt_path)
                data_to_write.append([
                    'lotte', dataset_name, split, "search_queries",
                    num_pcs, dim, gt_status
                ])

                # Process forum queries
                forum_queries_file = os.path.join(dataset_path, f'{dataset_name}_forum_queries.pcs')
                num_pcs, dim = get_fast_stats(forum_queries_file)
                gt_path = os.path.join(dataset_path, f'{dataset_name}_forum_chamfer_neighbors.gt')
                gt_status = check_gt_file(gt_path)
                data_to_write.append([
                    'lotte', dataset_name, split, "forum_queries",
                    num_pcs, dim, gt_status
                ])

    # Write to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data_to_write)

    print(f"Dataset statistics have been written to {output_file}")

if __name__ == "__main__":
    main()
