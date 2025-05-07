import pandas as pd
from pathlib import Path

root = Path("/ssd2/kishen/MVC/")

rows = []

for dataset_dir in root.iterdir():
    for index_dir in dataset_dir.iterdir():
        for k_file in index_dir.glob("k=*.csv"):
            k_val = int(k_file.stem.split('=')[1])
            df = pd.read_csv(k_file)
            for _, row in df.iterrows():
                rows.append({
                    "dataset": dataset_dir.name,
                    "index": index_dir.name,
                    "k": k_val,
                    **row.to_dict()
                })

all_results = pd.DataFrame(rows)
all_results.to_csv("benchmark_summary.csv", index=False)
