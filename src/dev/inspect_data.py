import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MERGED_DATA_DIR

data_path = MERGED_DATA_DIR / "merged_CSB_polygon.parquet"

if data_path.exists():
    try:
        df = pd.read_parquet(data_path)
        print("Columns:", df.columns.tolist())
        print("Shape:", df.shape)
        print(df.head())
    except Exception as e:
        print(f"Error reading parquet: {e}")
else:
    print(f"File not found: {data_path}")
