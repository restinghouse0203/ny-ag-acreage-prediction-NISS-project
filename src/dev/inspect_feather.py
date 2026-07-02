import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import interim_csb_path

data_path = interim_csb_path("20172024")

if data_path.exists():
    try:
        df = pd.read_feather(data_path)
        print("Columns:", df.columns.tolist())
        print("Shape:", df.shape)
        print(df.head())
    except Exception as e:
        print(f"Error reading feather: {e}")
else:
    print(f"File not found: {data_path}")
