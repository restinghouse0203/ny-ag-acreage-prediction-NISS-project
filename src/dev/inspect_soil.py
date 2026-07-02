import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SOIL_FEATURES

data_path = SOIL_FEATURES

if data_path.exists():
    try:
        df = pd.read_csv(data_path, nrows=5)
        print("Columns:", df.columns.tolist())
        print(df.head())
    except Exception as e:
        print(f"Error reading CSV: {e}")
else:
    print(f"File not found: {data_path}")
