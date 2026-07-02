import pandas as pd
import numpy as np

path = "../data/HDD/mctycddy.txt"
try:
    with open(path, 'r') as f:
        head = [next(f) for _ in range(20)]
        for i, line in enumerate(head):
            print(f"{i}: {repr(line)}")
except Exception as e:
    print(f"File read error: {e}")

