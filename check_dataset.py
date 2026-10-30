import pandas as pd
from pathlib import Path

for filename in [
    "data/raw/plot1_rcs.csv",
    "data/raw/moisture1.csv",
]:
    path = Path(filename)
    print(f"\n--- {filename} ---")

    df = pd.read_csv(path)
    print("Columns:", df.columns.tolist())
    print(df.head().to_string())
    print("Shape:", df.shape)