import pandas as pd
import numpy as np
# Load the original CSV file
file_path = "Provider_Info_Part_1.csv"
df = pd.read_csv(file_path)

# Split into 5 equal parts
num_splits = 5
split_dfs = np.array_split(df, num_splits)

# Save each part as a separate CSV file
split_files = []
for i, split_df in enumerate(split_dfs):
    split_file_path = f"Provider_Info_Part_new_{i+1}.csv"
    split_df.to_csv(split_file_path, index=False)
    split_files.append(split_file_path)

# Return file paths for download
split_files
