import os
import pandas as pd

# Paths
csv_file = "reciters.csv"  # Change to your CSV file path
clean_folder = "clean/"  # Change to your clean folder path

# Load the CSV file
df = pd.read_csv(csv_file)

# Get the list of actual files in the clean folder
existing_files = set(os.listdir(clean_folder))

# Filter out rows where the filename does not exist in the clean folder
df = df[df['fname'].isin(existing_files)]  # Change 'filename' to the actual column name

# Save the cleaned CSV file
df.to_csv(csv_file, index=False)

print("CSV file updated. Removed missing files.")
