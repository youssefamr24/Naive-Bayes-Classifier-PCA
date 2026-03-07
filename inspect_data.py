import pandas as pd
import os

# Update this with the actual name of your CSV file
file_path = 'data/Students Social Media Addiction.csv' 

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    
    # 1. Check data types and missing values
    print("--- Dataset Info ---")
    print(df.info())
    
    # 2. Look at the first 5 rows
    print("\n--- First 5 Rows ---")
    print(df.head())
    
    # 3. Check class distribution (important for your report)
    print("\n--- Class Distribution ---")
    print(df['Addicted_Score'].value_counts()) # Replace 'Addicted_Score' with your target column name
else:
    print(f"File not found. Check the filename in {file_path}")