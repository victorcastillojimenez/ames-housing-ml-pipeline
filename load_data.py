import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
import xgboost as xgb
import lightgbm as lgb
import openml

def main():
    print("Verifying library installation...")
    print(f"Pandas: {pd.__version__}")
    print(f"Numpy: {np.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    print(f"Seaborn: {sns.__version__}")
    
    dataset_id = 41211
    print(f"Fetching dataset from OpenML with ID {dataset_id}...")
    try:
        # Fetching using sklearn as requested (direct charge)
        dataset = fetch_openml(data_id=dataset_id, as_frame=True)
        df = dataset.frame
        print("Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print("\nFirst few rows:")
        print(df.head())
        
        # Basic info
        print("\nDataset Info:")
        print(df.info())
        
        # Save to local csv for easier access
        df.to_csv("dataset.csv", index=False)
        print("\nDataset saved as 'dataset.csv'")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    main()
