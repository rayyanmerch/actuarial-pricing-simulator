"""
Download French MTPL2 Dataset Automatically
This script downloads the data directly from OpenML
"""

import pandas as pd
import numpy as np
from pathlib import Path

def download_fremtpl2_data():
    """
    Download the French MTPL2 dataset from OpenML
    """
    print("🚀 Downloading French MTPL2 dataset...")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    try:
        # Download frequency data from OpenML
        print("📊 Downloading frequency data...")
        df_freq = pd.read_csv(
            "https://www.openml.org/data/get_csv/20649148/freMTPL2freq.arff",
            quotechar="'"
        )
        
        # Clean column names (remove quotes)
        df_freq.rename(lambda x: x.replace('"', ''), axis='columns', inplace=True)
        df_freq['IDpol'] = df_freq['IDpol'].astype(np.int64)
        
        # Save frequency data
        df_freq.to_csv(data_dir / "freMTPL2freq.csv", index=False)
        print(f"✅ Frequency data saved: {df_freq.shape}")
        
        # Download severity data from OpenML
        print("💰 Downloading severity data...")
        df_sev = pd.read_csv(
            "https://www.openml.org/data/get_csv/20649149/freMTPL2sev.arff"
        )
        
        # Save severity data
        df_sev.to_csv(data_dir / "freMTPL2sev.csv", index=False)
        print(f"✅ Severity data saved: {df_sev.shape}")
        
        # Show some basic info
        print("\n📈 Dataset Overview:")
        print(f"Frequency data: {df_freq.shape[0]:,} policies")
        print(f"Severity data: {df_sev.shape[0]:,} claims")
        print(f"Columns in frequency data: {list(df_freq.columns)}")
        
        return df_freq, df_sev
        
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        print("💡 Alternative: You can manually download from:")
        print("   https://www.kaggle.com/datasets/floser/french-motor-claims-datasets-fremtpl2freq")
        return None, None

if __name__ == "__main__":
    print("🎯 French MTPL2 Data Downloader")
    print("="*50)
    
    freq_df, sev_df = download_fremtpl2_data()
    
    if freq_df is not None:
        print("\n🎉 Data download complete!")
        print("Next steps:")
        print("1. Run: python 01_data_exploration.py")
        print("2. Run: python 02_feature_engineering.py")
        print("3. Run: python 03_glm_modeling.py")
    else:
        print("\n❌ Data download failed. Please check your internet connection.")