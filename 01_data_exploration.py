"""
Data Exploration and Initial Analysis for Actuarial Pricing Simulator
French MTPL2 Dataset Processing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up data directory
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

def load_and_explore_data():
    """
    Load the French MTPL2 dataset and perform initial exploration
    
    Note: Download the dataset from Kaggle:
    https://www.kaggle.com/datasets/florianbordesoule/french-motor-claim-datasets-fremtpl2-and-fremtpl2sev
    
    Files needed:
    - freMTPL2freq.csv (frequency data)
    - freMTPL2sev.csv (severity data)
    """
    
    # Load frequency data
    try:
        freq_df = pd.read_csv(data_dir / "freMTPL2freq.csv")
        print("✅ Frequency data loaded successfully")
        print(f"Shape: {freq_df.shape}")
    except FileNotFoundError:
        print("❌ freMTPL2freq.csv not found. Please download from Kaggle.")
        return None, None
    
    # Load severity data
    try:
        sev_df = pd.read_csv(data_dir / "freMTPL2sev.csv")
        print("✅ Severity data loaded successfully")
        print(f"Shape: {sev_df.shape}")
    except FileNotFoundError:
        print("❌ freMTPL2sev.csv not found. Please download from Kaggle.")
        return freq_df, None
    
    return freq_df, sev_df

def explore_frequency_data(freq_df):
    """Explore the frequency dataset"""
    print("\n" + "="*50)
    print("FREQUENCY DATA EXPLORATION")
    print("="*50)
    
    # Basic info
    print("\n1. Dataset Info:")
    print(freq_df.info())
    
    print("\n2. First few rows:")
    print(freq_df.head())
    
    print("\n3. Statistical Summary:")
    print(freq_df.describe())
    
    print("\n4. Missing Values:")
    print(freq_df.isnull().sum())
    
    print("\n5. Key Actuarial Variables:")
    print(f"Exposure range: {freq_df['Exposure'].min():.3f} to {freq_df['Exposure'].max():.3f}")
    print(f"Claims range: {freq_df['ClaimNb'].min()} to {freq_df['ClaimNb'].max()}")
    print(f"Percentage of policies with claims: {(freq_df['ClaimNb'] > 0).mean():.1%}")
    
    # Calculate frequency
    freq_df['Frequency'] = freq_df['ClaimNb'] / freq_df['Exposure']
    print(f"Average frequency: {freq_df['Frequency'].mean():.4f}")
    
    return freq_df

def explore_severity_data(sev_df):
    """Explore the severity dataset"""
    print("\n" + "="*50)
    print("SEVERITY DATA EXPLORATION")
    print("="*50)
    
    # Basic info
    print("\n1. Dataset Info:")
    print(sev_df.info())
    
    print("\n2. First few rows:")
    print(sev_df.head())
    
    print("\n3. Statistical Summary:")
    print(sev_df.describe())
    
    print("\n4. Missing Values:")
    print(sev_df.isnull().sum())
    
    print("\n5. Claim Amount Analysis:")
    print(f"Claim amount range: €{sev_df['ClaimAmount'].min():.2f} to €{sev_df['ClaimAmount'].max():.2f}")
    print(f"Average claim amount: €{sev_df['ClaimAmount'].mean():.2f}")
    print(f"Median claim amount: €{sev_df['ClaimAmount'].median():.2f}")
    
    return sev_df

def analyze_rating_factors(freq_df):
    """Analyze the key rating factors"""
    print("\n" + "="*50)
    print("RATING FACTOR ANALYSIS")
    print("="*50)
    
    # Driver Age
    print("\n1. Driver Age Distribution:")
    print(freq_df['DrivAge'].describe())
    
    # Vehicle Age
    print("\n2. Vehicle Age Distribution:")
    print(freq_df['VehAge'].describe())
    
    # Categorical variables
    categorical_vars = ['VehBrand', 'VehGas', 'Region', 'Area']
    
    for var in categorical_vars:
        if var in freq_df.columns:
            print(f"\n3. {var} Distribution:")
            print(freq_df[var].value_counts().head(10))
    
    return freq_df

def create_visualizations(freq_df, sev_df=None):
    """Create initial visualizations"""
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Frequency distribution
    axes[0, 0].hist(freq_df['Frequency'], bins=50, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Distribution of Claim Frequency')
    axes[0, 0].set_xlabel('Frequency (Claims per Exposure)')
    axes[0, 0].set_ylabel('Count')
    
    # 2. Exposure distribution
    axes[0, 1].hist(freq_df['Exposure'], bins=50, alpha=0.7, color='lightgreen')
    axes[0, 1].set_title('Distribution of Exposure')
    axes[0, 1].set_xlabel('Exposure (Years)')
    axes[0, 1].set_ylabel('Count')
    
    # 3. Driver Age vs Frequency
    age_freq = freq_df.groupby('DrivAge')['Frequency'].mean().reset_index()
    axes[1, 0].scatter(age_freq['DrivAge'], age_freq['Frequency'], alpha=0.6, color='coral')
    axes[1, 0].set_title('Driver Age vs Average Frequency')
    axes[1, 0].set_xlabel('Driver Age')
    axes[1, 0].set_ylabel('Average Frequency')
    
    # 4. Vehicle Age vs Frequency
    veh_freq = freq_df.groupby('VehAge')['Frequency'].mean().reset_index()
    axes[1, 1].scatter(veh_freq['VehAge'], veh_freq['Frequency'], alpha=0.6, color='gold')
    axes[1, 1].set_title('Vehicle Age vs Average Frequency')
    axes[1, 1].set_xlabel('Vehicle Age')
    axes[1, 1].set_ylabel('Average Frequency')
    
    plt.tight_layout()
    plt.savefig(data_dir / 'initial_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # If severity data is available, create severity plots
    if sev_df is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Severity distribution
        axes[0].hist(sev_df['ClaimAmount'], bins=50, alpha=0.7, color='lightcoral')
        axes[0].set_title('Distribution of Claim Amounts')
        axes[0].set_xlabel('Claim Amount (€)')
        axes[0].set_ylabel('Count')
        axes[0].set_xlim(0, sev_df['ClaimAmount'].quantile(0.95))  # Remove extreme outliers for visualization
        
        # Log-scale severity distribution
        axes[1].hist(np.log(sev_df['ClaimAmount']), bins=50, alpha=0.7, color='lightblue')
        axes[1].set_title('Distribution of Log(Claim Amounts)')
        axes[1].set_xlabel('Log(Claim Amount)')
        axes[1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(data_dir / 'severity_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main execution function"""
    print("🚀 Starting Actuarial Pricing Simulator - Data Exploration")
    print("="*60)
    
    # Load data
    freq_df, sev_df = load_and_explore_data()
    
    if freq_df is None:
        print("\n❌ Cannot proceed without frequency data. Please download the dataset.")
        return
    
    # Explore frequency data
    freq_df = explore_frequency_data(freq_df)
    
    # Explore severity data if available
    if sev_df is not None:
        sev_df = explore_severity_data(sev_df)
    
    # Analyze rating factors
    freq_df = analyze_rating_factors(freq_df)
    
    # Create visualizations
    create_visualizations(freq_df, sev_df)
    
    print("\n✅ Data exploration complete!")
    print("Next steps:")
    print("1. Download the French MTPL2 dataset from Kaggle if you haven't already")
    print("2. Run this script to explore the data")
    print("3. Proceed to feature engineering and modeling")

if __name__ == "__main__":
    main()