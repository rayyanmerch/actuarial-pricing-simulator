"""
Feature Engineering for Actuarial Pricing Simulator
Transform raw data into model-ready features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from pathlib import Path
import joblib

class ActuarialFeatureEngineer:
    """
    Feature engineering class for actuarial pricing models
    """
    
    def __init__(self):
        self.encoders = {}
        self.binning_rules = {}
        self.feature_names = []
        
    def create_age_bins(self, df, age_column='DrivAge'):
        """
        Create age bins for driver age - industry standard approach
        """
        # Define age bins based on typical actuarial practice
        bins = [17, 25, 35, 45, 55, 65, 75, 100]
        labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76+']
        
        # Create age groups
        df[f'{age_column}_Group'] = pd.cut(
            df[age_column], 
            bins=bins, 
            labels=labels, 
            right=False,
            include_lowest=True
        )
        
        # Store binning rules
        self.binning_rules[age_column] = {'bins': bins, 'labels': labels}
        
        return df
    
    def create_vehicle_age_bins(self, df, veh_age_column='VehAge'):
        """
        Create vehicle age bins
        """
        # Cap vehicle age at 20 years (older vehicles grouped together)
        df[veh_age_column] = df[veh_age_column].clip(upper=20)
        
        # Create vehicle age groups
        bins = [0, 2, 5, 10, 15, 25]
        labels = ['0-1', '2-4', '5-9', '10-14', '15+']
        
        df[f'{veh_age_column}_Group'] = pd.cut(
            df[veh_age_column], 
            bins=bins, 
            labels=labels, 
            right=False,
            include_lowest=True
        )
        
        # Store binning rules
        self.binning_rules[veh_age_column] = {'bins': bins, 'labels': labels}
        
        return df
    
    def create_power_bins(self, df, power_column='VehPower'):
        """
        Create vehicle power bins if the column exists
        """
        if power_column not in df.columns:
            return df
            
        # Create power bins
        bins = [0, 6, 8, 10, 12, 16, 50]
        labels = ['≤6', '7-8', '9-10', '11-12', '13-16', '17+']
        
        df[f'{power_column}_Group'] = pd.cut(
            df[power_column], 
            bins=bins, 
            labels=labels, 
            right=False,
            include_lowest=True
        )
        
        # Store binning rules
        self.binning_rules[power_column] = {'bins': bins, 'labels': labels}
        
        return df
    
    def handle_categorical_variables(self, df, categorical_columns):
        """
        Handle categorical variables with proper encoding
        """
        for col in categorical_columns:
            if col in df.columns:
                # Convert to string first to handle any type issues
                df[col] = df[col].astype(str)
                
                # Handle missing values (now they would be 'nan' strings)
                df[col] = df[col].replace(['nan', 'None', ''], 'Unknown')
                
                # Group rare categories
                value_counts = df[col].value_counts()
                rare_categories = value_counts[value_counts < 100].index
                
                if len(rare_categories) > 0:
                    df[col] = df[col].replace(rare_categories, 'Other')
                
        return df
    
    def apply_one_hot_encoding(self, df, categorical_columns, fit=True):
        """
        Apply one-hot encoding to categorical variables
        """
        encoded_dfs = []
        
        for col in categorical_columns:
            if col in df.columns:
                if fit:
                    # Fit new encoder
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded_data = encoder.fit_transform(df[[col]])
                    self.encoders[col] = encoder
                else:
                    # Use existing encoder
                    encoder = self.encoders[col]
                    encoded_data = encoder.transform(df[[col]])
                
                # Create column names
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)
                encoded_dfs.append(encoded_df)
        
        # Combine all encoded dataframes
        if encoded_dfs:
            encoded_combined = pd.concat(encoded_dfs, axis=1)
            return encoded_combined
        else:
            return pd.DataFrame(index=df.index)
    
    def prepare_frequency_data(self, freq_df, fit=True):
        """
        Prepare frequency data for modeling
        """
        print("🔧 Preparing frequency data...")
        
        # Create a copy to avoid modifying original data
        df = freq_df.copy()
        
        # Calculate frequency
        df['Frequency'] = df['ClaimNb'] / df['Exposure']
        
        # Handle infinite or NaN frequencies
        df = df[df['Exposure'] > 0]
        df = df[df['Frequency'].notna()]
        
        # Apply binning
        df = self.create_age_bins(df, 'DrivAge')
        df = self.create_vehicle_age_bins(df, 'VehAge')
        df = self.create_power_bins(df, 'VehPower')
        
        # Define categorical columns to encode
        categorical_columns = [
            'DrivAge_Group', 'VehAge_Group', 'VehPower_Group',
            'VehBrand', 'VehGas', 'Region', 'Area'
        ]
        
        # Filter for columns that actually exist
        categorical_columns = [col for col in categorical_columns if col in df.columns]
        
        # Handle categorical variables
        df = self.handle_categorical_variables(df, categorical_columns)
        
        # Apply one-hot encoding
        encoded_features = self.apply_one_hot_encoding(df, categorical_columns, fit=fit)
        
        # Store feature names
        if fit:
            self.feature_names = encoded_features.columns.tolist()
        
        # Create final dataset
        result_df = pd.DataFrame(index=df.index)
        result_df['ClaimNb'] = df['ClaimNb']
        result_df['Exposure'] = df['Exposure']
        result_df['Frequency'] = df['Frequency']
        
        # Add encoded features
        for col in encoded_features.columns:
            result_df[col] = encoded_features[col]
        
        print(f"✅ Frequency data prepared. Shape: {result_df.shape}")
        print(f"Features: {len(self.feature_names)} categorical features created")
        
        return result_df
    
    def prepare_severity_data(self, sev_df, freq_df, fit=True):
        """
        Prepare severity data for modeling by joining with frequency data
        """
        print("🔧 Preparing severity data...")
        
        if sev_df is None:
            print("❌ No severity data provided")
            return None
        
        # Create a copy
        df = sev_df.copy()
        
        # Remove zero claims
        df = df[df['ClaimAmount'] > 0]
        
        # Calculate severity (average claim amount)
        df['Severity'] = df['ClaimAmount']
        
        # Remove extreme outliers (top 1%)
        severity_99th = df['Severity'].quantile(0.99)
        df = df[df['Severity'] <= severity_99th]
        
        # Join with frequency data to get risk factors
        # First, get only policies that have claims
        freq_with_claims = freq_df[freq_df['ClaimNb'] > 0].copy()
        
        # Join severity data with frequency data on IDpol
        df = df.merge(freq_with_claims[['IDpol', 'DrivAge', 'VehAge', 'VehPower', 
                                       'VehBrand', 'VehGas', 'Region', 'Area']], 
                     on='IDpol', how='inner')
        
        print(f"📊 Joined severity data shape: {df.shape}")
        
        # Apply same binning as frequency data
        df = self.create_age_bins(df, 'DrivAge')
        df = self.create_vehicle_age_bins(df, 'VehAge')
        df = self.create_power_bins(df, 'VehPower')
        
        # Define categorical columns
        categorical_columns = [
            'DrivAge_Group', 'VehAge_Group', 'VehPower_Group',
            'VehBrand', 'VehGas', 'Region', 'Area'
        ]
        
        # Filter for columns that actually exist
        categorical_columns = [col for col in categorical_columns if col in df.columns]
        
        # Handle categorical variables
        df = self.handle_categorical_variables(df, categorical_columns)
        
        # Apply one-hot encoding (use existing encoders from frequency data)
        encoded_features = self.apply_one_hot_encoding(df, categorical_columns, fit=fit)
        
        # Ensure we have the same features as frequency data
        if not fit:
            # For severity, we need to align with frequency model features
            missing_features = set(self.feature_names) - set(encoded_features.columns)
            for feature in missing_features:
                encoded_features[feature] = 0
            
            # Reorder columns to match frequency data
            encoded_features = encoded_features.reindex(columns=self.feature_names, fill_value=0)
        
        # Create final dataset
        result_df = pd.DataFrame(index=df.index)
        result_df['ClaimAmount'] = df['ClaimAmount']
        result_df['Severity'] = df['Severity']
        
        # Add encoded features
        for col in encoded_features.columns:
            result_df[col] = encoded_features[col]
        
        print(f"✅ Severity data prepared. Shape: {result_df.shape}")
        
        return result_df
    
    def save_preprocessing_objects(self, filepath):
        """
        Save preprocessing objects for later use
        """
        preprocessing_data = {
            'encoders': self.encoders,
            'binning_rules': self.binning_rules,
            'feature_names': self.feature_names
        }
        
        joblib.dump(preprocessing_data, filepath)
        print(f"✅ Preprocessing objects saved to {filepath}")
    
    def load_preprocessing_objects(self, filepath):
        """
        Load preprocessing objects
        """
        preprocessing_data = joblib.load(filepath)
        self.encoders = preprocessing_data['encoders']
        self.binning_rules = preprocessing_data['binning_rules']
        self.feature_names = preprocessing_data['feature_names']
        print(f"✅ Preprocessing objects loaded from {filepath}")

def main():
    """
    Main execution function
    """
    print("🚀 Starting Feature Engineering")
    print("="*50)
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Load raw data
    try:
        freq_df = pd.read_csv(data_dir / "freMTPL2freq.csv")
        print(f"✅ Loaded frequency data: {freq_df.shape}")
    except FileNotFoundError:
        print("❌ Frequency data not found. Please run data exploration first.")
        return
    
    try:
        sev_df = pd.read_csv(data_dir / "freMTPL2sev.csv")
        print(f"✅ Loaded severity data: {sev_df.shape}")
    except FileNotFoundError:
        print("⚠️ Severity data not found. Proceeding with frequency only.")
        sev_df = None
    
    # Initialize feature engineer
    fe = ActuarialFeatureEngineer()
    
    # Prepare frequency data
    freq_processed = fe.prepare_frequency_data(freq_df, fit=True)
    
    # Prepare severity data
    sev_processed = fe.prepare_severity_data(sev_df, freq_df, fit=False) if sev_df is not None else None
    
    # Save processed data
    freq_processed.to_parquet(data_dir / "frequency_processed.parquet")
    print("✅ Processed frequency data saved")
    
    if sev_processed is not None:
        sev_processed.to_parquet(data_dir / "severity_processed.parquet")
        print("✅ Processed severity data saved")
    
    # Save preprocessing objects
    fe.save_preprocessing_objects(data_dir / "preprocessing_objects.pkl")
    
    print("\n🎉 Feature engineering complete!")
    print("Next steps:")
    print("1. Review the processed data")
    print("2. Proceed to GLM modeling")

if __name__ == "__main__":
    main()