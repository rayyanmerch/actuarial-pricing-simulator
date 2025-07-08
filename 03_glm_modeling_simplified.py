"""
Simplified GLM Modeling for Actuarial Pricing Simulator
Using a more robust approach for large datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

class SimplifiedActuarialModeler:
    """
    Simplified GLM modeling using scikit-learn for better stability
    """
    
    def __init__(self):
        self.frequency_model = None
        self.severity_model = None
        self.feature_names = []
        self.base_frequency = None
        self.base_severity = None
        
    def prepare_modeling_data(self, df, target_col, exclude_cols):
        """
        Prepare data for modeling
        """
        # Create clean dataset
        clean_df = df.copy()
        
        # Remove rows with invalid target values
        if target_col in clean_df.columns:
            clean_df = clean_df[clean_df[target_col] >= 0]
            clean_df = clean_df[np.isfinite(clean_df[target_col])]
        
        # Prepare features
        feature_columns = [col for col in clean_df.columns if col not in exclude_cols]
        
        # Remove constant columns
        feature_columns = [col for col in feature_columns 
                          if clean_df[col].nunique() > 1]
        
        # Sample data for faster processing (use 100k rows max)
        if len(clean_df) > 100000:
            print(f"📊 Sampling {100000} rows from {len(clean_df)} for modeling")
            clean_df = clean_df.sample(n=100000, random_state=42)
        
        # Prepare X and y
        X = clean_df[feature_columns]
        y = clean_df[target_col] if target_col in clean_df.columns else None
        
        return X, y, clean_df
    
    def fit_frequency_model(self, freq_df):
        """
        Fit Poisson model for claim frequency using scikit-learn
        """
        print("🔧 Fitting frequency model (Poisson Regression)...")
        
        # Prepare data
        exclude_cols = ['ClaimNb', 'Exposure', 'Frequency']
        X, y, clean_df = self.prepare_modeling_data(freq_df, 'ClaimNb', exclude_cols)
        
        if X is None or y is None:
            print("❌ Could not prepare frequency data")
            return None
        
        print(f"📊 Frequency modeling data: {X.shape}")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Calculate base frequency for reference
        self.base_frequency = clean_df['ClaimNb'].sum() / clean_df['Exposure'].sum()
        print(f"📈 Base frequency: {self.base_frequency:.4f}")
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        exposure_train = clean_df.loc[X_train.index, 'Exposure']
        exposure_test = clean_df.loc[X_test.index, 'Exposure']
        
        # Fit Poisson model
        self.frequency_model = PoissonRegressor(
            alpha=0.01,  # Small regularization
            max_iter=1000,
            fit_intercept=True
        )
        
        # For Poisson regression, we need to use exposure as sample weights
        sample_weights = exposure_train
        
        try:
            self.frequency_model.fit(X_train, y_train, sample_weight=sample_weights)
            
            # Validate model
            y_pred_train = self.frequency_model.predict(X_train)
            y_pred_test = self.frequency_model.predict(X_test)
            
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            print("✅ Frequency model fitted successfully")
            print(f"📊 Train MAE: {train_mae:.4f}")
            print(f"📊 Test MAE: {test_mae:.4f}")
            print(f"📊 Feature importance: Top 5 features identified")
            
            # Show top features
            feature_importance = abs(self.frequency_model.coef_)
            top_features = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False).head()
            
            print("🔝 Top risk factors:")
            for _, row in top_features.iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
            
            return self.frequency_model
            
        except Exception as e:
            print(f"❌ Error fitting frequency model: {e}")
            return None
    
    def fit_severity_model(self, sev_df):
        """
        Fit Gamma model for claim severity using scikit-learn
        """
        if sev_df is None:
            print("❌ No severity data provided")
            return None
            
        print("🔧 Fitting severity model (Gamma Regression)...")
        
        # Prepare data
        exclude_cols = ['ClaimAmount', 'Severity']
        X, y, clean_df = self.prepare_modeling_data(sev_df, 'Severity', exclude_cols)
        
        if X is None or y is None:
            print("❌ Could not prepare severity data")
            return None
        
        print(f"📊 Severity modeling data: {X.shape}")
        
        # Use only features that exist in frequency model
        if self.feature_names:
            available_features = [col for col in self.feature_names if col in X.columns]
            X = X[available_features]
            print(f"📊 Using {len(available_features)} common features")
        
        # Calculate base severity
        self.base_severity = clean_df['Severity'].mean()
        print(f"📈 Base severity: €{self.base_severity:.2f}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Fit Gamma model
        self.severity_model = GammaRegressor(
            alpha=0.01,
            max_iter=1000,
            fit_intercept=True
        )
        
        try:
            self.severity_model.fit(X_train, y_train)
            
            # Validate model
            y_pred_train = self.severity_model.predict(X_train)
            y_pred_test = self.severity_model.predict(X_test)
            
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            print("✅ Severity model fitted successfully")
            print(f"📊 Train MAE: €{train_mae:.2f}")
            print(f"📊 Test MAE: €{test_mae:.2f}")
            
            return self.severity_model
            
        except Exception as e:
            print(f"❌ Error fitting severity model: {e}")
            # Fallback to simple mean model
            self.severity_model = None
            print("📊 Using base severity for predictions")
            return None
    
    def predict_frequency(self, X_new):
        """
        Predict claim frequency for new data
        """
        if self.frequency_model is None:
            return np.full(len(X_new), self.base_frequency)
        
        # Ensure features match training data
        if hasattr(X_new, 'columns'):
            available_features = [col for col in self.feature_names if col in X_new.columns]
            X_pred = X_new[available_features].copy()
            
            # Add missing features as zeros
            for feature in self.feature_names:
                if feature not in X_pred.columns:
                    X_pred[feature] = 0
            
            # Reorder to match training
            X_pred = X_pred[self.feature_names]
        else:
            X_pred = X_new
        
        return self.frequency_model.predict(X_pred)
    
    def predict_severity(self, X_new):
        """
        Predict claim severity for new data
        """
        if self.severity_model is None:
            return np.full(len(X_new), self.base_severity)
        
        # Ensure features match training data
        if hasattr(X_new, 'columns'):
            # Only use features that were in training data
            available_features = [col for col in self.feature_names if col in X_new.columns]
            
            if len(available_features) == 0:
                # Fallback to base severity if no features match
                return np.full(len(X_new), self.base_severity)
            
            X_pred = X_new[available_features].copy()
            
            # Get features the severity model was actually trained on
            if hasattr(self.severity_model, 'feature_names_in_'):
                trained_features = self.severity_model.feature_names_in_
            else:
                trained_features = available_features
            
            # Create prediction dataframe with only trained features
            X_final = pd.DataFrame(index=X_pred.index)
            for feature in trained_features:
                if feature in X_pred.columns:
                    X_final[feature] = X_pred[feature]
                else:
                    X_final[feature] = 0
            
        else:
            X_final = X_new
        
        try:
            return self.severity_model.predict(X_final)
        except:
            # Fallback to base severity if prediction fails
            return np.full(len(X_new), self.base_severity)
    
    def predict_pure_premium(self, X_new, exposure=1.0):
        """
        Predict pure premium (frequency × severity × exposure)
        """
        freq_pred = self.predict_frequency(X_new)
        sev_pred = self.predict_severity(X_new)
        
        pure_premium = freq_pred * sev_pred * exposure
        
        return pure_premium
    
    def create_summary_plots(self, freq_df, sev_df=None):
        """
        Create summary visualizations
        """
        print("📊 Creating model summary plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Frequency distribution
        axes[0, 0].hist(freq_df['Frequency'], bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].axvline(self.base_frequency, color='red', linestyle='--', 
                          label=f'Base Rate: {self.base_frequency:.4f}')
        axes[0, 0].set_title('Claim Frequency Distribution')
        axes[0, 0].set_xlabel('Frequency (Claims per Exposure)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend()
        
        # 2. Feature importance (if model exists)
        if self.frequency_model is not None:
            feature_importance = abs(self.frequency_model.coef_)
            top_10_idx = np.argsort(feature_importance)[-10:]
            
            axes[0, 1].barh(range(10), feature_importance[top_10_idx])
            axes[0, 1].set_yticks(range(10))
            axes[0, 1].set_yticklabels([self.feature_names[i] for i in top_10_idx])
            axes[0, 1].set_title('Top 10 Risk Factors (Frequency)')
            axes[0, 1].set_xlabel('Coefficient Magnitude')
        
        # 3. Severity distribution
        if sev_df is not None:
            axes[1, 0].hist(sev_df['Severity'], bins=50, alpha=0.7, color='lightcoral')
            axes[1, 0].axvline(self.base_severity, color='red', linestyle='--',
                              label=f'Base Severity: €{self.base_severity:.0f}')
            axes[1, 0].set_title('Claim Severity Distribution')
            axes[1, 0].set_xlabel('Severity (€)')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].legend()
            
            # Limit x-axis to 95th percentile for better visualization
            axes[1, 0].set_xlim(0, sev_df['Severity'].quantile(0.95))
        
        # 4. Pure premium distribution (sample)
        try:
            sample_data = freq_df.sample(n=min(1000, len(freq_df)))[self.feature_names]
            pure_premiums = self.predict_pure_premium(sample_data)
            
            axes[1, 1].hist(pure_premiums, bins=50, alpha=0.7, color='gold')
            axes[1, 1].set_title('Pure Premium Distribution (Sample)')
            axes[1, 1].set_xlabel('Pure Premium (€)')
            axes[1, 1].set_ylabel('Count')
        except Exception as e:
            print(f"⚠️ Could not create pure premium plot: {e}")
            axes[1, 1].text(0.5, 0.5, 'Pure Premium\nPlot Unavailable', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Pure Premium Distribution')
        
        plt.tight_layout()
        plt.savefig(Path("data") / 'model_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self, filepath):
        """
        Save trained models
        """
        model_data = {
            'frequency_model': self.frequency_model,
            'severity_model': self.severity_model,
            'feature_names': self.feature_names,
            'base_frequency': self.base_frequency,
            'base_severity': self.base_severity
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Models saved to {filepath}")
    
    def load_models(self, filepath):
        """
        Load trained models
        """
        model_data = joblib.load(filepath)
        self.frequency_model = model_data['frequency_model']
        self.severity_model = model_data['severity_model']
        self.feature_names = model_data['feature_names']
        self.base_frequency = model_data['base_frequency']
        self.base_severity = model_data['base_severity']
        print(f"✅ Models loaded from {filepath}")

def main():
    """
    Main execution function
    """
    print("🚀 Starting Simplified GLM Modeling")
    print("="*50)
    
    # Create data directory
    data_dir = Path("data")
    
    # Load processed data
    try:
        freq_df = pd.read_parquet(data_dir / "frequency_processed.parquet")
        print(f"✅ Loaded processed frequency data: {freq_df.shape}")
    except FileNotFoundError:
        print("❌ Processed frequency data not found. Please run feature engineering first.")
        return
    
    try:
        sev_df = pd.read_parquet(data_dir / "severity_processed.parquet")
        print(f"✅ Loaded processed severity data: {sev_df.shape}")
    except FileNotFoundError:
        print("⚠️ Processed severity data not found. Proceeding with frequency only.")
        sev_df = None
    
    # Initialize modeler
    modeler = SimplifiedActuarialModeler()
    
    # Fit frequency model
    freq_model = modeler.fit_frequency_model(freq_df)
    
    # Fit severity model if data available
    if sev_df is not None:
        sev_model = modeler.fit_severity_model(sev_df)
    
    # Create summary plots
    modeler.create_summary_plots(freq_df, sev_df)
    
    # Test prediction on sample data
    print("\n🧪 Testing prediction on sample data:")
    sample_data = freq_df.sample(n=3)[modeler.feature_names]
    for i, (idx, row) in enumerate(sample_data.iterrows()):
        sample_premium = modeler.predict_pure_premium(row.to_frame().T)
        print(f"Sample {i+1} pure premium: €{sample_premium[0]:.2f}")
    
    # Save models
    modeler.save_models(data_dir / "trained_models_simplified.pkl")
    
    print("\n🎉 Simplified GLM modeling complete!")
    print("Key outputs:")
    print(f"- Base frequency: {modeler.base_frequency:.4f}")
    if modeler.base_severity:
        print(f"- Base severity: €{modeler.base_severity:.2f}")
    print(f"- Features: {len(modeler.feature_names)} risk factors")
    print("- Models saved and ready for web interface!")

if __name__ == "__main__":
    main()