"""
GLM Modeling for Actuarial Pricing Simulator
Frequency-Severity modeling using statsmodels
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ActuarialGLMModeler:
    """
    GLM modeling class for actuarial pricing
    """
    
    def __init__(self):
        self.frequency_model = None
        self.severity_model = None
        self.frequency_results = None
        self.severity_results = None
        self.feature_names = []
        
    def fit_frequency_model(self, freq_df):
        """
        Fit Poisson GLM for claim frequency
        """
        print("🔧 Fitting frequency model (Poisson GLM)...")
        
        # Data validation and cleaning
        print("📊 Validating data...")
        
        # Remove any rows with invalid exposure or claims
        clean_df = freq_df.copy()
        clean_df = clean_df[clean_df['Exposure'] > 0]
        clean_df = clean_df[clean_df['ClaimNb'] >= 0]
        clean_df = clean_df[np.isfinite(clean_df['Exposure'])]
        clean_df = clean_df[np.isfinite(clean_df['ClaimNb'])]
        
        print(f"📉 Data cleaned: {freq_df.shape[0]} → {clean_df.shape[0]} rows")
        
        # Prepare target variable
        y_freq = clean_df['ClaimNb']
        
        # Prepare features - exclude target and exposure columns
        feature_columns = [col for col in clean_df.columns 
                          if col not in ['ClaimNb', 'Exposure', 'Frequency']]
        X_freq = clean_df[feature_columns]
        
        # Check for and remove any columns with no variation
        constant_cols = []
        for col in X_freq.columns:
            if X_freq[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            print(f"⚠️ Removing constant columns: {constant_cols}")
            X_freq = X_freq.drop(columns=constant_cols)
        
        # Check for perfect separation or near-perfect separation
        # Remove features that are too sparse (less than 10 observations in either category)
        sparse_cols = []
        for col in X_freq.columns:
            if X_freq[col].sum() < 10 or (len(X_freq) - X_freq[col].sum()) < 10:
                sparse_cols.append(col)
        
        if sparse_cols:
            print(f"⚠️ Removing sparse columns: {len(sparse_cols)} features")
            X_freq = X_freq.drop(columns=sparse_cols)
        
        # Final feature validation
        print(f"📊 Final feature matrix: {X_freq.shape}")
        
        # Add intercept
        X_freq = sm.add_constant(X_freq)
        
        # Store feature names (excluding constant)
        self.feature_names = [col for col in feature_columns if col not in constant_cols + sparse_cols]
        
        try:
            # Fit Poisson GLM with exposure as offset
            self.frequency_model = sm.GLM(
                y_freq, 
                X_freq, 
                family=sm.families.Poisson(),
                exposure=clean_df['Exposure']
            )
            
            # Fit the model with more robust settings
            self.frequency_results = self.frequency_model.fit(
                maxiter=100,
                method='bfgs',  # More stable optimization
                disp=False
            )
            
            print("✅ Frequency model fitted successfully")
            print(f"AIC: {self.frequency_results.aic:.2f}")
            print(f"Deviance: {self.frequency_results.deviance:.2f}")
            print(f"Converged: {self.frequency_results.converged}")
            
        except Exception as e:
            print(f"❌ Error fitting frequency model: {e}")
            print("🔄 Trying with regularization...")
            
            # Try with fewer features if convergence fails
            # Keep only the most important categorical features
            important_features = []
            for col in X_freq.columns:
                if any(keyword in col.lower() for keyword in ['drivage', 'vehage', 'region', 'area']):
                    important_features.append(col)
            
            if 'const' not in important_features:
                important_features = ['const'] + important_features
            
            X_freq_reduced = X_freq[important_features]
            print(f"🔧 Reduced features: {X_freq_reduced.shape}")
            
            self.frequency_model = sm.GLM(
                y_freq, 
                X_freq_reduced, 
                family=sm.families.Poisson(),
                exposure=clean_df['Exposure']
            )
            
            self.frequency_results = self.frequency_model.fit(maxiter=50, disp=False)
            
            print("✅ Frequency model fitted with reduced features")
            print(f"AIC: {self.frequency_results.aic:.2f}")
        
        return self.frequency_results
    
    def fit_severity_model(self, sev_df):
        """
        Fit Gamma GLM for claim severity
        """
        if sev_df is None:
            print("❌ No severity data provided")
            return None
            
        print("🔧 Fitting severity model (Gamma GLM)...")
        
        # Data validation and cleaning
        clean_df = sev_df.copy()
        clean_df = clean_df[clean_df['Severity'] > 0]
        clean_df = clean_df[np.isfinite(clean_df['Severity'])]
        
        print(f"📉 Severity data cleaned: {sev_df.shape[0]} → {clean_df.shape[0]} rows")
        
        # Prepare target variable
        y_sev = clean_df['Severity']
        
        # Prepare features (same as frequency model)
        feature_columns = [col for col in clean_df.columns 
                          if col not in ['ClaimAmount', 'Severity']]
        X_sev = clean_df[feature_columns]
        
        # Use only features that were successful in frequency model
        if hasattr(self, 'feature_names') and self.feature_names:
            available_features = [col for col in self.feature_names if col in X_sev.columns]
            if available_features:
                X_sev = X_sev[available_features]
                print(f"📊 Using {len(available_features)} features from frequency model")
        
        # Add intercept
        X_sev = sm.add_constant(X_sev)
        
        try:
            # Fit Gamma GLM with log link
            self.severity_model = sm.GLM(
                y_sev, 
                X_sev, 
                family=sm.families.Gamma(link=sm.families.links.log())
            )
            
            # Fit the model
            self.severity_results = self.severity_model.fit(
                maxiter=100,
                method='bfgs',
                disp=False
            )
            
            print("✅ Severity model fitted successfully")
            print(f"AIC: {self.severity_results.aic:.2f}")
            print(f"Deviance: {self.severity_results.deviance:.2f}")
            print(f"Converged: {self.severity_results.converged}")
            
        except Exception as e:
            print(f"❌ Error fitting severity model: {e}")
            print("🔄 Trying simplified model...")
            
            # Try with just a few key features
            key_features = ['const']
            for col in X_sev.columns:
                if any(keyword in col.lower() for keyword in ['drivage', 'region']):
                    key_features.append(col)
                    if len(key_features) >= 10:  # Limit features
                        break
            
            X_sev_simple = X_sev[key_features]
            
            self.severity_model = sm.GLM(
                y_sev, 
                X_sev_simple, 
                family=sm.families.Gamma(link=sm.families.links.log())
            )
            
            self.severity_results = self.severity_model.fit(maxiter=50, disp=False)
            print("✅ Severity model fitted with simplified features")
        
        return self.severity_results
    
    def predict_frequency(self, X_new, exposure=1.0):
        """
        Predict claim frequency for new data
        """
        if self.frequency_results is None:
            raise ValueError("Frequency model not fitted yet")
        
        # Ensure X_new has the same features as training data
        X_pred = X_new.copy()
        
        # Add intercept
        X_pred = sm.add_constant(X_pred)
        
        # Make prediction
        freq_pred = self.frequency_results.predict(X_pred) * exposure
        
        return freq_pred
    
    def predict_severity(self, X_new):
        """
        Predict claim severity for new data
        """
        if self.severity_results is None:
            raise ValueError("Severity model not fitted yet")
        
        # Ensure X_new has the same features as training data
        X_pred = X_new.copy()
        
        # Add intercept
        X_pred = sm.add_constant(X_pred)
        
        # Make prediction
        sev_pred = self.severity_results.predict(X_pred)
        
        return sev_pred
    
    def predict_pure_premium(self, X_new, exposure=1.0):
        """
        Predict pure premium (frequency × severity)
        """
        freq_pred = self.predict_frequency(X_new, exposure)
        
        if self.severity_results is not None:
            sev_pred = self.predict_severity(X_new)
            pure_premium = freq_pred * sev_pred
        else:
            # If no severity model, use average severity from training data
            print("⚠️ No severity model available, using simplified calculation")
            pure_premium = freq_pred * 1000  # Placeholder average severity
        
        return pure_premium
    
    def get_coefficient_interpretation(self):
        """
        Get coefficient interpretation for both models
        """
        interpretation = {}
        
        if self.frequency_results is not None:
            freq_coefs = self.frequency_results.params
            freq_pvalues = self.frequency_results.pvalues
            
            interpretation['frequency'] = pd.DataFrame({
                'coefficient': freq_coefs,
                'exp_coefficient': np.exp(freq_coefs),
                'p_value': freq_pvalues,
                'significant': freq_pvalues < 0.05
            })
        
        if self.severity_results is not None:
            sev_coefs = self.severity_results.params
            sev_pvalues = self.severity_results.pvalues
            
            interpretation['severity'] = pd.DataFrame({
                'coefficient': sev_coefs,
                'exp_coefficient': np.exp(sev_coefs),
                'p_value': sev_pvalues,
                'significant': sev_pvalues < 0.05
            })
        
        return interpretation
    
    def calculate_factor_impacts(self, base_data):
        """
        Calculate the impact of each factor on the premium
        """
        if self.frequency_results is None:
            return None
        
        impacts = {}
        base_premium = self.predict_pure_premium(base_data)
        
        # Test impact of each feature
        for feature in self.feature_names:
            if feature in base_data.columns:
                # Find all related one-hot encoded columns
                related_cols = [col for col in base_data.columns if col.startswith(feature + '_')]
                
                if related_cols:
                    # For categorical variables, test different categories
                    for col in related_cols:
                        test_data = base_data.copy()
                        # Set all related columns to 0
                        for rel_col in related_cols:
                            test_data[rel_col] = 0
                        # Set current column to 1
                        test_data[col] = 1
                        
                        test_premium = self.predict_pure_premium(test_data)
                        impact = (test_premium / base_premium - 1).iloc[0] if hasattr(test_premium, 'iloc') else (test_premium / base_premium - 1)
                        impacts[col] = impact
        
        return impacts
    
    def validate_models(self, freq_df, sev_df=None):
        """
        Validate model performance
        """
        print("🔍 Validating model performance...")
        
        # Frequency model validation
        if self.frequency_results is not None:
            # Prepare features
            feature_columns = [col for col in freq_df.columns 
                              if col not in ['ClaimNb', 'Exposure', 'Frequency']]
            X_freq = freq_df[feature_columns]
            
            # Predict frequencies
            freq_pred = self.predict_frequency(X_freq, freq_df['Exposure'])
            freq_actual = freq_df['ClaimNb']
            
            # Calculate metrics
            freq_mae = mean_absolute_error(freq_actual, freq_pred)
            freq_rmse = np.sqrt(mean_squared_error(freq_actual, freq_pred))
            
            print(f"Frequency Model - MAE: {freq_mae:.4f}, RMSE: {freq_rmse:.4f}")
        
        # Severity model validation
        if self.severity_results is not None and sev_df is not None:
            feature_columns = [col for col in sev_df.columns 
                              if col not in ['ClaimAmount', 'Severity']]
            X_sev = sev_df[feature_columns]
            
            sev_pred = self.predict_severity(X_sev)
            sev_actual = sev_df['Severity']
            
            sev_mae = mean_absolute_error(sev_actual, sev_pred)
            sev_rmse = np.sqrt(mean_squared_error(sev_actual, sev_pred))
            
            print(f"Severity Model - MAE: {sev_mae:.2f}, RMSE: {sev_rmse:.2f}")
    
    def create_model_plots(self, freq_df, sev_df=None):
        """
        Create diagnostic plots for the models
        """
        print("📊 Creating model diagnostic plots...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        
        if self.frequency_results is not None:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Residuals vs Fitted (Frequency)
            feature_columns = [col for col in freq_df.columns 
                              if col not in ['ClaimNb', 'Exposure', 'Frequency']]
            X_freq = freq_df[feature_columns]
            freq_pred = self.predict_frequency(X_freq, freq_df['Exposure'])
            freq_residuals = freq_df['ClaimNb'] - freq_pred
            
            axes[0, 0].scatter(freq_pred, freq_residuals, alpha=0.5)
            axes[0, 0].axhline(y=0, color='red', linestyle='--')
            axes[0, 0].set_title('Frequency: Residuals vs Fitted')
            axes[0, 0].set_xlabel('Fitted Values')
            axes[0, 0].set_ylabel('Residuals')
            
            # 2. QQ Plot for frequency residuals
            from scipy import stats
            stats.probplot(freq_residuals, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Frequency: Q-Q Plot')
            
            # 3. Coefficient plot
            coef_data = self.get_coefficient_interpretation()
            if 'frequency' in coef_data:
                freq_coef = coef_data['frequency']
                significant_coef = freq_coef[freq_coef['significant']].copy()
                significant_coef = significant_coef.drop('const', errors='ignore')
                
                y_pos = np.arange(len(significant_coef))
                axes[1, 0].barh(y_pos, significant_coef['coefficient'])
                axes[1, 0].set_yticks(y_pos)
                axes[1, 0].set_yticklabels(significant_coef.index, fontsize=8)
                axes[1, 0].set_title('Frequency: Significant Coefficients')
                axes[1, 0].set_xlabel('Coefficient Value')
            
            # 4. Predicted vs Actual
            axes[1, 1].scatter(freq_df['ClaimNb'], freq_pred, alpha=0.5)
            axes[1, 1].plot([freq_df['ClaimNb'].min(), freq_df['ClaimNb'].max()], 
                           [freq_df['ClaimNb'].min(), freq_df['ClaimNb'].max()], 'r--')
            axes[1, 1].set_title('Frequency: Predicted vs Actual')
            axes[1, 1].set_xlabel('Actual Claims')
            axes[1, 1].set_ylabel('Predicted Claims')
            
            plt.tight_layout()
            plt.savefig(Path("data") / 'frequency_model_diagnostics.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Severity model plots
        if self.severity_results is not None and sev_df is not None:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            feature_columns = [col for col in sev_df.columns 
                              if col not in ['ClaimAmount', 'Severity']]
            X_sev = sev_df[feature_columns]
            sev_pred = self.predict_severity(X_sev)
            sev_residuals = sev_df['Severity'] - sev_pred
            
            # Severity diagnostic plots
            axes[0, 0].scatter(sev_pred, sev_residuals, alpha=0.5)
            axes[0, 0].axhline(y=0, color='red', linestyle='--')
            axes[0, 0].set_title('Severity: Residuals vs Fitted')
            axes[0, 0].set_xlabel('Fitted Values')
            axes[0, 0].set_ylabel('Residuals')
            
            stats.probplot(sev_residuals, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Severity: Q-Q Plot')
            
            # Severity coefficients
            if 'severity' in coef_data:
                sev_coef = coef_data['severity']
                significant_coef = sev_coef[sev_coef['significant']].copy()
                significant_coef = significant_coef.drop('const', errors='ignore')
                
                y_pos = np.arange(len(significant_coef))
                axes[1, 0].barh(y_pos, significant_coef['coefficient'])
                axes[1, 0].set_yticks(y_pos)
                axes[1, 0].set_yticklabels(significant_coef.index, fontsize=8)
                axes[1, 0].set_title('Severity: Significant Coefficients')
                axes[1, 0].set_xlabel('Coefficient Value')
            
            axes[1, 1].scatter(sev_df['Severity'], sev_pred, alpha=0.5)
            axes[1, 1].plot([sev_df['Severity'].min(), sev_df['Severity'].max()], 
                           [sev_df['Severity'].min(), sev_df['Severity'].max()], 'r--')
            axes[1, 1].set_title('Severity: Predicted vs Actual')
            axes[1, 1].set_xlabel('Actual Severity')
            axes[1, 1].set_ylabel('Predicted Severity')
            
            plt.tight_layout()
            plt.savefig(Path("data") / 'severity_model_diagnostics.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def save_models(self, filepath):
        """
        Save trained models
        """
        model_data = {
            'frequency_results': self.frequency_results,
            'severity_results': self.severity_results,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Models saved to {filepath}")
    
    def load_models(self, filepath):
        """
        Load trained models
        """
        model_data = joblib.load(filepath)
        self.frequency_results = model_data['frequency_results']
        self.severity_results = model_data['severity_results']
        self.feature_names = model_data['feature_names']
        print(f"✅ Models loaded from {filepath}")

def main():
    """
    Main execution function
    """
    print("🚀 Starting GLM Modeling")
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
    modeler = ActuarialGLMModeler()
    
    # Fit frequency model
    freq_results = modeler.fit_frequency_model(freq_df)
    print("\n📊 Frequency Model Summary:")
    print(freq_results.summary())
    
    # Fit severity model if data available
    if sev_df is not None:
        sev_results = modeler.fit_severity_model(sev_df)
        print("\n📊 Severity Model Summary:")
        print(sev_results.summary())
    
    # Validate models
    modeler.validate_models(freq_df, sev_df)
    
    # Create diagnostic plots
    modeler.create_model_plots(freq_df, sev_df)
    
    # Show coefficient interpretation
    print("\n📈 Coefficient Interpretation:")
    coef_interpretation = modeler.get_coefficient_interpretation()
    for model_type, coefs in coef_interpretation.items():
        print(f"\n{model_type.upper()} MODEL:")
        print(coefs.head(10))
    
    # Save models
    modeler.save_models(data_dir / "trained_models.pkl")
    
    # Test prediction on a sample
    print("\n🧪 Testing prediction on sample data:")
    sample_data = freq_df.iloc[[0]][modeler.feature_names]
    sample_premium = modeler.predict_pure_premium(sample_data)
    print(f"Sample pure premium: €{sample_premium.iloc[0] if hasattr(sample_premium, 'iloc') else sample_premium:.2f}")
    
    print("\n🎉 GLM modeling complete!")
    print("Next steps:")
    print("1. Review model diagnostics and coefficients")
    print("2. Build the Streamlit web interface")

if __name__ == "__main__":
    main()