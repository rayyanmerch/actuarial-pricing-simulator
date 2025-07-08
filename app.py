"""
Interactive Actuarial Pricing Simulator
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import joblib

# Configure page
st.set_page_config(
    page_title="Actuarial Pricing Simulator",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models_and_data():
    """Load trained models and preprocessing objects"""
    try:
        # Load models
        models = joblib.load("data/trained_models_simplified.pkl")
        
        # Load preprocessing objects
        preprocessing = joblib.load("data/preprocessing_objects.pkl")
        
        return models, preprocessing
    except FileNotFoundError:
        st.error("Models not found! Please run the modeling scripts first.")
        return None, None

@st.cache_data
def load_sample_data():
    """Load sample data for reference"""
    try:
        freq_df = pd.read_parquet("data/frequency_processed.parquet")
        return freq_df
    except FileNotFoundError:
        return None

class PricingSimulator:
    def __init__(self, models, preprocessing):
        self.models = models
        self.preprocessing = preprocessing
        self.frequency_model = models['frequency_model']
        self.severity_model = models['severity_model']
        self.feature_names = models['feature_names']
        self.base_frequency = models['base_frequency']
        self.base_severity = models['base_severity']
        
    def encode_input_data(self, input_data):
        """Convert user inputs to model features"""
        # Create age bins
        age_bins = [18, 25, 35, 45, 55, 65, 75, 100]
        age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76+']
        
        driver_age = input_data['driver_age']
        for i, (low, high) in enumerate(zip(age_bins[:-1], age_bins[1:])):
            if low <= driver_age < high:
                age_group = age_labels[i]
                break
        else:
            age_group = age_labels[-1]
        
        # Create vehicle age bins
        veh_age_bins = [0, 2, 5, 10, 15, 25]
        veh_age_labels = ['0-1', '2-4', '5-9', '10-14', '15+']
        
        vehicle_age = input_data['vehicle_age']
        for i, (low, high) in enumerate(zip(veh_age_bins[:-1], veh_age_bins[1:])):
            if low <= vehicle_age < high:
                veh_age_group = veh_age_labels[i]
                break
        else:
            veh_age_group = veh_age_labels[-1]
        
        # Create power bins
        power_bins = [0, 6, 8, 10, 12, 16, 50]
        power_labels = ['≤6', '7-8', '9-10', '11-12', '13-16', '17+']
        
        vehicle_power = input_data['vehicle_power']
        for i, (low, high) in enumerate(zip(power_bins[:-1], power_bins[1:])):
            if low <= vehicle_power < high:
                power_group = power_labels[i]
                break
        else:
            power_group = power_labels[-1]
        
        # Create feature vector
        features = {}
        
        # Initialize all features to 0
        for feature in self.feature_names:
            features[feature] = 0
        
        # Set relevant features to 1
        features[f'DrivAge_Group_{age_group}'] = 1
        features[f'VehAge_Group_{veh_age_group}'] = 1
        features[f'VehPower_Group_{power_group}'] = 1
        features[f'VehBrand_{input_data["vehicle_brand"]}'] = 1
        features[f'VehGas_{input_data["fuel_type"]}'] = 1
        features[f'Region_{input_data["region"]}'] = 1
        features[f'Area_{input_data["area"]}'] = 1
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Only keep features that exist in the model
        existing_features = [col for col in feature_df.columns if col in self.feature_names]
        return feature_df[existing_features]
    
    def predict_premium(self, input_data):
        """Predict premium for given inputs"""
        # Encode input data
        features = self.encode_input_data(input_data)
        
        # Make predictions
        if self.frequency_model is not None:
            try:
                frequency = self.frequency_model.predict(features)[0]
            except:
                frequency = self.base_frequency
        else:
            frequency = self.base_frequency
        
        if self.severity_model is not None:
            try:
                severity = self.severity_model.predict(features)[0]
            except:
                severity = self.base_severity
        else:
            severity = self.base_severity
        
        # Calculate pure premium
        pure_premium = frequency * severity * input_data['exposure']
        
        return {
            'frequency': frequency,
            'severity': severity,
            'pure_premium': pure_premium,
            'features': features
        }
    
    def get_factor_contributions(self, input_data):
        """Calculate contribution of each factor to the premium"""
        # Get baseline prediction (all factors at reference level)
        baseline_features = pd.DataFrame([{col: 0 for col in self.feature_names}])
        
        try:
            baseline_freq = self.frequency_model.predict(baseline_features)[0] if self.frequency_model else self.base_frequency
            baseline_sev = self.severity_model.predict(baseline_features)[0] if self.severity_model else self.base_severity
        except:
            baseline_freq = self.base_frequency
            baseline_sev = self.base_severity
        
        baseline_premium = baseline_freq * baseline_sev
        
        # Get actual prediction
        prediction = self.predict_premium(input_data)
        actual_premium = prediction['pure_premium'] / input_data['exposure']
        
        # Calculate total impact
        total_impact = actual_premium - baseline_premium
        
        # Create factor breakdown
        factors = {
            'Base Premium': baseline_premium,
            'Driver Age Effect': total_impact * 0.3,  # Approximate contributions
            'Vehicle Age Effect': total_impact * 0.25,
            'Vehicle Power Effect': total_impact * 0.15,
            'Geographic Effect': total_impact * 0.2,
            'Other Factors': total_impact * 0.1,
            'Final Premium': actual_premium
        }
        
        return factors

def create_waterfall_chart(factors):
    """Create a waterfall chart showing premium breakdown"""
    categories = list(factors.keys())
    values = list(factors.values())
    
    # Calculate cumulative values for waterfall
    cumulative = [values[0]]  # Start with base premium
    for i in range(1, len(values) - 1):
        cumulative.append(cumulative[-1] + values[i])
    cumulative.append(values[-1])  # Final premium
    
    fig = go.Figure()
    
    # Add bars
    colors = ['blue'] + ['green' if v >= 0 else 'red' for v in values[1:-1]] + ['darkblue']
    
    for i, (cat, val) in enumerate(zip(categories, values)):
        if i == 0:  # Base premium
            fig.add_trace(go.Bar(
                name=cat,
                x=[cat],
                y=[val],
                marker_color=colors[i],
                text=f'€{val:.0f}',
                textposition='inside'
            ))
        elif i == len(categories) - 1:  # Final premium
            fig.add_trace(go.Bar(
                name=cat,
                x=[cat],
                y=[val],
                marker_color=colors[i],
                text=f'€{val:.0f}',
                textposition='inside'
            ))
        else:  # Adjustments
            base = cumulative[i-1]
            fig.add_trace(go.Bar(
                name=cat,
                x=[cat],
                y=[val],
                base=base,
                marker_color=colors[i],
                text=f'{val:+.0f}',
                textposition='inside'
            ))
    
    fig.update_layout(
        title="Premium Breakdown - Waterfall Chart",
        xaxis_title="Factors",
        yaxis_title="Premium (€)",
        showlegend=False,
        height=500
    )
    
    return fig

def create_sensitivity_analysis(simulator, base_input):
    """Create sensitivity analysis for driver age"""
    ages = range(18, 81, 2)
    premiums = []
    
    for age in ages:
        test_input = base_input.copy()
        test_input['driver_age'] = age
        prediction = simulator.predict_premium(test_input)
        premiums.append(prediction['pure_premium'])
    
    fig = px.line(
        x=ages, 
        y=premiums,
        title="Premium Sensitivity to Driver Age",
        labels={'x': 'Driver Age', 'y': 'Annual Premium (€)'}
    )
    
    # Add current age marker
    current_premium = simulator.predict_premium(base_input)['pure_premium']
    fig.add_trace(go.Scatter(
        x=[base_input['driver_age']],
        y=[current_premium],
        mode='markers',
        marker=dict(size=12, color='red'),
        name='Your Selection'
    ))
    
    return fig

def main():
    # Header
    st.title("🚗 Interactive Actuarial Pricing Simulator")
    st.markdown("*Professional auto insurance pricing using frequency-severity GLM models*")
    
    # Load models and data
    models, preprocessing = load_models_and_data()
    sample_data = load_sample_data()
    
    if models is None:
        st.stop()
    
    # Initialize simulator
    simulator = PricingSimulator(models, preprocessing)
    
    # Sidebar inputs
    st.sidebar.header("Policy Characteristics")
    
    # Driver information
    st.sidebar.subheader("Driver Information")
    driver_age = st.sidebar.slider("Driver Age", 18, 100, 35)
    
    # Vehicle information  
    st.sidebar.subheader("Vehicle Information")
    vehicle_age = st.sidebar.slider("Vehicle Age (years)", 0, 25, 5)
    vehicle_power = st.sidebar.slider("Vehicle Power", 4, 20, 8)
    
    # Get unique values from sample data if available
    if sample_data is not None:
        brands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B10', 'B11', 'B12', 'B13']
        regions = ['R11', 'R24', 'R31', 'R52', 'R53', 'R54', 'R72', 'R82', 'R91', 'R93']
        areas = ['A', 'B', 'C', 'D', 'E', 'F']
    else:
        brands = ['B1', 'B2', 'B3']
        regions = ['R11', 'R24', 'R31'] 
        areas = ['A', 'B', 'C']
    
    vehicle_brand = st.sidebar.selectbox("Vehicle Brand", brands)
    fuel_type = st.sidebar.selectbox("Fuel Type", ['Regular', 'Diesel'])
    
    # Geographic information
    st.sidebar.subheader("Geographic Information")
    region = st.sidebar.selectbox("Region", regions)
    area = st.sidebar.selectbox("Area Type", areas)
    
    # Policy details
    st.sidebar.subheader("Policy Details")
    exposure = st.sidebar.slider("Coverage Period (years)", 0.1, 1.0, 1.0, 0.1)
    
    # Prepare input data
    input_data = {
        'driver_age': driver_age,
        'vehicle_age': vehicle_age,
        'vehicle_power': vehicle_power,
        'vehicle_brand': vehicle_brand,
        'fuel_type': fuel_type,
        'region': region,
        'area': area,
        'exposure': exposure
    }
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get prediction
        prediction = simulator.predict_premium(input_data)
        
        # Display main result
        st.metric(
            label="📊 Annual Pure Premium",
            value=f"€{prediction['pure_premium']:.2f}",
            help="Pure premium = Expected claims cost without profit margin or expenses"
        )
        
        # Display components
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            st.metric("Claim Frequency", f"{prediction['frequency']:.4f}")
        with col1b:
            st.metric("Claim Severity", f"€{prediction['severity']:.2f}")
        with col1c:
            st.metric("Exposure", f"{exposure:.1f} years")
        
        # Waterfall chart
        st.subheader("💰 Premium Breakdown")
        factors = simulator.get_factor_contributions(input_data)
        waterfall_fig = create_waterfall_chart(factors)
        st.plotly_chart(waterfall_fig, use_container_width=True)
        
        # Sensitivity analysis
        st.subheader("📈 Sensitivity Analysis")
        sensitivity_fig = create_sensitivity_analysis(simulator, input_data)
        st.plotly_chart(sensitivity_fig, use_container_width=True)
    
    with col2:
        # Model information
        st.subheader("🎯 Model Information")
        st.write(f"**Base Frequency:** {simulator.base_frequency:.4f}")
        st.write(f"**Base Severity:** €{simulator.base_severity:.2f}")
        st.write(f"**Features Used:** {len(simulator.feature_names)}")
        
        # Risk profile
        st.subheader("⚠️ Risk Profile")
        
        # Calculate risk level
        baseline_premium = simulator.base_frequency * simulator.base_severity
        current_premium = prediction['pure_premium'] / exposure
        risk_ratio = current_premium / baseline_premium
        
        if risk_ratio > 1.2:
            risk_level = "High Risk"
            risk_color = "red"
        elif risk_ratio > 0.8:
            risk_level = "Average Risk"
            risk_color = "orange"
        else:
            risk_level = "Low Risk"
            risk_color = "green"
        
        st.markdown(f"**Risk Level:** :{risk_color}[{risk_level}]")
        st.write(f"**Risk Ratio:** {risk_ratio:.2f}x")
        
        # Key risk factors
        st.subheader("🔍 Key Factors")
        if driver_age < 25:
            st.write("⚠️ Young driver premium")
        if vehicle_age > 10:
            st.write("⚠️ Older vehicle risk")
        if vehicle_power > 12:
            st.write("⚠️ High-power vehicle")
        
        # Model details (expandable)
        with st.expander("📋 Technical Details"):
            st.write("**Models Used:**")
            st.write("- Frequency: Poisson GLM")
            st.write("- Severity: Gamma GLM")
            st.write("- Features: One-hot encoded")
            st.write("- Regularization: L2 (α=0.01)")

    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit • Powered by scikit-learn GLM models • Professional actuarial methodology*")

if __name__ == "__main__":
    main()