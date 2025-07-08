"""
US-Focused Interactive Actuarial Pricing Simulator
Enhanced with US states, realistic pricing, and additional features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="US Auto Insurance Pricing Simulator",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# US State risk multipliers (based on typical insurance data)
US_STATE_RISK = {
    'California': 1.15, 'Texas': 1.05, 'Florida': 1.25, 'New York': 1.20,
    'Pennsylvania': 0.95, 'Illinois': 1.10, 'Ohio': 0.90, 'Georgia': 1.05,
    'North Carolina': 0.85, 'Michigan': 1.30, 'New Jersey': 1.35, 'Virginia': 0.80,
    'Washington': 0.95, 'Arizona': 1.00, 'Massachusetts': 1.10, 'Tennessee': 0.85,
    'Indiana': 0.80, 'Missouri': 0.90, 'Maryland': 1.05, 'Wisconsin': 0.75,
    'Colorado': 0.90, 'Minnesota': 0.70, 'South Carolina': 0.95, 'Alabama': 0.85,
    'Louisiana': 1.40, 'Kentucky': 0.80, 'Oregon': 0.85, 'Oklahoma': 0.90,
    'Connecticut': 1.15, 'Utah': 0.75, 'Iowa': 0.65, 'Nevada': 1.10,
    'Arkansas': 0.80, 'Mississippi': 0.85, 'Kansas': 0.75, 'New Mexico': 0.95,
    'Nebraska': 0.70, 'West Virginia': 0.85, 'Idaho': 0.70, 'Hawaii': 1.20,
    'New Hampshire': 0.80, 'Maine': 0.75, 'Montana': 0.70, 'Rhode Island': 1.25,
    'Delaware': 1.00, 'South Dakota': 0.65, 'North Dakota': 0.60, 'Alaska': 0.90,
    'Vermont': 0.75, 'Wyoming': 0.65
}

# Vehicle makes with typical risk factors
VEHICLE_MAKES = {
    'Toyota': 0.90, 'Honda': 0.85, 'Ford': 1.00, 'Chevrolet': 1.05,
    'Nissan': 0.95, 'Hyundai': 0.88, 'Subaru': 0.82, 'Volkswagen': 1.02,
    'BMW': 1.25, 'Mercedes-Benz': 1.30, 'Audi': 1.28, 'Lexus': 1.10,
    'Acura': 1.08, 'Infiniti': 1.15, 'Cadillac': 1.20, 'Lincoln': 1.18,
    'Jeep': 1.05, 'Ram': 1.08, 'GMC': 1.03, 'Buick': 0.95,
    'Chrysler': 1.00, 'Dodge': 1.12, 'Kia': 0.92, 'Mazda': 0.98,
    'Tesla': 1.35, 'Volvo': 0.88, 'Land Rover': 1.40, 'Porsche': 1.50
}

@st.cache_resource
def load_base_models():
    """Load the base European models for calibration"""
    # Instead of loading pickle files, return None to use default values
    # This avoids the sklearn version compatibility issue
    return None

class USPricingSimulator:
    def __init__(self, base_models=None):
        if base_models:
            self.base_frequency = base_models['base_frequency'] * 0.8  # US typically lower
            self.base_severity = base_models['base_severity'] * 1.1 * 1.5  # Convert EUR to USD and adjust
        else:
            self.base_frequency = 0.08  # Typical US auto frequency
            self.base_severity = 2800  # Typical US claim severity in USD
    
    def calculate_age_factor(self, age):
        """Calculate age-based risk factor using US actuarial curves"""
        if age < 20:
            return 2.8
        elif age < 25:
            return 2.2
        elif age < 30:
            return 1.4
        elif age < 40:
            return 1.0
        elif age < 50:
            return 0.9
        elif age < 65:
            return 0.85
        elif age < 75:
            return 1.0
        else:
            return 1.3
    
    def calculate_vehicle_age_factor(self, vehicle_age):
        """Vehicle age risk factor"""
        if vehicle_age == 0:
            return 1.2  # New cars are expensive to repair
        elif vehicle_age <= 3:
            return 1.0
        elif vehicle_age <= 8:
            return 0.95
        elif vehicle_age <= 15:
            return 1.1
        else:
            return 1.25  # Very old cars
    
    def calculate_credit_score_factor(self, credit_score):
        """Credit score impact (legal in most US states)"""
        if credit_score >= 800:
            return 0.75
        elif credit_score >= 740:
            return 0.85
        elif credit_score >= 670:
            return 1.0
        elif credit_score >= 580:
            return 1.15
        else:
            return 1.30
    
    def calculate_mileage_factor(self, annual_mileage):
        """Annual mileage risk factor"""
        if annual_mileage < 5000:
            return 0.80
        elif annual_mileage < 10000:
            return 0.90
        elif annual_mileage < 15000:
            return 1.0
        elif annual_mileage < 20000:
            return 1.15
        else:
            return 1.35
    
    def predict_premium(self, inputs):
        """Calculate premium based on US factors"""
        # Base calculation
        frequency = self.base_frequency
        severity = self.base_severity
        
        # Apply risk factors
        age_factor = self.calculate_age_factor(inputs['age'])
        vehicle_age_factor = self.calculate_vehicle_age_factor(inputs['vehicle_age'])
        state_factor = US_STATE_RISK.get(inputs['state'], 1.0)
        make_factor = VEHICLE_MAKES.get(inputs['vehicle_make'], 1.0)
        credit_factor = self.calculate_credit_score_factor(inputs['credit_score'])
        mileage_factor = self.calculate_mileage_factor(inputs['annual_mileage'])
        
        # Coverage adjustments
        coverage_factor = inputs['coverage_level']  # 0.5 = minimum, 1.0 = full, 1.5 = premium
        
        # Calculate frequency and severity with factors
        adjusted_frequency = frequency * age_factor * mileage_factor * state_factor * 0.8
        adjusted_severity = severity * make_factor * vehicle_age_factor * coverage_factor
        
        # Pure premium calculation
        pure_premium = adjusted_frequency * adjusted_severity
        
        # Apply credit score to final premium
        final_premium = pure_premium * credit_factor
        
        # Add profit margin and expenses (typical 25-30%)
        commercial_premium = final_premium * 1.28
        
        return {
            'pure_premium': pure_premium,
            'commercial_premium': commercial_premium,
            'frequency': adjusted_frequency,
            'severity': adjusted_severity,
            'factors': {
                'Age Factor': age_factor,
                'Vehicle Age': vehicle_age_factor,
                'State Risk': state_factor,
                'Vehicle Make': make_factor,
                'Credit Score': credit_factor,
                'Mileage': mileage_factor,
                'Coverage Level': coverage_factor
            }
        }

def create_us_waterfall_chart(base_premium, factors, final_premium):
    """Create waterfall chart for US pricing"""
    # Calculate step-by-step impacts
    cumulative = base_premium
    steps = [('Base Premium', base_premium, base_premium)]
    
    factor_impacts = {}
    for factor_name, factor_value in factors.items():
        if factor_value != 1.0:
            impact = cumulative * (factor_value - 1.0)
            factor_impacts[factor_name] = impact
            cumulative += impact
            steps.append((factor_name, impact, cumulative))
    
    steps.append(('Final Premium', final_premium - cumulative, final_premium))
    
    fig = go.Figure()
    
    colors = ['blue']
    for name, impact, _ in steps[1:-1]:
        colors.append('green' if impact >= 0 else 'red')
    colors.append('darkblue')
    
    for i, (name, value, cumulative) in enumerate(steps):
        if i == 0 or i == len(steps) - 1:
            # Base and final bars
            fig.add_trace(go.Bar(
                name=name,
                x=[name],
                y=[cumulative if i == 0 else value],
                marker_color=colors[i],
                text=f'${value:.0f}' if i == 0 else f'${cumulative:.0f}',
                textposition='inside',
                showlegend=False
            ))
        else:
            # Factor adjustments
            fig.add_trace(go.Bar(
                name=name,
                x=[name],
                y=[abs(value)],
                base=steps[i-1][2] if value >= 0 else steps[i-1][2] + value,
                marker_color=colors[i],
                text=f'{value:+.0f}',
                textposition='inside',
                showlegend=False
            ))
    
    fig.update_layout(
        title="Premium Calculation Breakdown",
        xaxis_title="Risk Factors",
        yaxis_title="Premium ($)",
        height=500,
        xaxis_tickangle=-45
    )
    
    return fig

def main():
    # Header
    st.title("🇺🇸 US Auto Insurance Pricing Simulator")
    st.markdown("*Professional actuarial pricing with US-specific risk factors*")
    
    # Load base models for calibration
    base_models = load_base_models()
    simulator = USPricingSimulator(base_models)
    
    # Sidebar inputs
    st.sidebar.header("Driver & Vehicle Information")
    
    # Driver information
    st.sidebar.subheader("👤 Driver Details")
    age = st.sidebar.slider("Age", 16, 85, 35)
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 720)
    annual_mileage = st.sidebar.slider("Annual Mileage", 1000, 30000, 12000)
    
    # Vehicle information
    st.sidebar.subheader("🚗 Vehicle Details")
    vehicle_age = st.sidebar.slider("Vehicle Age (years)", 0, 25, 5)
    vehicle_make = st.sidebar.selectbox("Vehicle Make", sorted(VEHICLE_MAKES.keys()))
    
    # Location
    st.sidebar.subheader("📍 Location")
    state = st.sidebar.selectbox("State", sorted(US_STATE_RISK.keys()))
    
    # Coverage
    st.sidebar.subheader("🛡️ Coverage Level")
    coverage_options = {
        "Minimum Coverage": 0.5,
        "Standard Coverage": 1.0,
        "Full Coverage": 1.3,
        "Premium Coverage": 1.6
    }
    coverage_choice = st.sidebar.selectbox("Coverage Level", list(coverage_options.keys()), index=1)
    coverage_level = coverage_options[coverage_choice]
    
    # Prepare inputs
    inputs = {
        'age': age,
        'credit_score': credit_score,
        'annual_mileage': annual_mileage,
        'vehicle_age': vehicle_age,
        'vehicle_make': vehicle_make,
        'state': state,
        'coverage_level': coverage_level
    }
    
    # Calculate premium
    result = simulator.predict_premium(inputs)
    
    # Main display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Main results
        st.metric(
            label="💰 Annual Premium (USD)",
            value=f"${result['commercial_premium']:.2f}",
            delta=f"${result['commercial_premium'] - (simulator.base_frequency * simulator.base_severity * 1.28):.2f}",
            help="Full premium including profit margin and expenses"
        )
        
        # Components
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            st.metric("Pure Premium", f"${result['pure_premium']:.2f}")
        with col1b:
            st.metric("Expected Frequency", f"{result['frequency']:.4f}")
        with col1c:
            st.metric("Expected Severity", f"${result['severity']:.2f}")
        
        # Waterfall chart
        st.subheader("📊 Premium Breakdown")
        base_premium = simulator.base_frequency * simulator.base_severity
        waterfall_fig = create_us_waterfall_chart(
            base_premium, 
            result['factors'], 
            result['commercial_premium']
        )
        st.plotly_chart(waterfall_fig, use_container_width=True)
        
        # Comparison with national average
        st.subheader("🗺️ State Comparison")
        national_avg = simulator.base_frequency * simulator.base_severity * 1.28
        state_premium = result['commercial_premium']
        
        comparison_data = pd.DataFrame({
            'Category': ['National Average', f'{state} (You)', 'Difference'],
            'Premium': [national_avg, state_premium, state_premium - national_avg]
        })
        
        fig = px.bar(comparison_data, x='Category', y='Premium', 
                    title=f"Your Premium vs National Average",
                    color_discrete_sequence=['lightblue', 'darkblue', 'green' if state_premium < national_avg else 'red'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk factors breakdown
        st.subheader("⚠️ Risk Factors")
        
        for factor, value in result['factors'].items():
            if value > 1.1:
                st.markdown(f"🔴 **{factor}**: +{(value-1)*100:.0f}%")
            elif value < 0.9:
                st.markdown(f"🟢 **{factor}**: {(value-1)*100:.0f}%")
            else:
                st.markdown(f"⚪ **{factor}**: Neutral")
        
        # State info
        st.subheader(f"📍 {state} Statistics")
        state_factor = US_STATE_RISK[state]
        if state_factor > 1.1:
            st.markdown(f"🔴 **High-risk state** (+{(state_factor-1)*100:.0f}%)")
        elif state_factor < 0.9:
            st.markdown(f"🟢 **Low-risk state** ({(state_factor-1)*100:.0f}%)")
        else:
            st.markdown("⚪ **Average-risk state**")
        
        # Money-saving tips
        st.subheader("💡 Money-Saving Tips")
        tips = []
        
        if credit_score < 720:
            tips.append("📈 Improve credit score for better rates")
        if annual_mileage > 15000:
            tips.append("🚗 Consider reducing annual mileage")
        if coverage_level > 1.0:
            tips.append("📋 Review if you need premium coverage")
        if vehicle_age == 0:
            tips.append("🕐 New car premiums decrease after first year")
        
        if not tips:
            tips.append("✅ Your profile is already optimized!")
        
        for tip in tips:
            st.markdown(f"• {tip}")
        
        # Technical details
        with st.expander("🔧 Technical Details"):
            st.write("**Model Features:**")
            st.write("• Age-based risk curves")
            st.write("• State-specific multipliers")
            st.write("• Credit score modeling")
            st.write("• Vehicle make/age factors")
            st.write("• Mileage-based pricing")
            st.write("• Coverage level adjustments")
    
    # Footer
    st.markdown("---")
    st.markdown("*Professional actuarial model • US-specific risk factors • Real insurance principles*")

if __name__ == "__main__":
    main()