import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_model():
    try:
        model = joblib.load('Renewable_Energy_Adoption_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model file exists in the current directory.")
        return None

# Set page config
st.set_page_config(
    page_title="Renewable Energy Adoption Predictor",
    page_icon="🌱",
    layout="wide"
)

# Page title and description
st.title("🌱 Renewable Energy Adoption Predictor")
st.markdown("""
This application predicts the likelihood of renewable energy adoption based on various environmental 
and economic factors. Enter the required information below to get a prediction.
""")

# Load the model
model = load_model()

if model is None:
    st.stop()

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    carbon_emissions = st.slider(
        "Carbon Emissions (tons CO2)",
        min_value=50,
        max_value=400,
        value=200,
        help="Annual carbon emissions in tons of CO2"
    )
    
    energy_output = st.slider(
        "Energy Output (MWh)",
        min_value=100,
        max_value=1000,
        value=500,
        help="Annual energy output in Megawatt-hours"
    )

with col2:
    renewability_index = st.slider(
        "Renewability Index",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Index measuring the proportion of renewable energy (0-1)"
    )
    
    cost_efficiency = st.slider(
        "Cost Efficiency Ratio",
        min_value=0.5,
        max_value=5.0,
        value=2.7,
        step=0.1,
        help="Ratio of energy output to implementation costs"
    )

# Create a DataFrame with the input values
input_data = pd.DataFrame({
    'carbon_emissions': [carbon_emissions],
    'energy_output': [energy_output],
    'renewability_index': [renewability_index],
    'cost_efficiency': [cost_efficiency]
})

# Add a prediction button
if st.button("Predict Adoption", type="primary"):
    # Make prediction
    with st.spinner("Analyzing factors..."):
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        # Display results
        st.markdown("### Prediction Results")
        
        # Create columns for results
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if prediction[0] == 1:
                st.success("✅ High likelihood of renewable energy adoption!")
            else:
                st.error("❌ Low likelihood of renewable energy adoption")
        
        with result_col2:
            st.metric(
                label="Probability of Adoption",
                value=f"{probability[0][1]:.2%}"
            )
        
        # Feature importance visualization
        st.markdown("### Feature Analysis")
        
        # Create a bar chart of input values compared to typical ranges
        st.markdown("#### Current Input Values vs. Typical Ranges")
        
        # Sample ranges (based on the dataset description)
        ranges = pd.DataFrame({
            'Feature': ['Carbon Emissions', 'Energy Output', 'Renewability Index', 'Cost Efficiency'],
            'Current': [carbon_emissions, energy_output, renewability_index, cost_efficiency],
            'Min': [50, 100, 0, 0.5],
            'Max': [400, 1000, 1, 5]
        })
        
        # Calculate percentage of max for each feature
        ranges['Percentage'] = (ranges['Current'] - ranges['Min']) / (ranges['Max'] - ranges['Min']) * 100
        
        # Create a horizontal bar chart using matplotlib
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(ranges['Feature'], ranges['Percentage'], color='skyblue')
        ax.set_xlabel('Percentage of Maximum Range')
        ax.set_title('Current Values (as % of range)')
        
        # Add percentage labels on the bars
        for i, v in enumerate(ranges['Percentage']):
            ax.text(v + 1, i, f'{v:.1f}%', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Recommendations
        st.markdown("### Recommendations")
        
        recommendations = []
        
        if carbon_emissions > 300:
            recommendations.append("Consider implementing carbon reduction strategies.")
        if renewability_index < 0.4:
            recommendations.append("Look into increasing the proportion of renewable energy sources.")
        if cost_efficiency < 2.0:
            recommendations.append("Explore ways to improve cost efficiency ratio.")
        if energy_output < 300:
            recommendations.append("Consider methods to increase energy output capacity.")
            
        if recommendations:
            for rec in recommendations:
                st.warning(rec)
        else:
            st.success("Your current parameters are within optimal ranges!")
