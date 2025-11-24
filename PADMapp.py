# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

# Configure page
st.set_page_config(
    page_title="PADM Prediction Model",
    page_icon="🏥",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .risk-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
    }
    .input-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained PADM model"""
    try:
        model_info = joblib.load('PADM_model.pkl')
        return model_info
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_risk(model_info, input_data):
    """Make prediction using the loaded model"""
    try:
        calibrated_model = model_info['model']
        prediction_proba = calibrated_model.predict_proba(input_data)[0, 1]
        return prediction_proba
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def get_risk_level(probability, thresholds):
    """Determine risk level based on probability and thresholds"""
    if probability < thresholds[0]:
        return "Low Risk", "risk-low"
    elif probability <= thresholds[1]:
        return "Medium Risk", "risk-medium"
    else:
        return "High Risk", "risk-high"

def main():
    # Header
    st.markdown('<div class="main-header">🏥 PADM Prediction Model</div>', unsafe_allow_html=True)
    st.markdown("**P**T, **A**PTT, **D**-Dimer, **M**PV based DIC Prediction Model")
    
    # Load model
    model_info = load_model()
    if model_info is None:
        st.error("Failed to load the prediction model. Please ensure PADM_model.pkl exists.")
        return
    
    # Input section
    st.markdown("### Patient Information Input")
    
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            PT = st.number_input(
                "PT (s)",
                min_value=0.0,
                max_value=100.0,
                value=12.0,
                step=0.1,
                help="Prothrombin Time in seconds"
            )
            
            D_Dimer = st.number_input(
                "D-Dimer (mg/L)",
                min_value=0.0,
                max_value=50.0,
                value=0.5,
                step=0.1,
                help="D-Dimer concentration in mg/L"
            )
        
        with col2:
            APTT = st.number_input(
                "APTT (s)",
                min_value=0.0,
                max_value=200.0,
                value=30.0,
                step=0.1,
                help="Activated Partial Thromboplastin Time in seconds"
            )
            
            MPV = st.number_input(
                "MPV (fL)",
                min_value=0.0,
                max_value=20.0,
                value=10.0,
                step=0.1,
                help="Mean Platelet Volume in femtoliters"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction button
    if st.button("Predict DIC Risk", type="primary"):
        # Prepare input data
        input_df = pd.DataFrame({
            'PT': [PT],
            'APTT': [APTT],
            'D-Dimer': [D_Dimer],
            'MPV': [MPV]
        })
        
        # Make prediction
        probability = predict_risk(model_info, input_df)
        
        if probability is not None:
            # Display results
            st.markdown("### Prediction Results")
            
            # Risk level
            risk_level, risk_class = get_risk_level(probability, model_info['risk_thresholds'])
            
            # Probability gauge
            st.metric(
                label="DIC Probability",
                value=f"{probability:.3f}",
                help="Probability of Disseminated Intravascular Coagulation"
            )
            
            # Risk level display
            st.markdown(f'<div class="{risk_class}"><h4>{risk_level}</h4></div>', unsafe_allow_html=True)
            
            # Risk interpretation
            st.markdown("#### Risk Stratification")
            st.info(f"""
            **Risk Categories:**
            - **Low Risk**: Probability < 0.222
            - **Medium Risk**: 0.222 ≤ Probability ≤ 0.640  
            - **High Risk**: Probability > 0.640
            """)
            
            # Threshold visualization
            thresholds = model_info['risk_thresholds']
            st.markdown("#### Probability Scale")
            
            # Create a simple bar visualization
            low_width = thresholds[0] * 100
            medium_width = (thresholds[1] - thresholds[0]) * 100
            high_width = (1 - thresholds[1]) * 100
            
            col_low, col_medium, col_high = st.columns(3)
            
            with col_low:
                st.markdown(f"**Low Risk**\n\n< 0.222", unsafe_allow_html=True)
            with col_medium:
                st.markdown(f"**Medium Risk**\n\n0.222 - 0.640", unsafe_allow_html=True)
            with col_high:
                st.markdown(f"**High Risk**\n\n> 0.640", unsafe_allow_html=True)
    
    # Model information
    with st.expander("About PADM Model"):
        st.markdown("""
        **PADM Prediction Model**
        
        This model predicts the risk of Disseminated Intravascular Coagulation (DIC) 
        using four key laboratory parameters:
        
        - **PT** (Prothrombin Time) - seconds
        - **APTT** (Activated Partial Thromboplastin Time) - seconds  
        - **D-Dimer** - mg/L
        - **MPV** (Mean Platelet Volume) - fL
        
        The model uses isotonic calibration for improved probability estimation.
        Risk stratification is based on validated clinical thresholds.
        """)
        
        st.markdown("**Units:**")
        st.markdown("- PT, APTT: seconds (s)")
        st.markdown("- D-Dimer: milligrams per liter (mg/L)")
        st.markdown("- MPV: femtoliters (fL)")

if __name__ == "__main__":
    main()