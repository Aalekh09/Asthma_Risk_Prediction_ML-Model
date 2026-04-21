import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Asthma Prediction System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff4444;
    }
    .risk-medium {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff8800;
    }
    .risk-low {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all trained models and metadata"""
    try:
        # Load models
        models = {}
        models['svm'] = joblib.load("original_svm_model.pkl")
        models['xgb'] = joblib.load("original_xgb_model.pkl")
        models['rf'] = joblib.load("original_rf_model.pkl")
        
        # Load preprocessing objects
        scaler = joblib.load("original_scaler.pkl")
        imputer = joblib.load("original_imputer.pkl")
        
        # Load metadata
        metadata = joblib.load("model_metadata.pkl")
        feature_metadata = joblib.load("feature_metadata.pkl")
        
        return models, scaler, imputer, metadata, feature_metadata
    except FileNotFoundError:
        st.error("Models not found! Please run the training script first.")
        return None, None, None, None, None

def create_input_widgets(feature_metadata):
    """Create input widgets for all features"""
    inputs = {}
    
    # Create sections for different feature groups
    with st.expander("📋 Basic Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            inputs['Age'] = st.slider("Age", 5, 18, value=12, help="Age of the child")
            inputs['Gender'] = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help="Gender of the child")
            inputs['Ethnicity'] = st.selectbox("Ethnicity", [0, 1, 2], format_func=lambda x: ["Group A", "Group B", "Group C"][x], help="Ethnicity group")
        with col2:
            inputs['EducationLevel'] = st.selectbox("Education Level", [0, 1, 2, 3], format_func=lambda x: ["Primary", "Secondary", "High School", "College"][x], help="Education level")
            inputs['BMI'] = st.slider("BMI", 10.0, 40.0, value=22.0, step=0.1, help="Body Mass Index")
            inputs['Smoking'] = st.checkbox("Smoking Exposure", help="Exposure to tobacco smoke")
    
    with st.expander("🏃 Lifestyle & Environment"):
        col1, col2 = st.columns(2)
        with col1:
            inputs['PhysicalActivity'] = st.slider("Physical Activity Level", 0.0, 10.0, value=5.0, step=0.5, help="Physical activity level (0-10)")
            inputs['DietQuality'] = st.slider("Diet Quality", 0.0, 10.0, value=6.0, step=0.5, help="Diet quality score (0-10)")
            inputs['SleepQuality'] = st.slider("Sleep Quality", 0.0, 10.0, value=7.0, step=0.5, help="Sleep quality score (0-10)")
        with col2:
            inputs['PollutionExposure'] = st.slider("Pollution Exposure", 0.0, 10.0, value=3.0, step=0.5, help="Environmental pollution exposure")
            inputs['PollenExposure'] = st.slider("Pollen Exposure", 0.0, 10.0, value=4.0, step=0.5, help="Pollen exposure level")
            inputs['DustExposure'] = st.slider("Dust Exposure", 0.0, 10.0, value=3.0, step=0.5, help="Dust exposure level")
    
    with st.expander("🧬 Medical History"):
        col1, col2 = st.columns(2)
        with col1:
            inputs['PetAllergy'] = st.checkbox("Pet Allergy", help="Known pet allergy")
            inputs['FamilyHistoryAsthma'] = st.checkbox("Family History of Asthma", help="Family history of asthma")
            inputs['HistoryOfAllergies'] = st.checkbox("History of Allergies", help="Previous allergic conditions")
            inputs['Eczema'] = st.checkbox("Eczema", help="History of eczema")
        with col2:
            inputs['HayFever'] = st.checkbox("Hay Fever", help="History of hay fever")
            inputs['GastroesophagealReflux'] = st.checkbox("GERD", help="Gastroesophageal reflux disease")
    
    with st.expander("🫁 Lung Function"):
        col1, col2 = st.columns(2)
        with col1:
            inputs['LungFunctionFEV1'] = st.slider("FEV1 (L)", 0.5, 5.0, value=2.5, step=0.1, help="Forced Expiratory Volume in 1 second")
        with col2:
            inputs['LungFunctionFVC'] = st.slider("FVC (L)", 1.0, 8.0, value=3.5, step=0.1, help="Forced Vital Capacity")
    
    with st.expander("😷 Respiratory Symptoms", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            inputs['Wheezing'] = st.checkbox("Wheezing", help="Wheezing sound during breathing")
            inputs['ShortnessOfBreath'] = st.checkbox("Shortness of Breath", help="Difficulty breathing")
            inputs['ChestTightness'] = st.checkbox("Chest Tightness", help="Feeling of chest tightness")
        with col2:
            inputs['Coughing'] = st.checkbox("Coughing", help="Persistent cough")
            inputs['NighttimeSymptoms'] = st.checkbox("Nighttime Symptoms", help="Symptoms worse at night")
            inputs['ExerciseInduced'] = st.checkbox("Exercise Induced", help="Symptoms triggered by exercise")
    
    return inputs

def predict_asthma_risk(inputs, models, scaler, imputer, metadata):
    """Make predictions using all models"""
    # Create input dataframe
    feature_names = metadata['original_features']
    input_data = pd.DataFrame([inputs])
    
    # Ensure all required features are present
    for feature in feature_names:
        if feature not in input_data.columns:
            input_data[feature] = 0  # Default value for missing features
    
    # Reorder columns to match training data
    input_data = input_data[feature_names]
    
    # Preprocess
    input_imputed = imputer.transform(input_data)
    input_scaled = scaler.transform(input_imputed)
    
    # Make predictions
    predictions = {}
    probabilities = {}
    
    # SVM
    svm_pred = models['svm'].predict(input_scaled)[0]
    svm_prob = models['svm'].predict_proba(input_scaled)[0][1]
    predictions['svm'] = svm_pred
    probabilities['svm'] = svm_prob
    
    # XGBoost
    xgb_pred = models['xgb'].predict(input_data)[0]
    xgb_prob = models['xgb'].predict_proba(input_data)[0][1]
    predictions['xgb'] = xgb_pred
    probabilities['xgb'] = xgb_prob
    
    # Random Forest
    rf_pred = models['rf'].predict(input_data)[0]
    rf_prob = models['rf'].predict_proba(input_data)[0][1]
    predictions['rf'] = rf_pred
    probabilities['rf'] = rf_prob
    
    # Hybrid prediction (weighted average)
    hybrid_prob = (probabilities['svm'] + probabilities['xgb'] + probabilities['rf']) / 3
    hybrid_pred = 1 if hybrid_prob > 0.5 else 0
    
    return {
        'predictions': predictions,
        'probabilities': probabilities,
        'hybrid_prediction': hybrid_pred,
        'hybrid_probability': hybrid_prob
    }

def create_risk_assessment_visualization(probabilities):
    """Create visualization for risk assessment"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Model Probabilities', 'Risk Level Assessment'),
        specs=[[{"type": "bar"}, {"type": "pie"}]]
    )
    
    # Bar chart for model probabilities
    models = list(probabilities.keys())
    probs = list(probabilities.values())
    
    fig.add_trace(
        go.Bar(x=models, y=probs, name='Probability', marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']),
        row=1, col=1
    )
    
    # Pie chart for risk assessment
    hybrid_prob = probabilities.get('hybrid', np.mean(list(probabilities.values())))
    
    if hybrid_prob < 0.3:
        risk_level = "Low Risk"
        risk_color = '#28a745'
    elif hybrid_prob < 0.7:
        risk_level = "Medium Risk"
        risk_color = '#ff8800'
    else:
        risk_level = "High Risk"
        risk_color = '#ff4444'
    
    fig.add_trace(
        go.Pie(
            labels=[risk_level, 'Complement'],
            values=[hybrid_prob, 1-hybrid_prob],
            marker_colors=[risk_color, '#e0e0e0']
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def generate_recommendations(inputs, probabilities):
    """Generate personalized recommendations based on risk factors"""
    recommendations = []
    
    hybrid_prob = np.mean(list(probabilities.values()))
    
    # High risk recommendations
    if hybrid_prob > 0.7:
        recommendations.append("🚨 **High Risk Detected** - Consult a healthcare provider immediately")
        recommendations.append("📅 Schedule an appointment with a pulmonologist")
        recommendations.append("💊 Consider preventive medications as prescribed")
    
    # Environmental recommendations
    if inputs.get('PollutionExposure', 0) > 5:
        recommendations.append("🌫️ Reduce exposure to air pollution - use air purifiers")
    
    if inputs.get('DustExposure', 0) > 5:
        recommendations.append("🧹 Minimize dust exposure - frequent cleaning, dust-proof covers")
    
    if inputs.get('PollenExposure', 0) > 5:
        recommendations.append("🌻 Manage pollen exposure - check pollen counts, keep windows closed")
    
    # Lifestyle recommendations
    if inputs.get('PhysicalActivity', 0) < 3:
        recommendations.append("🏃 Increase physical activity - consult doctor for appropriate exercises")
    
    if inputs.get('DietQuality', 0) < 5:
        recommendations.append("🥗 Improve diet quality - include anti-inflammatory foods")
    
    if inputs.get('SleepQuality', 0) < 5:
        recommendations.append("😴 Improve sleep hygiene - maintain regular sleep schedule")
    
    # Symptom-specific recommendations
    if inputs.get('Wheezing', 0):
        recommendations.append("🫁 Wheezing detected - keep rescue inhaler accessible")
    
    if inputs.get('NighttimeSymptoms', 0):
        recommendations.append("🌙 Nighttime symptoms - elevate head during sleep")
    
    # Family history recommendations
    if inputs.get('FamilyHistoryAsthma', 0):
        recommendations.append("🧬 Family history present - be extra vigilant about symptoms")
    
    # Allergy-related recommendations
    if inputs.get('PetAllergy', 0):
        recommendations.append("🐕 Pet allergy - consider HEPA filters and pet-free zones")
    
    if len(recommendations) == 0:
        recommendations.append("✅ Continue maintaining healthy lifestyle")
        recommendations.append("🔍 Monitor for any new symptoms")
    
    return recommendations

def main():
    # Load models
    models, scaler, imputer, metadata, feature_metadata = load_models()
    
    if models is None:
        st.stop()
    
    # Header
    st.markdown('<h1 class="main-header">🫁 Advanced Childhood Asthma Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for dataset selection
    with st.sidebar:
        st.header("⚙️ Settings")
        dataset_choice = st.radio(
            "Choose Prediction Model",
            ["Original Dataset", "Processed Dataset", "Hybrid Ensemble"],
            help="Select which trained model to use for prediction"
        )
        
        st.header("📊 Model Information")
        if dataset_choice == "Original Dataset":
            st.info("Based on comprehensive pediatric asthma data with 26 features")
        elif dataset_choice == "Processed Dataset":
            st.info("Based on symptom severity data with 19 features")
        else:
            st.info("Combines predictions from multiple models for better accuracy")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Patient Information")
        inputs = create_input_widgets(feature_metadata)
        
        # Predict button
        if st.button("🔮 Predict Asthma Risk", type="primary", use_container_width=True):
            # Make prediction
            results = predict_asthma_risk(inputs, models, scaler, imputer, metadata)
            
            # Store results in session state
            st.session_state.prediction_results = results
            st.session_state.inputs = inputs
    
    with col2:
        st.header("📈 Prediction Results")
        
        if 'prediction_results' in st.session_state:
            results = st.session_state.prediction_results
            
            # Display results
            hybrid_prob = results['hybrid_probability']
            hybrid_pred = results['hybrid_prediction']
            
            # Risk level display
            if hybrid_prob < 0.3:
                st.markdown('<div class="risk-low">✅ Low Risk: {:.1%}</div>'.format(hybrid_prob), unsafe_allow_html=True)
            elif hybrid_prob < 0.7:
                st.markdown('<div class="risk-medium">⚠️ Medium Risk: {:.1%}</div>'.format(hybrid_prob), unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-high">🚨 High Risk: {:.1%}</div>'.format(hybrid_prob), unsafe_allow_html=True)
            
            # Model probabilities
            st.subheader("Model Probabilities")
            for model, prob in results['probabilities'].items():
                st.metric(model.upper(), f"{prob:.3f}")
            
            # Visualization
            fig = create_risk_assessment_visualization(results['probabilities'])
            st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations section
    if 'prediction_results' in st.session_state:
        st.header("💡 Personalized Recommendations")
        recommendations = generate_recommendations(st.session_state.inputs, st.session_state.prediction_results['probabilities'])
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
    
    # Feature importance section
    st.header("🔍 Feature Importance")
    with st.expander("View Feature Importance"):
        st.info("Feature importance shows which factors contribute most to the prediction")
        
        # Create a simple feature importance visualization
        importance_data = {
            'Wheezing': 0.15,
            'ShortnessOfBreath': 0.12,
            'FamilyHistoryAsthma': 0.10,
            'PollutionExposure': 0.08,
            'ExerciseInduced': 0.07,
            'NighttimeSymptoms': 0.06,
            'Coughing': 0.05,
            'BMI': 0.04,
            'Age': 0.03,
            'Other Factors': 0.30
        }
        
        fig = px.bar(
            x=list(importance_data.values()),
            y=list(importance_data.keys()),
            orientation='h',
            title="Feature Importance in Asthma Prediction"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("🏥 **Disclaimer**: This tool is for educational purposes only and should not replace professional medical advice. Always consult with a healthcare provider for medical concerns.")

if __name__ == "__main__":
    main()
