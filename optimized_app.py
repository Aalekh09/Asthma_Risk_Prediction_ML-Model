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
    page_title="Optimized Asthma Prediction System",
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
    .partition-info {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #0066cc;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_optimized_models():
    """Load optimized models and metadata for both datasets"""
    try:
        models = {}
        scalers = {}
        imputers = {}
        metadata = {}
        feature_metadata = {}
        model_type = "optimized"
        
        # Try to load optimized models
        try:
            # Load optimized feature metadata first
            feature_metadata = joblib.load("optimized_feature_metadata.pkl")
            metadata = joblib.load("optimized_model_metadata.pkl")
            
            # Load ensemble models (most comprehensive)
            try:
                models['original_ensemble'] = {
                    'svm': joblib.load("optimized_original_ensemble_svm_model.pkl"),
                    'xgb': joblib.load("optimized_original_ensemble_xgb_model.pkl"),
                    'rf': joblib.load("optimized_original_ensemble_rf_model.pkl")
                }
                scalers['original_ensemble'] = joblib.load("optimized_original_ensemble_scaler.pkl")
                imputers['original_ensemble'] = joblib.load("optimized_original_ensemble_imputer.pkl")
            except FileNotFoundError:
                st.warning("Original ensemble models not found")
            
            try:
                models['processed_ensemble'] = {
                    'svm': joblib.load("optimized_processed_ensemble_svm_model.pkl"),
                    'xgb': joblib.load("optimized_processed_ensemble_xgb_model.pkl"),
                    'rf': joblib.load("optimized_processed_ensemble_rf_model.pkl")
                }
            except FileNotFoundError:
                st.warning("Processed ensemble models not found")
            
            # Try to load age-specific models if ensemble not available
            if not models.get('original_ensemble') and not models.get('processed_ensemble'):
                # Try to load any available optimized models
                age_groups = ['5-9', '10-14', '14-18']
                
                for age_group in age_groups:
                    try:
                        models[f'original_{age_group}'] = {
                            'svm': joblib.load(f"optimized_original_{age_group}_svm_model.pkl"),
                            'xgb': joblib.load(f"optimized_original_{age_group}_xgb_model.pkl"),
                            'rf': joblib.load(f"optimized_original_{age_group}_rf_model.pkl")
                        }
                        scalers[f'original_{age_group}'] = joblib.load(f"optimized_original_{age_group}_scaler.pkl")
                        imputers[f'original_{age_group}'] = joblib.load(f"optimized_original_{age_group}_imputer.pkl")
                    except FileNotFoundError:
                        pass
                    
                    try:
                        models[f'processed_{age_group}'] = {
                            'svm': joblib.load(f"optimized_processed_{age_group}_svm_model.pkl"),
                            'xgb': joblib.load(f"optimized_processed_{age_group}_xgb_model.pkl"),
                            'rf': joblib.load(f"optimized_processed_{age_group}_rf_model.pkl")
                        }
                    except FileNotFoundError:
                        pass
            
        except FileNotFoundError:
            st.error("Optimized models not found! Please run 'python optimized_train_model.py' first.")
            return None, None, None, None, None, None
        
        return models, scalers, imputers, metadata, feature_metadata, model_type
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None, None

def create_default_feature_metadata():
    """Create default feature metadata if not available"""
    return {
        'processed_features': {
            'Tiredness': {'type': 'binary', 'options': [0, 1], 'description': 'Tiredness symptom'},
            'Dry-Cough': {'type': 'binary', 'options': [0, 1], 'description': 'Dry cough symptom'},
            'Difficulty-in-Breathing': {'type': 'binary', 'options': [0, 1], 'description': 'Difficulty breathing'},
            'Sore-Throat': {'type': 'binary', 'options': [0, 1], 'description': 'Sore throat symptom'},
            'None_Sympton': {'type': 'binary', 'options': [0, 1], 'description': 'No symptoms'},
            'Pains': {'type': 'binary', 'options': [0, 1], 'description': 'Body pains'},
            'Nasal-Congestion': {'type': 'binary', 'options': [0, 1], 'description': 'Nasal congestion'},
            'Runny-Nose': {'type': 'binary', 'options': [0, 1], 'description': 'Runny nose'},
            'None_Experiencing': {'type': 'binary', 'options': [0, 1], 'description': 'Not experiencing symptoms'},
            'Age_0-9': {'type': 'binary', 'options': [0, 1], 'description': 'Age group 0-9'},
            'Age_10-19': {'type': 'binary', 'options': [0, 1], 'description': 'Age group 10-19'},
            'Age_20-24': {'type': 'binary', 'options': [0, 1], 'description': 'Age group 20-24'},
            'Age_25-59': {'type': 'binary', 'options': [0, 1], 'description': 'Age group 25-59'},
            'Age_60+': {'type': 'binary', 'options': [0, 1], 'description': 'Age group 60+'},
            'Gender_Female': {'type': 'binary', 'options': [0, 1], 'description': 'Female gender'},
            'Gender_Male': {'type': 'binary', 'options': [0, 1], 'description': 'Male gender'},
            'Severity_Mild': {'type': 'binary', 'options': [0, 1], 'description': 'Mild severity'},
            'Severity_Moderate': {'type': 'binary', 'options': [0, 1], 'description': 'Moderate severity'},
            'Severity_None': {'type': 'binary', 'options': [0, 1], 'description': 'No severity'}
        }
    }

def create_original_input_widgets():
    """Create input widgets for original dataset prediction"""
    inputs = {}
    
    # Basic Information
    st.subheader("📋 Basic Information")
    col1, col2 = st.columns(2)
    
    with col1:
        inputs['Age'] = st.slider("Age", 5, 18, value=12, help="Age of the child")
        inputs['Gender'] = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help="Gender of the child")
        inputs['Ethnicity'] = st.selectbox("Ethnicity", [0, 1, 2], format_func=lambda x: ["Group A", "Group B", "Group C"][x], help="Ethnicity group")
        
    with col2:
        inputs['EducationLevel'] = st.selectbox("Education Level", [0, 1, 2, 3], format_func=lambda x: ["Primary", "Secondary", "High School", "College"][x], help="Education level")
        inputs['BMI'] = st.slider("BMI", 10.0, 40.0, value=22.0, step=0.1, help="Body Mass Index")
        inputs['Smoking'] = st.checkbox("Smoking Exposure", help="Exposure to tobacco smoke")
    
    # Lifestyle & Environment
    st.subheader("🏃 Lifestyle & Environment")
    col1, col2 = st.columns(2)
    
    with col1:
        inputs['PhysicalActivity'] = st.slider("Physical Activity Level", 0.0, 10.0, value=5.0, step=0.5, help="Physical activity level (0-10)")
        inputs['DietQuality'] = st.slider("Diet Quality", 0.0, 10.0, value=6.0, step=0.5, help="Diet quality score (0-10)")
        inputs['SleepQuality'] = st.slider("Sleep Quality", 0.0, 10.0, value=7.0, step=0.5, help="Sleep quality score (0-10)")
        
    with col2:
        inputs['PollutionExposure'] = st.slider("Pollution Exposure", 0.0, 10.0, value=3.0, step=0.5, help="Environmental pollution exposure")
        inputs['PollenExposure'] = st.slider("Pollen Exposure", 0.0, 10.0, value=4.0, step=0.5, help="Pollen exposure level")
        inputs['DustExposure'] = st.slider("Dust Exposure", 0.0, 10.0, value=3.0, step=0.5, help="Dust exposure level")
    
    # Medical History
    st.subheader("🧬 Medical History")
    col1, col2 = st.columns(2)
    
    with col1:
        inputs['PetAllergy'] = st.checkbox("Pet Allergy", help="Known pet allergy")
        inputs['FamilyHistoryAsthma'] = st.checkbox("Family History of Asthma", help="Family history of asthma")
        inputs['HistoryOfAllergies'] = st.checkbox("History of Allergies", help="Previous allergic conditions")
        inputs['Eczema'] = st.checkbox("Eczema", help="History of eczema")
        
    with col2:
        inputs['HayFever'] = st.checkbox("Hay Fever", help="History of hay fever")
        inputs['GastroesophagealReflux'] = st.checkbox("GERD", help="Gastroesophageal reflux disease")
    
    # Lung Function
    st.subheader("🫁 Lung Function")
    col1, col2 = st.columns(2)
    
    with col1:
        inputs['LungFunctionFEV1'] = st.slider("FEV1 (L)", 0.5, 5.0, value=2.5, step=0.1, help="Forced Expiratory Volume in 1 second")
        
    with col2:
        inputs['LungFunctionFVC'] = st.slider("FVC (L)", 1.0, 8.0, value=3.5, step=0.1, help="Forced Vital Capacity")
    
    # Respiratory Symptoms
    st.subheader("😷 Respiratory Symptoms")
    col1, col2 = st.columns(2)
    
    with col1:
        inputs['Wheezing'] = st.checkbox("Wheezing", help="Wheezing sound during breathing")
        inputs['ShortnessOfBreath'] = st.checkbox("Shortness of Breath", help="Difficulty breathing")
        inputs['ChestTightness'] = st.checkbox("Chest Tightness", help="Feeling of chest tightness")
        
    with col2:
        inputs['Coughing'] = st.checkbox("Coughing", help="Persistent cough")
        inputs['NighttimeSymptoms'] = st.checkbox("Nighttime Symptoms", help="Symptoms worse at night")
        inputs['ExerciseInduced'] = st.checkbox("Exercise Induced", help="Symptoms triggered by exercise")
    
    # Determine age group for recommendations
    age = inputs['Age']
    if 5 <= age < 10:
        age_group = "5-9 years"
    elif 10 <= age < 15:
        age_group = "10-14 years"
    else:
        age_group = "14-18 years"
    
    return inputs, age_group

def create_universal_input_widgets():
    """Create universal input widgets that work for both datasets"""
    inputs = {}
    
    # Age group selection
    st.subheader("👤 Age Group")
    age_group = st.selectbox(
        "Select Age Group",
        ["5-9 years", "10-14 years", "14-18 years"],
        help="Select the age group for prediction"
    )
    inputs['age_group'] = age_group
    
    # Gender selection
    st.subheader("⚧ Gender")
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
    inputs['Gender'] = 0 if gender == "Female" else 1
    inputs['Gender_Male'] = 1 if gender == "Male" else 0
    
    # Basic health information
    st.subheader("🏥 Basic Health Information")
    col1, col2 = st.columns(2)
    
    with col1:
        inputs['BMI'] = st.slider("BMI", 10.0, 40.0, value=22.0, step=0.1, help="Body Mass Index")
        inputs['Smoking'] = st.checkbox("Smoking Exposure", help="Exposure to tobacco smoke")
        inputs['FamilyHistoryAsthma'] = st.checkbox("Family History of Asthma", help="Family history of asthma")
        
    with col2:
        inputs['PhysicalActivity'] = st.slider("Physical Activity Level", 0.0, 10.0, value=5.0, step=0.5, help="Physical activity level (0-10)")
        inputs['DietQuality'] = st.slider("Diet Quality", 0.0, 10.0, value=6.0, step=0.5, help="Diet quality score (0-10)")
        inputs['SleepQuality'] = st.slider("Sleep Quality", 0.0, 10.0, value=7.0, step=0.5, help="Sleep quality score (0-10)")
    
    # Environmental factors
    st.subheader("🌍 Environmental Factors")
    col1, col2 = st.columns(2)
    
    with col1:
        inputs['PollutionExposure'] = st.slider("Pollution Exposure", 0.0, 10.0, value=3.0, step=0.5, help="Environmental pollution exposure")
        inputs['PollenExposure'] = st.slider("Pollen Exposure", 0.0, 10.0, value=4.0, step=0.5, help="Pollen exposure level")
        
    with col2:
        inputs['DustExposure'] = st.slider("Dust Exposure", 0.0, 10.0, value=3.0, step=0.5, help="Dust exposure level")
        inputs['PetAllergy'] = st.checkbox("Pet Allergy", help="Known pet allergy")
    
    # Respiratory symptoms
    st.subheader("😷 Respiratory Symptoms")
    col1, col2 = st.columns(2)
    
    with col1:
        inputs['Wheezing'] = st.checkbox("Wheezing", help="Wheezing sound during breathing")
        inputs['ShortnessOfBreath'] = st.checkbox("Shortness of Breath", help="Difficulty breathing")
        inputs['ChestTightness'] = st.checkbox("Chest Tightness", help="Feeling of chest tightness")
        
    with col2:
        inputs['Coughing'] = st.checkbox("Coughing", help="Persistent cough")
        inputs['NighttimeSymptoms'] = st.checkbox("Nighttime Symptoms", help="Symptoms worse at night")
        inputs['ExerciseInduced'] = st.checkbox("Exercise Induced", help="Symptoms triggered by exercise")
    
    # General symptoms (for processed dataset)
    st.subheader("🤒 General Symptoms")
    col1, col2 = st.columns(2)
    
    with col1:
        inputs['Tiredness'] = st.checkbox("Tiredness", help="Feeling of fatigue")
        inputs['Dry-Cough'] = st.checkbox("Dry Cough", help="Persistent dry cough")
        inputs['Difficulty-in-Breathing'] = st.checkbox("Difficulty Breathing", help="Shortness of breath")
        inputs['Sore-Throat'] = st.checkbox("Sore Throat", help="Throat irritation")
        inputs['Pains'] = st.checkbox("Body Pains", help="General body aches")
        
    with col2:
        inputs['Nasal-Congestion'] = st.checkbox("Nasal Congestion", help="Stuffy nose")
        inputs['Runny-Nose'] = st.checkbox("Runny Nose", help="Nasal discharge")
        inputs['None_Sympton'] = st.checkbox("No Symptoms", help="No respiratory symptoms")
        inputs['None_Experiencing'] = st.checkbox("Not Experiencing Issues", help="Generally feeling well")
    
    return inputs, age_group

def create_dual_dataset_visualization(probabilities, age_group, datasets_used):
    """Create visualization for dual dataset predictions"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Original Dataset Models', 'Processed Dataset Models', 'Combined Risk Assessment', 'Dataset Comparison'),
        specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "pie"}, {"type": "bar"}]]
    )
    
    # Separate probabilities by dataset
    original_probs = {k.replace('original_', ''): v for k, v in probabilities.items() if 'original' in k}
    processed_probs = {k.replace('processed_', ''): v for k, v in probabilities.items() if 'processed' in k}
    
    # Original dataset bar chart
    if original_probs:
        fig.add_trace(
            go.Bar(x=list(original_probs.keys()), y=list(original_probs.values()), 
                   name='Original', marker_color='#1f77b4'),
            row=1, col=1
        )
    
    # Processed dataset bar chart
    if processed_probs:
        fig.add_trace(
            go.Bar(x=list(processed_probs.keys()), y=list(processed_probs.values()), 
                   name='Processed', marker_color='#ff7f0e'),
            row=1, col=2
        )
    
    # Combined risk pie chart
    all_probs = list(probabilities.values())
    if all_probs:
        hybrid_prob = np.mean(all_probs)
        
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
            row=2, col=1
        )
    
    # Dataset comparison
    if original_probs and processed_probs:
        orig_avg = np.mean(list(original_probs.values()))
        proc_avg = np.mean(list(processed_probs.values()))
        
        fig.add_trace(
            go.Bar(x=['Original Dataset', 'Processed Dataset'], y=[orig_avg, proc_avg],
                   marker_color=['#1f77b4', '#ff7f0e']),
            row=2, col=2
        )
    
    fig.update_layout(height=600, showlegend=False, title_text=f"Dual Dataset Analysis - {age_group}")
    return fig

def create_optimized_input_widgets():
    """Create optimized input widgets for symptom-based prediction"""
    inputs = {}
    
    # Age group selection
    st.subheader("👤 Age Group")
    age_group = st.selectbox(
        "Select Age Group",
        ["5-9 years", "10-14 years", "14-18 years"],
        help="Select the age group for prediction"
    )
    
    # Map age group to features
    if age_group == "5-9 years":
        inputs['Age_0-9'] = 1
        inputs['Age_10-19'] = 0
    elif age_group == "10-14 years":
        inputs['Age_0-9'] = 0
        inputs['Age_10-19'] = 1
    else:  # 14-18 years
        inputs['Age_0-9'] = 0
        inputs['Age_10-19'] = 1
    
    # Set other age groups to 0
    inputs['Age_20-24'] = 0
    inputs['Age_25-59'] = 0
    inputs['Age_60+'] = 0
    
    # Gender selection
    st.subheader("⚧ Gender")
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
    inputs['Gender_Female'] = 1 if gender == "Female" else 0
    inputs['Gender_Male'] = 1 if gender == "Male" else 0
    
    # Symptoms section
    st.subheader("😷 Symptoms")
    
    col1, col2 = st.columns(2)
    
    with col1:
        inputs['Tiredness'] = st.checkbox("Tiredness", help="Feeling of fatigue")
        inputs['Dry-Cough'] = st.checkbox("Dry Cough", help="Persistent dry cough")
        inputs['Difficulty-in-Breathing'] = st.checkbox("Difficulty Breathing", help="Shortness of breath")
        inputs['Sore-Throat'] = st.checkbox("Sore Throat", help="Throat irritation")
        inputs['Pains'] = st.checkbox("Body Pains", help="General body aches")
        
    with col2:
        inputs['Nasal-Congestion'] = st.checkbox("Nasal Congestion", help="Stuffy nose")
        inputs['Runny-Nose'] = st.checkbox("Runny Nose", help="Nasal discharge")
        inputs['None_Sympton'] = st.checkbox("No Symptoms", help="No respiratory symptoms")
        inputs['None_Experiencing'] = st.checkbox("Not Experiencing Issues", help="Generally feeling well")
    
    # Ensure at least some symptom selection logic
    if inputs['None_Sympton'] == 1:
        # If no symptoms, set other symptoms to 0
        for symptom in ['Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat', 'Pains', 'Nasal-Congestion', 'Runny-Nose']:
            inputs[symptom] = 0
    
    return inputs, age_group

def predict_asthma_risk_optimized(inputs, models, dataset_choice, scaler=None, imputer=None):
    """Make predictions using optimized models for both datasets"""
    # Create input dataframe
    input_data = pd.DataFrame([inputs])
    
    try:
        # Preprocess based on dataset type
        if dataset_choice == "Original Dataset":
            # For original dataset, apply preprocessing if available
            if scaler is not None and imputer is not None:
                # Ensure all required features are present
                required_features = ['Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking',
                                    'PhysicalActivity', 'DietQuality', 'SleepQuality', 'PollutionExposure',
                                    'PollenExposure', 'DustExposure', 'PetAllergy', 'FamilyHistoryAsthma',
                                    'HistoryOfAllergies', 'Eczema', 'HayFever', 'GastroesophagealReflux',
                                    'LungFunctionFEV1', 'LungFunctionFVC', 'Wheezing', 'ShortnessOfBreath',
                                    'ChestTightness', 'Coughing', 'NighttimeSymptoms', 'ExerciseInduced']
                
                for feature in required_features:
                    if feature not in input_data.columns:
                        input_data[feature] = 0
                
                # Reorder columns
                input_data = input_data[required_features]
                
                # Apply preprocessing
                input_imputed = imputer.transform(input_data)
                input_scaled = scaler.transform(input_imputed)
                final_input = input_scaled
            else:
                # No preprocessing available, use raw features
                final_input = input_data.values
        else:
            # Processed dataset - use features as-is
            final_input = input_data.values
        
        # Make predictions
        predictions = {}
        probabilities = {}
        
        # SVM
        if 'svm' in models:
            svm_pred = models['svm'].predict(final_input)[0]
            svm_prob = models['svm'].predict_proba(final_input)[0][1]
            predictions['svm'] = svm_pred
            probabilities['svm'] = svm_prob
        
        # XGBoost
        if 'xgb' in models:
            xgb_pred = models['xgb'].predict(final_input)[0]
            xgb_prob = models['xgb'].predict_proba(final_input)[0][1]
            predictions['xgb'] = xgb_pred
            probabilities['xgb'] = xgb_prob
        
        # Random Forest
        if 'rf' in models:
            rf_pred = models['rf'].predict(final_input)[0]
            rf_prob = models['rf'].predict_proba(final_input)[0][1]
            predictions['rf'] = rf_pred
            probabilities['rf'] = rf_prob
        
        # Hybrid prediction (weighted average)
        if probabilities:
            hybrid_prob = np.mean(list(probabilities.values()))
            hybrid_pred = 1 if hybrid_prob > 0.5 else 0
        else:
            hybrid_prob = 0.0
            hybrid_pred = 0
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'hybrid_prediction': hybrid_pred,
            'hybrid_probability': hybrid_prob,
            'success': True
        }
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return {
            'predictions': {},
            'probabilities': {},
            'hybrid_prediction': 0,
            'hybrid_probability': 0.0,
            'success': False,
            'error': str(e)
        }

def create_optimized_visualization(probabilities, age_group):
    """Create optimized visualization for risk assessment"""
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
    hybrid_prob = np.mean(list(probabilities.values())) if probabilities else 0
    
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

def generate_age_specific_recommendations(inputs, probabilities, age_group):
    """Generate age-specific recommendations"""
    recommendations = []
    
    hybrid_prob = np.mean(list(probabilities.values())) if probabilities else 0
    
    # Age-specific recommendations
    if age_group == "5-9 years":
        recommendations.append("👶 **Age 5-9**: Monitor for symptoms during play activities")
        recommendations.append("🎯 Ensure age-appropriate asthma action plan at school")
    elif age_group == "10-14 years":
        recommendations.append("👦 **Age 10-14**: Encourage self-monitoring and responsibility")
        recommendations.append("🏫 Coordinate with school nurses for medication management")
    else:  # 14-18 years
        recommendations.append("👨 **Age 14-18**: Focus on sports and activity management")
        recommendations.append("💬 Discuss medication independence and adherence")
    
    # Risk-based recommendations
    if hybrid_prob > 0.7:
        recommendations.append("🚨 **High Risk Detected** - Consult healthcare provider immediately")
        recommendations.append("📅 Schedule pulmonary function testing")
    elif hybrid_prob > 0.4:
        recommendations.append("⚠️ **Moderate Risk** - Regular monitoring recommended")
        recommendations.append("🌬️ Consider preventive measures during high pollen days")
    else:
        recommendations.append("✅ **Low Risk** - Continue healthy lifestyle")
    
    # Symptom-specific recommendations
    if inputs.get('Difficulty-in-Breathing', 0):
        recommendations.append("🫁 Breathing difficulty - Keep rescue inhaler accessible")
    
    if inputs.get('Dry-Cough', 0):
        recommendations.append("🤧 Dry cough - Monitor for triggers like dust or pollen")
    
    if inputs.get('Tiredness', 0):
        recommendations.append("😴 Fatigue - Ensure adequate sleep and rest")
    
    return recommendations

def predict_with_both_datasets(inputs, models, scalers, imputers, age_group):
    """Combine predictions from both datasets automatically"""
    all_results = {}
    
    # Determine which models to use based on age group
    age_group_key = age_group.replace(" years", "").replace("-", "-")
    
    # Try to get age-specific models first, fallback to ensemble
    original_model_key = None
    processed_model_key = None
    
    # Check for age-specific models
    if f'original_{age_group_key}' in models:
        original_model_key = f'original_{age_group_key}'
    elif 'original_ensemble' in models:
        original_model_key = 'original_ensemble'
    
    if f'processed_{age_group_key}' in models:
        processed_model_key = f'processed_{age_group_key}'
    elif 'processed_ensemble' in models:
        processed_model_key = 'processed_ensemble'
    
    # Get predictions from original dataset if available
    if original_model_key:
        original_inputs = create_original_inputs_from_universal(inputs)
        original_result = predict_asthma_risk_optimized(
            original_inputs,
            models[original_model_key],
            "Original Dataset",
            scalers.get(original_model_key),
            imputers.get(original_model_key)
        )
        if original_result['success']:
            all_results['original'] = original_result
    
    # Get predictions from processed dataset if available
    if processed_model_key:
        processed_inputs = create_processed_inputs_from_universal(inputs)
        processed_result = predict_asthma_risk_optimized(
            processed_inputs,
            models[processed_model_key],
            "Processed Dataset",
            scalers.get(processed_model_key),
            imputers.get(processed_model_key)
        )
        if processed_result['success']:
            all_results['processed'] = processed_result
    
    # Combine predictions from both datasets
    if all_results:
        combined_result = combine_dataset_predictions(all_results)
        return combined_result, list(all_results.keys())
    else:
        return {
            'predictions': {},
            'probabilities': {},
            'hybrid_prediction': 0,
            'hybrid_probability': 0.0,
            'success': False,
            'error': 'No models available for prediction'
        }, []

def create_original_inputs_from_universal(universal_inputs):
    """Convert universal inputs to original dataset format"""
    # For original dataset, we need comprehensive features
    original_inputs = {}
    
    # Map basic info
    if 'Age' in universal_inputs:
        original_inputs['Age'] = universal_inputs['Age']
    else:
        # Derive age from age group
        age_group = universal_inputs.get('age_group', '10-14 years')
        if '5-9' in age_group:
            original_inputs['Age'] = 7
        elif '10-14' in age_group:
            original_inputs['Age'] = 12
        else:
            original_inputs['Age'] = 16
    
    # Gender mapping
    if 'Gender' in universal_inputs:
        original_inputs['Gender'] = universal_inputs['Gender']
    elif 'Gender_Male' in universal_inputs:
        original_inputs['Gender'] = universal_inputs['Gender_Male']
    else:
        original_inputs['Gender'] = 0
    
    # Default values for other features (can be customized)
    original_inputs.update({
        'Ethnicity': 1,
        'EducationLevel': 1,
        'BMI': 22.0,
        'Smoking': 0,
        'PhysicalActivity': 5.0,
        'DietQuality': 6.0,
        'SleepQuality': 7.0,
        'PollutionExposure': 3.0,
        'PollenExposure': 4.0,
        'DustExposure': 3.0,
        'PetAllergy': 0,
        'FamilyHistoryAsthma': 0,
        'HistoryOfAllergies': 0,
        'Eczema': 0,
        'HayFever': 0,
        'GastroesophagealReflux': 0,
        'LungFunctionFEV1': 2.5,
        'LungFunctionFVC': 3.5,
        'Wheezing': universal_inputs.get('Wheezing', 0),
        'ShortnessOfBreath': universal_inputs.get('ShortnessOfBreath', 0),
        'ChestTightness': universal_inputs.get('ChestTightness', 0),
        'Coughing': universal_inputs.get('Coughing', 0),
        'NighttimeSymptoms': universal_inputs.get('NighttimeSymptoms', 0),
        'ExerciseInduced': universal_inputs.get('ExerciseInduced', 0)
    })
    
    return original_inputs

def create_processed_inputs_from_universal(universal_inputs):
    """Convert universal inputs to processed dataset format"""
    processed_inputs = {}
    
    # Age group mapping
    age_group = universal_inputs.get('age_group', '10-14 years')
    if '5-9' in age_group:
        processed_inputs['Age_0-9'] = 1
        processed_inputs['Age_10-19'] = 0
    elif '10-14' in age_group or '14-18' in age_group:
        processed_inputs['Age_0-9'] = 0
        processed_inputs['Age_10-19'] = 1
    else:
        processed_inputs['Age_0-9'] = 0
        processed_inputs['Age_10-19'] = 1
    
    # Set other age groups to 0
    processed_inputs.update({
        'Age_20-24': 0,
        'Age_25-59': 0,
        'Age_60+': 0
    })
    
    # Gender mapping
    if 'Gender_Male' in universal_inputs:
        processed_inputs['Gender_Female'] = 1 - universal_inputs['Gender_Male']
        processed_inputs['Gender_Male'] = universal_inputs['Gender_Male']
    else:
        processed_inputs['Gender_Female'] = 1
        processed_inputs['Gender_Male'] = 0
    
    # Symptoms mapping
    symptom_mapping = {
        'Tiredness': 'Tiredness',
        'Dry-Cough': 'Dry-Cough',
        'Difficulty-in-Breathing': 'Difficulty-in-Breathing',
        'Sore-Throat': 'Sore-Throat',
        'Pains': 'Pains',
        'Nasal-Congestion': 'Nasal-Congestion',
        'Runny-Nose': 'Runny-Nose',
        'None_Sympton': 'None_Sympton',
        'None_Experiencing': 'None_Experiencing'
    }
    
    for symptom, processed_key in symptom_mapping.items():
        processed_inputs[processed_key] = universal_inputs.get(symptom, 0)
    
    return processed_inputs

def combine_dataset_predictions(dataset_results):
    """Combine predictions from multiple datasets"""
    combined_probabilities = {}
    combined_predictions = {}
    
    # Collect all probabilities from all datasets
    all_probs = []
    for dataset_name, result in dataset_results.items():
        for model_name, prob in result['probabilities'].items():
            combined_key = f"{dataset_name}_{model_name}"
            combined_probabilities[combined_key] = prob
            combined_predictions[combined_key] = result['predictions'][model_name]
            all_probs.append(prob)
    
    # Calculate combined hybrid probability
    if all_probs:
        hybrid_prob = np.mean(all_probs)
        hybrid_pred = 1 if hybrid_prob > 0.5 else 0
    else:
        hybrid_prob = 0.0
        hybrid_pred = 0
    
    return {
        'predictions': combined_predictions,
        'probabilities': combined_probabilities,
        'hybrid_prediction': hybrid_pred,
        'hybrid_probability': hybrid_prob,
        'success': True,
        'datasets_used': list(dataset_results.keys())
    }

def main():
    # Load models
    models, scalers, imputers, metadata, feature_metadata, model_type = load_optimized_models()
    
    if models is None:
        st.stop()
    
    # Header
    st.markdown('<h1 class="main-header">🫁 Dual-Dataset Childhood Asthma Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Model information
    st.markdown('<div class="partition-info">✅ Automatically combining predictions from BOTH datasets (Original + Processed) for comprehensive analysis</div>', unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Patient Information")
        
        # Create universal input widgets that work for both datasets
        inputs, age_group = create_universal_input_widgets()
        
        # Predict button
        if st.button("🔮 Predict Asthma Risk (Both Datasets)", type="primary", use_container_width=True):
            with st.spinner("Analyzing with both datasets..."):
                # Make prediction using both datasets
                results, datasets_used = predict_with_both_datasets(
                    inputs, 
                    models, 
                    scalers, 
                    imputers, 
                    age_group
                )
                
                if results['success']:
                    # Store results in session state
                    st.session_state.prediction_results = results
                    st.session_state.inputs = inputs
                    st.session_state.age_group = age_group
                    st.session_state.datasets_used = datasets_used
                else:
                    st.error("Prediction failed. Please check your inputs.")
    
    with col2:
        st.header("📈 Combined Prediction Results")
        
        if 'prediction_results' in st.session_state:
            results = st.session_state.prediction_results
            
            if results['success']:
                # Display datasets used
                datasets_info = " + ".join([d.title() for d in results['datasets_used']])
                st.info(f"🔄 Combined Analysis: {datasets_info}")
                
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
                
                # Model probabilities from both datasets
                st.subheader("Model Probabilities (Both Datasets)")
                
                # Group by dataset
                original_probs = {k: v for k, v in results['probabilities'].items() if 'original' in k}
                processed_probs = {k: v for k, v in results['probabilities'].items() if 'processed' in k}
                
                if original_probs:
                    st.write("**Original Dataset Models:**")
                    for model, prob in original_probs.items():
                        model_name = model.replace('original_', '').upper()
                        st.metric(model_name, f"{prob:.3f}")
                
                if processed_probs:
                    st.write("**Processed Dataset Models:**")
                    for model, prob in processed_probs.items():
                        model_name = model.replace('processed_', '').upper()
                        st.metric(model_name, f"{prob:.3f}")
                
                # Visualization
                fig = create_dual_dataset_visualization(results['probabilities'], st.session_state.age_group, results['datasets_used'])
                st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations section
    if 'prediction_results' in st.session_state and st.session_state.prediction_results['success']:
        st.header("💡 Personalized Recommendations")
        recommendations = generate_age_specific_recommendations(
            st.session_state.inputs, 
            st.session_state.prediction_results['probabilities'],
            st.session_state.age_group
        )
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
    
    # Information section
    with st.expander("ℹ️ About This Dual-Dataset System"):
        st.info("""
        **Dual-Dataset Asthma Prediction System**
        
        This advanced system automatically combines predictions from BOTH datasets to provide comprehensive asthma risk assessment.
        
        **🔄 How It Works:**
        - **Original Dataset**: Comprehensive medical features (26 features)
        - **Processed Dataset**: Symptom-based features (19 features)
        - **Automatic Combination**: No manual model selection required
        - **Age-Specific Models**: 5-9, 10-14, 14-18 years for each dataset
        
        **🎯 Key Benefits:**
        - **Dual Analysis**: Uses both datasets simultaneously
        - **Automatic Selection**: No need to choose models manually
        - **Comprehensive Results**: Combines strengths of both approaches
        - **Age-Appropriate**: Tailored predictions for each age group
        
        **📊 Model Performance:**
        - Original Dataset: 93-97% accuracy (high-quality features)
        - Processed Dataset: 47-53% accuracy (symptom-focused)
        - Combined Results: Balanced assessment from multiple perspectives
        - Training optimized for systems with limited resources
        """)
    
    # Model comparison section
    if 'prediction_results' in st.session_state and st.session_state.prediction_results['success']:
        with st.expander("📊 Dataset Comparison"):
            original_probs = {k: v for k, v in st.session_state.prediction_results['probabilities'].items() if 'original' in k}
            processed_probs = {k: v for k, v in st.session_state.prediction_results['probabilities'].items() if 'processed' in k}
            
            if original_probs and processed_probs:
                orig_avg = np.mean(list(original_probs.values()))
                proc_avg = np.mean(list(processed_probs.values()))
                
                st.write("**Dataset Performance Comparison:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Original Dataset Avg", f"{orig_avg:.3f}")
                    st.write("• Comprehensive medical data")
                    st.write("• Higher accuracy overall")
                    st.write("• Detailed health features")
                
                with col2:
                    st.metric("Processed Dataset Avg", f"{proc_avg:.3f}")
                    st.write("• Symptom-focused data")
                    st.write("• Quick assessment")
                    st.write("• General health indicators")
                
                # Show which dataset gave higher prediction
                if orig_avg > proc_avg:
                    st.success("🏆 Original Dataset shows higher risk probability")
                elif proc_avg > orig_avg:
                    st.info("🔍 Processed Dataset shows higher risk probability")
                else:
                    st.info("⚖️ Both datasets show similar risk levels")
    
    # Footer
    st.markdown("---")
    st.markdown("🏥 **Disclaimer**: This tool is for educational purposes only and should not replace professional medical advice. Always consult with a healthcare provider for medical concerns.")

if __name__ == "__main__":
    main()
