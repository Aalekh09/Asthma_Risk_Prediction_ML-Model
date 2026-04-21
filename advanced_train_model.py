import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedAsthmaPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.imputers = {}
        self.feature_names = {}
        
    def train_original_dataset(self):
        """Train on original asthma dataset"""
        print("=== Training on Original Asthma Dataset ===")
        data = pd.read_csv("asthma_disease_data.csv")
        
        # Filter for children
        data = data[data["Age"] < 18]
        print(f"Dataset shape after age filter: {data.shape}")
        
        # Select all relevant features (excluding ID and doctor info)
        feature_cols = [col for col in data.columns if col not in ['PatientID', 'DoctorInCharge', 'Diagnosis']]
        X = data[feature_cols]
        y = data["Diagnosis"]
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train multiple models
        models_config = {
            'svm': SVC(probability=True, class_weight='balanced', random_state=42),
            'xgb': XGBClassifier(scale_pos_weight=18, random_state=42),
            'rf': RandomForestClassifier(class_weight='balanced', random_state=42)
        }
        
        results = {}
        for name, model in models_config.items():
            print(f"Training {name.upper()} model...")
            model.fit(X_train, y_train)
            
            # Predictions and accuracy
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': model.predict_proba(X_test)
            }
            
            print(f"{name.upper()} Accuracy: {accuracy:.4f}")
        
        # Create hybrid model
        svm_prob = results['svm']['probabilities'][:, 1]
        xgb_prob = results['xgb']['probabilities'][:, 1]
        rf_prob = results['rf']['probabilities'][:, 1]
        
        hybrid_prob = (svm_prob + xgb_prob + rf_prob) / 3
        hybrid_pred = (hybrid_prob > 0.5).astype(int)
        hybrid_accuracy = accuracy_score(y_test, hybrid_pred)
        
        print(f"Hybrid Model Accuracy: {hybrid_accuracy:.4f}")
        
        # Store models and preprocessing objects
        self.models['original'] = results
        self.scalers['original'] = scaler
        self.imputers['original'] = imputer
        self.feature_names['original'] = feature_cols
        
        return results, hybrid_accuracy
    
    def train_processed_dataset(self):
        """Train on processed dataset"""
        print("\n=== Training on Processed Dataset ===")
        data = pd.read_csv("processed-data.csv")
        print(f"Processed dataset shape: {data.shape}")
        
        # Create target variable based on severity
        # Assuming Severity_None = 0, Severity_Mild = 1, Severity_Moderate = 2
        # Convert to binary: No symptoms (0) vs Has symptoms (1)
        data['target'] = ((data['Severity_Mild'] == 1) | (data['Severity_Moderate'] == 1)).astype(int)
        
        # Features are all columns except target and severity columns
        severity_cols = ['Severity_Mild', 'Severity_Moderate', 'Severity_None']
        feature_cols = [col for col in data.columns if col not in severity_cols + ['target']]
        X = data[feature_cols]
        y = data['target']
        
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train models
        models_config = {
            'svm': SVC(probability=True, class_weight='balanced', random_state=42),
            'xgb': XGBClassifier(scale_pos_weight=2, random_state=42),
            'rf': RandomForestClassifier(class_weight='balanced', random_state=42)
        }
        
        results = {}
        for name, model in models_config.items():
            print(f"Training {name.upper()} model...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': model.predict_proba(X_test)
            }
            
            print(f"{name.upper()} Accuracy: {accuracy:.4f}")
        
        # Create hybrid model
        svm_prob = results['svm']['probabilities'][:, 1]
        xgb_prob = results['xgb']['probabilities'][:, 1]
        rf_prob = results['rf']['probabilities'][:, 1]
        
        hybrid_prob = (svm_prob + xgb_prob + rf_prob) / 3
        hybrid_pred = (hybrid_prob > 0.5).astype(int)
        hybrid_accuracy = accuracy_score(y_test, hybrid_pred)
        
        print(f"Hybrid Model Accuracy: {hybrid_accuracy:.4f}")
        
        # Store models
        self.models['processed'] = results
        self.feature_names['processed'] = feature_cols
        
        return results, hybrid_accuracy
    
    def save_models(self):
        """Save all trained models"""
        print("\n=== Saving Models ===")
        
        # Save original dataset models
        for name, result in self.models['original'].items():
            joblib.dump(result['model'], f"original_{name}_model.pkl")
        
        joblib.dump(self.scalers['original'], "original_scaler.pkl")
        joblib.dump(self.imputers['original'], "original_imputer.pkl")
        
        # Save processed dataset models
        for name, result in self.models['processed'].items():
            joblib.dump(result['model'], f"processed_{name}_model.pkl")
        
        # Save feature names and metadata
        metadata = {
            'original_features': self.feature_names['original'],
            'processed_features': self.feature_names['processed'],
            'model_types': ['svm', 'xgb', 'rf']
        }
        joblib.dump(metadata, "model_metadata.pkl")
        
        print("All models saved successfully!")
    
    def generate_feature_importance(self):
        """Generate feature importance for tree-based models"""
        print("\n=== Feature Importance Analysis ===")
        
        importance_data = {}
        
        # Original dataset - XGBoost
        if 'original' in self.models and 'xgb' in self.models['original']:
            xgb_model = self.models['original']['xgb']['model']
            importance = xgb_model.feature_importances_
            feature_names = self.feature_names['original']
            
            importance_data['original'] = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
            
            print("Original Dataset - XGBoost Feature Importance:")
            for feat, imp in importance_data['original'][:10]:
                print(f"  {feat}: {imp:.4f}")
        
        # Processed dataset - XGBoost
        if 'processed' in self.models and 'xgb' in self.models['processed']:
            xgb_model = self.models['processed']['xgb']['model']
            importance = xgb_model.feature_importances_
            feature_names = self.feature_names['processed']
            
            importance_data['processed'] = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
            
            print("\nProcessed Dataset - XGBoost Feature Importance:")
            for feat, imp in importance_data['processed'][:10]:
                print(f"  {feat}: {imp:.4f}")
        
        return importance_data
    
    def get_feature_metadata(self):
        """Get metadata about features for web interface"""
        metadata = {
            'original_features': {
                'Age': {'type': 'numeric', 'min': 5, 'max': 18, 'description': 'Age of child'},
                'Gender': {'type': 'binary', 'options': [0, 1], 'description': 'Gender (0: Female, 1: Male)'},
                'Ethnicity': {'type': 'categorical', 'options': [0, 1, 2], 'description': 'Ethnicity group'},
                'EducationLevel': {'type': 'categorical', 'options': [0, 1, 2, 3], 'description': 'Education level'},
                'BMI': {'type': 'numeric', 'min': 10.0, 'max': 40.0, 'description': 'Body Mass Index'},
                'Smoking': {'type': 'binary', 'options': [0, 1], 'description': 'Smoking exposure'},
                'PhysicalActivity': {'type': 'numeric', 'min': 0.0, 'max': 10.0, 'description': 'Physical activity level'},
                'DietQuality': {'type': 'numeric', 'min': 0.0, 'max': 10.0, 'description': 'Diet quality score'},
                'SleepQuality': {'type': 'numeric', 'min': 0.0, 'max': 10.0, 'description': 'Sleep quality score'},
                'PollutionExposure': {'type': 'numeric', 'min': 0.0, 'max': 10.0, 'description': 'Pollution exposure level'},
                'PollenExposure': {'type': 'numeric', 'min': 0.0, 'max': 10.0, 'description': 'Pollen exposure level'},
                'DustExposure': {'type': 'numeric', 'min': 0.0, 'max': 10.0, 'description': 'Dust exposure level'},
                'PetAllergy': {'type': 'binary', 'options': [0, 1], 'description': 'Pet allergy present'},
                'FamilyHistoryAsthma': {'type': 'binary', 'options': [0, 1], 'description': 'Family history of asthma'},
                'HistoryOfAllergies': {'type': 'binary', 'options': [0, 1], 'description': 'History of allergies'},
                'Eczema': {'type': 'binary', 'options': [0, 1], 'description': 'Eczema present'},
                'HayFever': {'type': 'binary', 'options': [0, 1], 'description': 'Hay fever present'},
                'GastroesophagealReflux': {'type': 'binary', 'options': [0, 1], 'description': 'GERD present'},
                'LungFunctionFEV1': {'type': 'numeric', 'min': 0.5, 'max': 5.0, 'description': 'FEV1 lung function'},
                'LungFunctionFVC': {'type': 'numeric', 'min': 1.0, 'max': 8.0, 'description': 'FVC lung function'},
                'Wheezing': {'type': 'binary', 'options': [0, 1], 'description': 'Wheezing symptom'},
                'ShortnessOfBreath': {'type': 'binary', 'options': [0, 1], 'description': 'Shortness of breath'},
                'ChestTightness': {'type': 'binary', 'options': [0, 1], 'description': 'Chest tightness'},
                'Coughing': {'type': 'binary', 'options': [0, 1], 'description': 'Coughing symptom'},
                'NighttimeSymptoms': {'type': 'binary', 'options': [0, 1], 'description': 'Nighttime symptoms'},
                'ExerciseInduced': {'type': 'binary', 'options': [0, 1], 'description': 'Exercise induced symptoms'}
            },
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
        return metadata

def main():
    predictor = AdvancedAsthmaPredictor()
    
    # Train on both datasets
    orig_results, orig_hybrid_acc = predictor.train_original_dataset()
    proc_results, proc_hybrid_acc = predictor.train_processed_dataset()
    
    # Generate feature importance
    predictor.generate_feature_importance()
    
    # Save all models
    predictor.save_models()
    
    # Save feature metadata for web interface
    feature_metadata = predictor.get_feature_metadata()
    joblib.dump(feature_metadata, "feature_metadata.pkl")
    
    print(f"\n=== Final Results ===")
    print(f"Original Dataset Hybrid Accuracy: {orig_hybrid_acc:.4f}")
    print(f"Processed Dataset Hybrid Accuracy: {proc_hybrid_acc:.4f}")
    print("Feature metadata saved for web interface!")

if __name__ == "__main__":
    main()
