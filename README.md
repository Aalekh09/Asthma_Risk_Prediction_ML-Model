# Asthma Risk Prediction ML Model

## 🚀 Overview

The Asthma Risk Prediction ML Model is a machine learning system designed to predict asthma risk in children and adolescents based on various health and environmental factors. The system uses age-based partitioning and ensemble modeling techniques to provide accurate predictions across different age groups.

## 🔧 Key Features

### Age-Based Partitioning
- **5-9 years**: Uses `Age_0-9` column data with focus on play-related symptoms and school environment
- **10-14 years**: Subset of `Age_10-19` column data with self-monitoring capabilities
- **14-18 years**: Remaining `Age_10-19` column data with independence in medication management

### Machine Learning Models
- **SVM (Support Vector Machine)**: With probability estimates for classification
- **XGBoost**: Gradient boosting with balanced class weights
- **Random Forest**: Ensemble decision trees with optimized parameters
- **Ensemble Model**: Combines predictions from all individual models for robustness

### Optimization Features
- Configurable batch sizes (2000, 3000, or 5000 samples per partition)
- Reduced training time by ~80% while maintaining accuracy
- Suitable for systems with limited RAM/CPU resources
- Memory-efficient processing for large datasets

## 📁 Project Structure

### Core Files
- `optimized_app.py` - Streamlit web application for predictions
- `optimized_train_model.py` - Main training script with optimizations
- `train_model.py` - Basic training script
- `test_optimized_training.py` - Test script for verification
- `test_both_datasets.py` - Testing script for both datasets

### Data Files
- `asthma_disease_data.csv` - Original dataset (518KB)
- `processed-data.csv` - Processed dataset (12MB)

### Model Files
- `optimized_*_model.pkl` - Trained models for each age partition
- `optimized_model_metadata.pkl` - Model metadata
- `optimized_feature_metadata.pkl` - Feature information
- `svm_model.pkl` - SVM model
- `xgb_model.pkl` - XGBoost model

### Supporting Files
- `requirements.txt` - Python dependencies
- `ds.html` - Data visualization exports
- `README_OPTIMIZED.md` - Detailed optimization documentation

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- pandas>=1.4.0
- numpy>=1.23.0
- scikit-learn>=1.0.0
- xgboost>=1.5.0
- streamlit>=1.50.0
- plotly>=6.0.0
- joblib>=1.0.0

## 🚀 Usage

### Step 1: Train the Models
```bash
python optimized_train_model.py
```

This will:
- Load and preprocess the asthma dataset
- Partition data by age groups (5-9, 10-14, 14-18)
- Train SVM, XGBoost, and Random Forest models for each partition
- Create an ensemble model combining all predictions
- Save trained models and metadata

### Step 2: Run the Web Application
```bash
streamlit run optimized_app.py
```

The web application provides:
- Interactive form for inputting patient data
- Real-time asthma risk prediction
- Visual risk assessment with color-coded indicators
- Feature importance analysis
- Model performance metrics

### Step 3: Test the System
```bash
python test_optimized_training.py
```

## 📊 Model Performance

### Performance Metrics
| Metric | Original System | Optimized System |
|--------|----------------|------------------|
| Dataset Size | 316,800 samples | 2,000-5,000 per partition |
| Training Time | ~30-60 minutes | ~5-10 minutes |
| Memory Usage | High | Low |
| Accuracy | High | Maintained |

### Age Group Specific Features

#### 5-9 Years
- Focus on play-related symptoms
- School environment considerations
- Parental monitoring emphasis

#### 10-14 Years
- Self-monitoring capabilities
- School nurse coordination
- Sports activity management

#### 14-18 Years
- Independence in medication
- Sports and exercise management
- Transition to adult care

## 🏥 Medical Disclaimer

**IMPORTANT**: This tool is for educational and research purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns, diagnosis, or treatment.

## 🔍 Model Architecture

### Individual Partition Models
- **SVM**: Radial basis function kernel with probability calibration
- **XGBoost**: Gradient boosting with balanced class weights and early stopping
- **Random Forest**: Limited depth trees with bootstrap aggregation

### Ensemble Model
- Combines predictions from all age partitions
- Hybrid approach for robustness
- Weighted probability averaging
- Cross-validation for optimal weights

## 🚨 Troubleshooting

### Common Issues

1. **Models not found**
   - Solution: Run training script first: `python optimized_train_model.py`
   - Check for `optimized_*.pkl` files in the directory

2. **Memory issues**
   - Solution: Reduce sample size in training script
   - Close other applications to free up RAM

3. **Web app not loading**
   - Solution: Ensure all model files exist
   - Check Streamlit installation: `pip install streamlit`

4. **Prediction errors**
   - Solution: Verify input format matches training data
   - Check feature names consistency

### Error Messages

- `FileNotFoundError`: Run training script first
- `MemoryError`: Reduce batch size in training script
- `ImportError`: Install required packages from requirements.txt

## 📈 Future Enhancements

- [ ] Dynamic batch sizing based on system resources
- [ ] Real-time performance monitoring dashboard
- [ ] Additional age groups (0-4, 19+ years)
- [ ] Mobile application support
- [ ] Integration with electronic health records
- [ ] Multi-language support
- [ ] Advanced explainable AI features

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Run test script for diagnostics
3. Review error messages carefully
4. Verify system requirements

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- Developed for asthma risk prediction research
- Uses open-source machine learning libraries
- Designed with healthcare applications in mind

---

**Note**: The Asthma Risk Prediction ML Model is designed to provide accurate predictions while being computationally efficient, making it suitable for deployment in resource-constrained environments.
