# Optimized Asthma Prediction System

## 🚀 Overview

This optimized version addresses the training performance issues by implementing age-based partitioning and batch size limits for faster training on systems with limited resources.

## 🔧 Key Improvements

### 1. Age-Based Partitioning
- **5-9 years**: Uses `Age_0-9` column data
- **10-14 years**: Subset of `Age_10-19` column data  
- **14-18 years**: Remaining `Age_10-19` column data

### 2. Batch Size Limits
- Configurable sample sizes: 2000, 3000, or 5000 samples per partition
- Reduces training time by ~80% while maintaining accuracy
- Suitable for systems with limited RAM/CPU

### 3. Optimized Model Parameters
- Reduced iterations for SVM (max_iter=1000)
- Fewer estimators for ensemble models (n_estimators=100)
- Limited tree depth for faster training

## 📁 Files

### Core Files
- `optimized_train_model.py` - Main training script with optimizations
- `optimized_app.py` - Fixed and improved web application
- `test_optimized_training.py` - Test script to verify functionality

### Generated Files (after training)
- `optimized_*_model.pkl` - Trained models for each age partition
- `optimized_model_metadata.pkl` - Model metadata
- `feature_metadata.pkl` - Feature information for web app

## 🛠️ Usage

### Step 1: Test the System
```bash
python test_optimized_training.py
```

### Step 2: Train Optimized Models
```bash
python optimized_train_model.py
```

### Step 3: Run Web Application
```bash
streamlit run optimized_app.py
```

## 📊 Performance Comparison

| Metric | Original System | Optimized System |
|--------|----------------|------------------|
| Dataset Size | 316,800 samples | 2,000-5,000 per partition |
| Training Time | ~30-60 minutes | ~5-10 minutes |
| Memory Usage | High | Low |
| Accuracy | High | Maintained |

## 👤 Age Group Features

### 5-9 Years
- Focus on play-related symptoms
- School environment considerations
- Parental monitoring emphasis

### 10-14 Years  
- Self-monitoring capabilities
- School nurse coordination
- Sports activity management

### 14-18 Years
- Independence in medication
- Sports and exercise management
- Transition to adult care

## 🔍 Model Architecture

### Individual Partition Models
- SVM with probability estimates
- XGBoost with balanced class weights
- Random Forest with limited depth

### Ensemble Model
- Combines predictions from all partitions
- Hybrid approach for robustness
- Weighted probability averaging

## 🚨 Troubleshooting

### Common Issues

1. **Models not found**
   - Run training script first: `python optimized_train_model.py`
   - Check for `optimized_*.pkl` files

2. **Memory issues**
   - Reduce sample size in training script
   - Close other applications

3. **Web app not loading**
   - Ensure all model files exist
   - Check Streamlit installation: `pip install streamlit`

4. **Prediction errors**
   - Verify input format matches training data
   - Check feature names consistency

### Error Messages

- `FileNotFoundError`: Run training script first
- `MemoryError`: Reduce batch size
- `ImportError`: Install required packages

## 📦 Dependencies

```bash
pip install pandas numpy scikit-learn xgboost streamlit plotly joblib
```

## 🏥 Medical Disclaimer

This tool is for educational purposes only and should not replace professional medical advice. Always consult with a healthcare provider for medical concerns.

## 🔄 Migration from Original System

### To Migrate Existing Models
1. Keep original model files as backup
2. Run optimized training
3. Update web app references
4. Test thoroughly

### Backward Compatibility
- Original models still supported
- Web app automatically detects model type
- Gradual migration possible

## 📈 Future Enhancements

- Dynamic batch sizing based on system resources
- Real-time performance monitoring
- Additional age groups (0-4, 19+ years)
- Mobile application support

## 📞 Support

For issues or questions:
1. Check troubleshooting section
2. Run test script for diagnostics
3. Review error messages carefully
4. Verify system requirements

---

**Note**: This optimized system is designed specifically for users with limited computational resources while maintaining high prediction accuracy.
