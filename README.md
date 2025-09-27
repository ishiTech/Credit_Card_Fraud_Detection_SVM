# 🔍 Credit Card Fraud Detection using SVM

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> A machine learning solution for detecting fraudulent credit card transactions using Linear Support Vector Machine with advanced threshold optimization.

## 🎯 Project Highlights

| Metric | Before Optimization | After Optimization | Improvement |
|--------|-------------------|-------------------|-------------|
| **Fraud Precision** | 6.96% | **86.96%** | +1,149% 📈 |
| **False Positives** | 294 alerts | **3 alerts** | -98.9% 🎯 |
| **F1-Score** | 0.129 | **0.851** | +560% 🚀 |
| **Overall Accuracy** | 97.91% | **99.95%** | +2.04% ✨ |

## 📊 The Challenge

Credit card fraud detection faces a critical business problem:
- **Massive class imbalance**: Only 0.173% of transactions are fraudulent
- **High cost of false positives**: Each false alarm frustrates customers
- **High cost of false negatives**: Each missed fraud results in financial loss

**Dataset**: 284,807 transactions, 31 features, severe imbalance (492 fraud vs 284,315 normal)

## 🔬 Technical Solution

### 1. **Data Preprocessing Pipeline**
```python
✅ Removed 1,081 duplicate transactions
✅ No missing values detected
✅ Excluded 'Time' feature (non-predictive)
✅ Maintained 29 PCA-transformed features
```

### 2. **Class Imbalance Handling**
```python
# Applied SMOTE oversampling
Before: 269,090 normal | 449 fraud  (598:1 ratio)
After:  269,090 normal | 269,090 fraud  (1:1 ratio)
```

### 3. **Model Architecture**
- **Linear SVM** with probability calibration
- **StandardScaler** for feature normalization
- **CalibratedClassifierCV** for reliable probability estimates
- **Custom threshold optimization** using precision-recall curve

### 4. **Business-Optimized Threshold**
- Default threshold (0.5) → **294 false alarms** 😡
- Optimized threshold (1.0) → **3 false alarms** ✅
- **Precision improved from 7% to 87%** while maintaining 83% recall

## 📈 Results & Business Impact

### Confusion Matrix Comparison

**Default Threshold (0.5):**
```
                Predicted
                Normal  Fraud
Actual Normal   13,869   294  ← 294 angry customers!
       Fraud        2    22  ← Good detection
```

**Optimized Threshold (1.0):**
```
                Predicted
                Normal  Fraud
Actual Normal   14,160     3  ← Only 3 false alarms!
       Fraud        4    20  ← Still catching frauds
```

### Business Value
- 🎯 **Customer Experience**: 98.9% fewer false fraud alerts
- 💰 **Cost Reduction**: Massive reduction in manual review workload
- 🔒 **Security**: Still catches 83% of actual fraud cases
- ⚡ **Scalability**: Linear SVM handles large datasets efficiently

## 🛠️ Technical Implementation

### Quick Start
```bash
# Clone repository
git clone https://github.com/ishiTech/Credit_Card_Fraud_Detection_SVM.git
cd Credit_Card_Fraud_Detection_SVM

# Download dataset from Kaggle and place as 'creditcard.csv'
# Run the notebook
jupyter notebook CreditCardFraudDetectionSVM.ipynb
```

### Key Technologies
- **Python 3.8+** - Core programming language
- **scikit-learn** - Machine learning framework
- **imbalanced-learn** - SMOTE implementation
- **pandas & NumPy** - Data manipulation
- **Jupyter** - Interactive development

### Why Linear SVM?
- ⚡ **10x faster** than RBF SVM on large datasets
- 📊 **Excellent performance** on high-dimensional data
- 🎯 **Scalable** to enterprise-level transaction volumes
- 🔧 **Interpretable** results for business stakeholders

## 🧠 Machine Learning Concepts Demonstrated

- [x] **Imbalanced Dataset Handling** - SMOTE oversampling
- [x] **Feature Engineering** - Scaling and preprocessing
- [x] **Model Selection** - Linear vs RBF SVM comparison
- [x] **Threshold Optimization** - Business-oriented tuning
- [x] **Probability Calibration** - Reliable confidence scores
- [x] **Cross-validation** - Robust model evaluation
- [x] **Business Metrics** - Precision-recall tradeoffs

## 📊 Model Performance Details

```python
🎯 Optimized Threshold Results:
├── Fraud Precision: 86.96% (20 true positives / 23 predictions)
├── Fraud Recall: 83.33% (20 detected / 24 actual frauds)  
├── F1-Score: 0.851 (excellent balance)
└── ROC-AUC: High performance on probability ranking
```

**What this means for business:**
- Out of 23 fraud alerts, 20 are real → **87% accuracy in alerts**
- Out of 24 real frauds, 20 are caught → **83% detection rate**
- Only 3 customers get false fraud notifications → **Happy customers**

## 🚀 Why This Project Stands Out

1. **Real-world Problem**: Addresses actual business challenges in fintech
2. **Advanced Techniques**: Beyond basic classification - threshold optimization
3. **Business Acumen**: Focuses on actionable metrics, not just accuracy
4. **Scalable Solution**: Linear SVM handles enterprise-scale data
5. **Complete Pipeline**: End-to-end ML workflow with proper evaluation

## 🔮 Future Enhancements

- [ ] **Ensemble Methods**: Random Forest, XGBoost comparison
- [ ] **Deep Learning**: Neural network architectures
- [ ] **Real-time API**: Flask/FastAPI deployment
- [ ] **Feature Engineering**: Transaction pattern analysis
- [ ] **Cost-sensitive Learning**: Asymmetric loss functions
- [ ] **Model Monitoring**: Drift detection and retraining

## 📚 Learning Outcomes

This project demonstrates proficiency in:
- **Data Science**: EDA, preprocessing, feature selection
- **Machine Learning**: Classification, imbalanced learning, optimization  
- **Business Intelligence**: Metric interpretation, stakeholder communication
- **Software Engineering**: Clean code, documentation, reproducibility

## 📄 License & Contributing

MIT License - feel free to fork, modify, and contribute!

---
