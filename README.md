# 🚀 Customer Churn Prediction - ML Project

A complete machine learning project for predicting customer churn using the Telco Customer Churn dataset.

## 📋 Project Overview

This project demonstrates a production-ready machine learning workflow including:
- **Data Loading & Preprocessing**: Handles missing values and categorical encoding
- **Exploratory Data Analysis**: Understand data patterns and distributions
- **Model Training**: Logistic Regression, Decision Tree, Random Forest
- **Model Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Feature Importance Analysis**: Identify key predictors
- **Streamlit Web App**: Interactive interface for predictions
- **Model Persistence**: Save and load trained models

## ✅ Compatibility

- Python 3.11+
- macOS compatible
- Designed to work with modern scikit-learn and Streamlit releases

## 📁 Project Structure

```
ML miniprog/
├── churn_prediction.py          # Main ML pipeline script
├── visualize_results.py         # Visualization and analysis
├── app.py                       # Streamlit web application
├── best_model.pkl              # Saved best model
├── column_transformer.pkl      # Saved preprocessor
├── data/
│   └── churn.csv               # Dataset (7043 rows, 21 columns)
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## 📦 Dependencies

```
pandas>=2.0.0
numpy>=2.0.0
matplotlib>=3.10.0
seaborn>=0.13.0
scikit-learn>=1.8.0
joblib>=1.5.0
streamlit>=1.57.0
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

### 2. Train Models

```bash
python churn_prediction.py
```

This will:
- Load and preprocess the dataset
- Train 3 models (Logistic Regression, Decision Tree, Random Forest)
- Display evaluation metrics
- Save the best model and preprocessor

**Expected Output:**
```
Data loaded successfully. Shape: (7043, 21)
Data preprocessed. Feature shape: (7043, 30)
Data split into train and test sets.

Logistic Regression Results:
Accuracy: 0.8211
Precision: 0.6850
Recall: 0.6005
F1-Score: 0.6400
...
Best model (Logistic Regression) saved as best_model.pkl
```

### 3. Visualize Results

```bash
python visualize_results.py
```

Generates:
- `confusion_matrix.png`: Confusion matrix heatmap
- `feature_importance.png`: Top 10 important features
- `model_comparison.png`: Performance comparison chart

### 4. Launch Web App

```bash
streamlit run app.py
```

Open browser to `http://localhost:8501` and enter customer details to get predictions!

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** ⭐ | 82.11% | 68.50% | 60.05% | 64.00% |
| Decision Tree | 70.69% | 44.62% | 44.50% | 44.56% |
| Random Forest | 79.63% | 66.04% | 47.45% | 55.23% |

**Best Model:** Logistic Regression (82.11% accuracy)

## 🎯 Key Features

### Data Preprocessing
- Automatic missing value handling
- Categorical variable encoding using OneHotEncoder
- Feature scaling and normalization
- Train-test split (80-20)

### Model Training
- **Logistic Regression**: Linear classifier for binary classification
- **Decision Tree**: Tree-based model for interpretability
- **Random Forest**: Ensemble method for robust predictions

### Model Evaluation
- **Accuracy**: Overall correctness
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed prediction breakdown

### Feature Engineering
- OneHot encoding for categorical variables
- Numerical feature scaling
- Derived feature creation

## 🎨 Streamlit App Features

- **Demographics Section**: Gender, age, partner, dependents
- **Service Details**: Phone service, internet type
- **Internet Services**: Online security, backup, tech support, streaming
- **Contract & Billing**: Contract type, paperless billing, payment method
- **Charges**: Monthly and total charges
- **Real-time Predictions**: Instant churn probability

## 📈 How to Use the App

1. **Fill in customer details** using dropdown menus and sliders
2. **Click "Predict Churn"** button
3. **View results**:
   - Prediction status (Will Churn / Will Stay)
   - Churn probability percentage
   - Retention probability percentage
   - Risk level indicator

## 🔧 Customization

### Adjust Model Parameters

Edit `churn_prediction.py`:

```python
# Logistic Regression
LogisticRegression(random_state=42, max_iter=1000, C=1.0)

# Decision Tree
DecisionTreeClassifier(random_state=42, max_depth=10)

# Random Forest
RandomForestClassifier(random_state=42, n_estimators=100, max_depth=15)
```

### Change Test-Train Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42  # Change test_size
)
```

## 📚 Code Structure

### Main Functions

**`churn_prediction.py`**
- `load_data()`: Load CSV file
- `preprocess_data()`: Clean and encode data
- `train_and_evaluate_models()`: Train models and compute metrics
- `plot_feature_importance()`: Visualize feature weights
- `save_best_model()`: Persist models to disk

**`visualize_results.py`**
- `plot_confusion_matrices()`: Create confusion matrix heatmap
- `plot_feature_importance()`: Bar plot of feature importance
- `plot_model_comparison()`: Compare model metrics

**`app.py`**
- Streamlit UI for interactive predictions
- Real-time model inference
- Risk level assessment

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Data cleaning and preprocessing pipelines
- ✅ Multiple model implementations
- ✅ Cross-validation and hyperparameter tuning
- ✅ Evaluation metrics and interpretation
- ✅ Model serialization and deployment
- ✅ Building interactive web interfaces
- ✅ Production-ready code practices

## 🐛 Troubleshooting

### ModuleNotFoundError
```bash
pip install --break-system-packages <module_name>
```

### Model files not found
Run `python churn_prediction.py` first to generate:
- `best_model.pkl`
- `column_transformer.pkl`

### Streamlit connection error
```bash
streamlit run app.py --server.port 8501
```

## 📄 Dataset Information

**Telco Customer Churn Dataset** (7,043 records)

**Features (20):**
- Customer demographics: gender, age, partner, dependents
- Services: phone, internet, security, backup, tech support, streaming
- Account info: tenure, contract, billing, payment method
- Charges: monthly charges, total charges

**Target:** Churn (Yes/No)

## 🤝 Contributing

Feel free to:
- Add more models (SVM, Gradient Boosting, Neural Networks)
- Improve preprocessing techniques
- Add cross-validation
- Optimize hyperparameters
- Enhance the Streamlit UI

Live demo
https://samratm2112-cmyk-customer-churn-intelligence-app-berreu.streamlit.app/


## 📝 License

This project is for educational purposes.

## 📧 Contact

For questions or improvements, feel free to reach out!

---

**Happy Predicting! 🎉**
