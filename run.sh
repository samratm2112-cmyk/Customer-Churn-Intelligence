#!/bin/bash
# Quick start script for the Churn Prediction Project

echo "🚀 Customer Churn Prediction Project"
echo "===================================="
echo ""

# Check if models exist
if [ ! -f "best_model.pkl" ] || [ ! -f "column_transformer.pkl" ]; then
    echo "📚 Training models..."
    python3 churn_prediction.py
    echo ""
fi

# Check if data exists
if [ ! -f "data/churn.csv" ]; then
    echo "⚠️  Dataset not found! Please ensure data/churn.csv exists."
    exit 1
fi

# Ask user what to do
echo "Select an option:"
echo "1) Run Streamlit App (predictions)"
echo "2) Generate Visualizations"
echo "3) View Model Performance"
echo "4) Train Models Again"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "🎨 Launching Streamlit app..."
        streamlit run app.py
        ;;
    2)
        echo "📊 Generating visualizations..."
        python3 visualize_results.py
        ;;
    3)
        echo "📈 Model Performance Summary:"
        echo "Logistic Regression: 82.11% accuracy"
        echo "Random Forest: 79.63% accuracy"
        echo "Decision Tree: 70.69% accuracy"
        ;;
    4)
        echo "🔄 Retraining models..."
        python3 churn_prediction.py
        ;;
    *)
        echo "Invalid option"
        ;;
esac
