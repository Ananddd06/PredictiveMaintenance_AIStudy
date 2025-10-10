#!/bin/bash
echo "Installing required packages..."

pip3 install pandas==2.0.3
pip3 install numpy==1.24.3
pip3 install scikit-learn==1.3.0
pip3 install imbalanced-learn==0.11.0
pip3 install joblib==1.3.2
pip3 install streamlit==1.28.1
pip3 install plotly==5.17.0

echo "✅ All packages installed successfully!"
echo "Now run: python3 ml_pipeline.py"
