# Fraud Prediction Analysis

##  Project Overview
Fraud detection is a critical challenge in various industries, especially in insurance and financial sectors. This project focuses on predicting fraudulent claims using machine learning techniques. The dataset contains customer and claim-related information, and we aim to build a model that can accurately classify claims as fraudulent or legitimate.

##  Project Structure
```
fraud_detection_project/
│-- data/
│   │-- fraud_insurance.xlsx         # Raw dataset
│   │-- cleaned_fraud_data.xlsx      # Cleaned and preprocessed data
│   │-- resampled_fraud_data.xlsx    # Balanced dataset using SMOTE
│-- models/
│   │-- fraud_model.pkl              # Trained Machine Learning Model
│-- results/
│   │-- fraud_predictions.xlsx       # Model predictions
│-- insurance.py                     # Python script for analysis
│-- README.md                        # Project documentation
```

##  Data Preprocessing
- **Handling Missing Values**: Missing numerical values were filled with the median, while categorical features were imputed with the mode.
- **Feature Encoding**: Categorical variables were transformed using Label Encoding.
- **Datetime Processing**: Extracted year, month, and day features from datetime columns.
- **Feature Selection**: Removed unnecessary columns such as `policy_number` and `insured_zip`.
- **Handling Imbalanced Data**: Used **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset.

##  Machine Learning Model
- **Model Used**: Random Forest Classifier
- **Hyperparameter Tuning**: GridSearchCV was used to optimize model parameters.
- **Model Training & Testing**: Data was split into 80% training and 20% testing sets.

##  Model Evaluation
- **Accuracy Score**: XX%
- **Classification Report**:
```
Precision    Recall    F1-Score    Support
-------------------------------------------
Fraudulent     X.XX       X.XX        X.XX       XXX
Legitimate     X.XX       X.XX        X.XX       XXX
```

##  How to Run the Project
1. Clone the repository:
   ```sh
   git clone https://github.com/rushikeshkgholap/Fraud-Insurance.git
   cd Fraud-Insurance
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the fraud detection script:
   ```sh
   python insurance.py
   ```

##  Future Improvements
- Implement more advanced ML models (XGBoost, Neural Networks).
- Explore deep learning techniques for better fraud detection.
- Deploy the model as a web service using Flask/Django.

##  Contributors
- **Rushikesh Gholap** (Author & Data Scientist)

---
###  Feel free to contribute, raise issues, and star this repository if you find it useful!


