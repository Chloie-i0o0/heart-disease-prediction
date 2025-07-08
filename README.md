## Heart Disease Prediction for Insurance Risk Stratification
### Preface
>This project was an industry-academic collaboration with Cathay Life Insurance and NCCU Machine Learning and AI course.
Our goal is to predict the risk of suffering from heart disease using Kaggle health survey data and to translate our model insights into a business strategy in the life insurance industry.

### Problem Statement
>Purpose: Using customer's health data to predict whether the customer will suffer heart disease in the future, and provide a model for insurance company to do risk evaluation.

Based on the following 3 backgrounds:
>1. New regulation - **IFRS 17** and **ICS** conducted in life insurance industry to capture the **health insurance market** to CSM
>2. Cathay Life integrated all of their health insurance into **spillover mechanism** in 2025 Q2
>3. In 2023, heart disease accounted for **14.2%** of all life insurance claims in Taiwan, ranking as the **second most common cause**.

We set our **BUSINESS PROBLEM** as
>* Identify high-risk heart disease groups using health data and predictive models
>* Improve underwriting and policyholder management
>* Optimize spillover insurance product design
### Dataset
> * **Source**: Kaggle Indicators of Heart Disease Dataset (2022 UPDATE)
> * **Dataset Size**: 445,132 records
> * **Features**:heart_disease, gender, age, self_reported_health and other features regarding Basic Information and Lifestyle Habits, Health Status and Medical History, Functional Impairments and Physical Limitations, Vaccination and Screening Records
### Methods

#### Data Processing
* Outlier removal on BMI and other skewed variables
* One-hot encoding

#### Modeling
* **Baseline models**: Logistic Regression, Decision Tree, Random Forest, XGBoost
* **Imbalance Solutions**: Over Sampling, Under Sampling, SMOTE, Class Weight(final choice)
* **Hyperparameter Tuning and Optimization**: Grid search
* **VotingClassifier**: Soft Voting Classifier combining XGBoost, RF, LR, LightGBM
* **Feature Selection**: Feature Importanc and SHAP
#### Evaluation Metrics
* AUC, F1-score, **Recall (priority)**, Precision, Accuracy
* Classification Report
* Confusion Matrix

#### Key Results
* **Best Model**:
![classification_report](./figures/08_final_results/final_model_classification_report.png)
![confusion_matrix](./figures/08_final_results/final_model_confusion_matrix.png)
* **Important Feature**:'ChestScan_Yes', 'Sex_Male', 'HadStroke_Yes', 'GeneralHealth', 'RemovedTeeth_None of them', 'AgeCategory', 'HadDiabetes_Yes', 'RemovedTeeth_All', 'SmokerStatus_Never smoked', 'HadAngina_Yes'...
* **Risk Stratification**:
![probability_distribution](./figures/08_final_results/predicted_heart_disease_probability_distribution.png)
#### Business Impact
Our result could be applied to progression in the industry:
* **Underwriting**: Use model to classify applicants into high/moderate/low risk. Incorporate non-lifestyle features into underwriting to reduce adverse selection.
* **Policy Management & Claims**: Apply lifestyle features in incentive-based policies to encourage healthy behavior and reduce claim costs.
#### Project Structure
- data/ # deleted_outlier-2.csv
- notebooks/
  - 01_data_preprocessing.ipynb
  - 02_model_xgboost(main).ipynb
  - 03_model_logistic_regression.ipynb
  - 04_model_decision_tree.ipynb
  - 05_model_random_forest.ipynb
- figures/
  - 01_outlier/ # bmi_distribution
  - 02_eda/ # heart_disease, gender, age, health, Health And Lifestyle features...
  - 03_model_baseline/ #model_comparison_table, classification_report, confusion_matrix, train_test_metrics
  - 04_imbalance_handling/ #Over Sampling, Under Sampling, SMOTE, Class Weight
  - 05_hyperparameter_tuning/ # gridsearch
  - 06_model_ensemble/ # soft_voting_classification_report, confusion_matrix 
  - 07_feature_interpretation/ # feature_importance and SHAP Top20
  - 08_final_results/ #classification_report, confusion_matrix, probability_distribution
- requirements.txt
- README.md
#### Tools
- Python, VS Code
- Scikit-learn, XGBoost, LightGBM
- SHAP, Matplotlib, Pandas
#### Author
Chien-Ying Chen 
Project Lead – Responsible for business problem framing, full modeling pipeline (XGBoost, Soft Voting).  
Final project score: 96/100; commended by both professor and Cathay Life management.