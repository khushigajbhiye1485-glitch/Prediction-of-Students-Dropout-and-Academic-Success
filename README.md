# Prediction of Students Dropout and Academic Success

A Machine Learning project that predicts a student's academic outcome as **Dropout, Enrolled, or Graduate** using student admission, demographic, financial, and economic information.

The project includes data analysis, feature selection, preprocessing, model comparison, hyperparameter tuning, evaluation, and a Streamlit application for prediction.

---

## 📌 Project Overview

Student dropout is an important issue in higher education. The purpose of this project is to use historical student data to build a classification model that can predict the student's academic outcome.

The project follows this workflow:

```text
Dataset
   ↓
Data Analysis & Cleaning
   ↓
Feature Audit
   ↓
Leakage Check
   ↓
Feature Selection
   ↓
Preprocessing
   ↓
Model Comparison
   ↓
Cross-Validation
   ↓
Hyperparameter Tuning
   ↓
Final Model
   ↓
Streamlit Application
```

---

## 🎯 Objectives

- Analyze the student dataset and understand its characteristics.
- Identify useful features for predicting student outcomes.
- Avoid temporal data leakage during feature selection.
- Preprocess categorical and numerical features appropriately.
- Compare different Machine Learning classification models.
- Tune the selected model using cross-validation.
- Evaluate the final model on unseen test data.
- Deploy the trained model through a Streamlit application.

---

## 🤖 Problem Statement

This is a **supervised multiclass classification** problem.

### Input

The model uses information available around the time of student enrollment, including:

- Personal information
- Admission information
- Previous qualification
- Parents' education and occupation
- Financial information
- Scholarship information
- Economic indicators

### Output

The model predicts one of three outcomes:

| Target Class | Meaning |
|---|---|
| Dropout | Student discontinued their studies |
| Enrolled | Student remained enrolled |
| Graduate | Student completed/graduated |

---

# 📊 Dataset

The project uses the **Predict Students' Dropout and Academic Success** dataset from the UCI Machine Learning Repository.

**Dataset:**  
https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

| Dataset Property | Value |
|---|---:|
| Total Records | 4,424 |
| Original Features | 35 |
| Target Variable | `Target` |
| Prediction Type | Multiclass Classification |
| Target Classes | 3 |

---

## 🎯 Target Distribution

| Target | Students | Percentage |
|---|---:|---:|
| Graduate | 2,209 | 49.93% |
| Dropout | 1,421 | 32.12% |
| Enrolled | 794 | 17.95% |
| **Total** | **4,424** | **100%** |

The target classes are not equally distributed, so **macro F1-score** is considered along with accuracy during model evaluation.

---

# 🔍 Data Analysis

The dataset was checked before model development.

| Check | Result |
|---|---:|
| Records | 4,424 |
| Original Columns | 35 |
| Missing Values | 0 |
| Duplicate Rows | 0 |
| Target Classes | 3 |

The analysis also included:

- Data types
- Unique values
- Target distribution
- Feature roles
- Correlation analysis
- Outlier review
- Feature leakage analysis

---

# ⚠️ Feature Selection and Data Leakage

The original dataset contains information about the student's first and second semester performance.

These variables include:

| Semester | Information |
|---|---|
| 1st Semester | Credited units |
| 1st Semester | Enrolled units |
| 1st Semester | Evaluations |
| 1st Semester | Approved units |
| 1st Semester | Grade |
| 1st Semester | Without evaluations |
| 2nd Semester | Credited units |
| 2nd Semester | Enrolled units |
| 2nd Semester | Evaluations |
| 2nd Semester | Approved units |
| 2nd Semester | Grade |
| 2nd Semester | Without evaluations |

A total of **12 semester-related features** were excluded.

### Why?

The project is designed for prediction around the **enrollment stage**.

First- and second-semester performance information is only available after the student has started studying. Using those variables for an enrollment-time prediction would introduce **temporal data leakage**.

Therefore, these features were removed from the final model.

---

# 🧩 Final Features

After removing the target, identifier-related information, and the 12 semester-performance variables, the model uses **22 predictors**.

| Feature Type | Count |
|---|---:|
| Categorical | 18 |
| Numerical | 4 |
| **Total** | **22** |

### Numerical Features

| No. | Feature |
|---:|---|
| 1 | Age at enrollment |
| 2 | Unemployment rate |
| 3 | Inflation rate |
| 4 | GDP |

### Categorical Features

| No. | Feature |
|---:|---|
| 1 | Marital status |
| 2 | Application mode |
| 3 | Application order |
| 4 | Course |
| 5 | Daytime/evening attendance |
| 6 | Previous qualification |
| 7 | Nacionality |
| 8 | Mother's qualification |
| 9 | Father's qualification |
| 10 | Mother's occupation |
| 11 | Father's occupation |
| 12 | Displaced |
| 13 | Educational special needs |
| 14 | Debtor |
| 15 | Tuition fees up to date |
| 16 | Gender |
| 17 | Scholarship holder |
| 18 | International |

---

# 🛠️ Data Preprocessing

The project uses different preprocessing methods for categorical and numerical variables.

| Feature Type | Preprocessing |
|---|---|
| Categorical | One-Hot Encoding |
| Numerical | Standard Scaling |

### Categorical Encoding

```python
OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)
```

One-Hot Encoding is used because the categorical values do not represent meaningful numerical order.

`handle_unknown="ignore"` allows the pipeline to handle an unseen category during prediction without producing an encoding error.

### Numerical Scaling

```python
StandardScaler()
```

StandardScaler puts numerical variables on a comparable scale before they are passed to the model.

---

# 🔗 Preprocessing Pipeline

The project uses `ColumnTransformer` to apply the appropriate transformation to each feature group.

```text
                Input Data
                    |
          ----------------------
          |                    |
     Categorical            Numerical
          |                    |
  One-Hot Encoding       Standard Scaling
          |                    |
          -----------+----------
                     |
              Processed Features
                     |
                ML Model
```

The preprocessing and classifier are combined into a single **scikit-learn Pipeline** so that the same transformations are used during training and prediction.

---

# 🤖 Machine Learning Models

Five models were evaluated:

| Model | Purpose |
|---|---|
| Dummy Classifier | Baseline |
| Logistic Regression | Linear classification model |
| Decision Tree | Tree-based classification |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential ensemble model |

The Dummy Classifier provides a baseline against which the Machine Learning models can be compared.

---

# 📈 Model Validation

The dataset was divided using an **80/20 stratified train-test split**.

| Setting | Value |
|---|---|
| Training Data | 80% |
| Test Data | 20% |
| Split Type | Stratified |
| Random State | 42 |
| Test Samples | 885 |

Stratification helps maintain a similar class distribution in both the training and test sets.

---

# 🔬 Cross-Validation

Model comparison was performed using **5-fold Stratified Cross-Validation**.

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Macro F1 was considered important because the target classes are imbalanced.

---

# 📊 Model Comparison

The cross-validation results were:

| Model | CV Accuracy | CV Macro F1 |
|---|---:|---:|
| Dummy Classifier | 49.93% | 22.20% |
| Logistic Regression | 64.79% | 54.42% |
| Decision Tree | 61.43% | 49.19% |
| Random Forest | 64.54% | 52.16% |
| Gradient Boosting | 64.82% | 53.13% |

These results were used to select Logistic Regression for further tuning.

---

# ⚙️ Hyperparameter Tuning

Logistic Regression was tuned using `GridSearchCV`.

The following values of `C` were tested:

```python
[0.1, 0.3, 1, 3, 10]
```

The tuning configuration included:

| Parameter | Value |
|---|---|
| Model | Logistic Regression |
| Parameters Tested | `C` |
| C Values | 0.1, 0.3, 1, 3, 10 |
| Scoring | Macro F1 |
| Cross-Validation | 5-Fold Stratified |
| Class Weight | Balanced |
| Maximum Iterations | 3000 |

`class_weight="balanced"` was used to give more consideration to the less represented classes during training.

---

# 🏆 Final Model

The selected model is:

**Logistic Regression**

| Parameter | Final Value |
|---|---|
| Model | Logistic Regression |
| C | 0.1 |
| Class Weight | `balanced` |
| Maximum Iterations | 3000 |
| Preprocessing | One-Hot Encoding + StandardScaler |

The final model is stored together with the preprocessing steps as a single pipeline.

---

# 🧪 Final Test Results

The final model was evaluated on the held-out test set containing **885 students**.

| Metric | Score |
|---|---:|
| Accuracy | 57.40% |
| Macro Precision | 56.29% |
| Macro Recall | 56.51% |
| Macro F1 | 55.08% |
| Weighted F1 | 59.06% |

---

# 📋 Class-wise Results

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Dropout | 0.67 | 0.58 | 0.62 |
| Enrolled | 0.30 | 0.53 | 0.39 |
| Graduate | 0.71 | 0.59 | 0.64 |

The **Enrolled** class has the lowest F1-score among the three classes. This is an important limitation of the current model.

---

# 💾 Saved Model Files

The trained model and feature configuration are stored in the `models` directory.

| File | Purpose |
|---|---|
| `student_outcome_pipeline.joblib` | Saved preprocessing + trained model |
| `feature_config.json` | Model feature configuration |

Saving the complete pipeline means the application does not need to retrain the model whenever it starts.

---

# 🌐 Streamlit Application

The trained model is integrated into a Streamlit application.

The application collects student information and sends it through the saved Machine Learning pipeline.

### Application Flow

```text
User
 ↓
Streamlit Form
 ↓
Student Information
 ↓
Input DataFrame
 ↓
Saved ML Pipeline
 ↓
Preprocessing
 ↓
Logistic Regression
 ↓
Prediction + Probabilities
 ↓
Result
```

The application predicts:

```text
Dropout
Enrolled
Graduate
```

---

# 🏷️ User Interface

The dataset internally contains encoded categorical values.

Instead of exposing raw codes to the user, the application interface can display **human-readable labels** while maintaining the encoded values internally for the trained model.

This keeps the model input compatible with the training data while making the application easier to use.

---

# 🛡️ Application Validation

The application includes checks for:

- Required model files
- Feature configuration
- Dataset availability
- Expected feature names
- Prediction errors

The input DataFrame is created using the same feature order expected by the trained pipeline.

This helps prevent issues caused by missing or incorrectly ordered features.

---

# 📁 Project Structure

```text
project/
│
├── app.py
├── student_outcome_dashboard.py
├── dataset.csv
├── student_outcome_corrected.ipynb
│
├── models/
│   ├── student_outcome_pipeline.joblib
│   └── feature_config.json
│
└── README.md
```

---

# 🚀 How to Run

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

```bash
cd <PROJECT_FOLDER>
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

If the repository contains `requirements.txt`:

```bash
pip install -r requirements.txt
```

Otherwise, install the main dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib jupyter
```

---

## 4. Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

If the dashboard version is being used:

```bash
streamlit run student_outcome_dashboard.py
```

The application will normally open at:

```text
http://localhost:8501
```

---

# 📓 Running the Notebook

To open Jupyter Notebook:

```bash
jupyter notebook
```

Open the project notebook:

```text
student_outcome_corrected.ipynb
```

The notebook contains the complete workflow:

| Stage | Included |
|---|---|
| Dataset Loading | ✓ |
| Data Inspection | ✓ |
| Data Quality Checks | ✓ |
| Exploratory Analysis | ✓ |
| Feature Audit | ✓ |
| Leakage Check | ✓ |
| Feature Selection | ✓ |
| Preprocessing | ✓ |
| Model Comparison | ✓ |
| Cross-Validation | ✓ |
| Hyperparameter Tuning | ✓ |
| Final Evaluation | ✓ |
| Model Saving | ✓ |

---

# 🧰 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Matplotlib | Data visualization |
| Seaborn | Data visualization |
| Joblib | Model saving/loading |
| Streamlit | Web application |
| Jupyter Notebook | Model development |
| Git/GitHub | Version control |

---

# 👥 Team Contributions

| Member | Main Responsibility |
|---|---|
| Member 1 | Dataset understanding, data quality checks and EDA |
| Member 2 | Feature audit, feature selection and leakage analysis |
| Member 3 | Preprocessing and candidate model development |
| Member 4 | Cross-validation, model comparison, tuning and evaluation |
| Member 5 | Streamlit deployment and model integration |

### Member 1 — Data Understanding & EDA

- Dataset loading and inspection
- Data types
- Missing values
- Duplicate records
- Target distribution
- Unique-value analysis
- Exploratory analysis
- Correlation and outlier review

### Member 2 — Feature Audit & Leakage Analysis

- Feature-role analysis
- Feature auditing
- Temporal leakage analysis
- Identification of semester-related variables
- Final feature selection

### Member 3 — Preprocessing & Model Development

- Categorical/numerical feature separation
- One-Hot Encoding
- StandardScaler
- ColumnTransformer
- Pipeline
- Candidate model implementation

### Member 4 — Validation & Optimization

- Train/test split
- Stratified cross-validation
- Model comparison
- GridSearchCV
- Hyperparameter tuning
- Final test evaluation
- Classification report
- Confusion matrix

### Member 5 — Deployment & Application

- Streamlit application
- Saved model integration
- Joblib loading
- Feature configuration
- User input handling
- Prediction
- Probability display
- Application validation
- Error handling

---

# ❗ Limitations

| Limitation | Description |
|---|---|
| Model Accuracy | Final test accuracy is 57.40% |
| Class Imbalance | The three target classes have different sample sizes |
| Enrolled Class | F1-score is lower than the other classes |
| Historical Data | Model patterns are based on the available historical dataset |
| Prediction Timing | Semester-performance features were excluded for enrollment-time prediction |
| Generalization | Performance may change on a different student population or newer data |

The model should therefore be treated as a **prediction tool for academic demonstration**, not as a guaranteed prediction of an individual student's future.

---

# 🔮 Future Improvements

Possible improvements include:

- Testing additional Machine Learning algorithms.
- More extensive hyperparameter tuning.
- Feature engineering.
- Additional techniques for handling class imbalance.
- Collecting newer student data.
- Improving performance for the Enrolled class.
- Adding model explainability.
- Adding more visualizations to the application.
- Adding database integration.
- Deploying the application online.
- Adding authentication and user management.
- Monitoring model performance after deployment.

---

# 📚 Dataset Reference

**Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021).**

*Predict Students' Dropout and Academic Success.*

UCI Machine Learning Repository.

**DOI:** `10.24432/C5MC89`

Dataset:

https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

---

# 📜 License

The dataset is provided through the UCI Machine Learning Repository under its applicable licensing terms.

Refer to the official UCI dataset page for the current license and citation requirements.

---

# ⚠️ Disclaimer

This project was developed for **academic and educational purposes**.

The predictions generated by the application should not be treated as definitive statements about a student's future academic outcome.

The model is intended to demonstrate the application of Machine Learning techniques to student outcome prediction.

---

# 📌 Project Summary

| Component | Details |
|---|---|
| Problem | Student outcome prediction |
| Learning Type | Supervised Learning |
| Task | Multiclass Classification |
| Dataset Size | 4,424 records |
| Original Features | 35 |
| Final Predictors | 22 |
| Categorical Features | 18 |
| Numerical Features | 4 |
| Removed Semester Features | 12 |
| Train/Test Split | 80/20 |
| Cross-Validation | 5-Fold Stratified |
| Final Model | Logistic Regression |
| Final C | 0.1 |
| Final Accuracy | 57.40% |
| Final Macro F1 | 55.08% |
| Deployment | Streamlit |

---

## ⭐ End-to-End Workflow

```text
Dataset
   ↓
Data Understanding
   ↓
Data Quality Checks
   ↓
EDA
   ↓
Feature Audit
   ↓
Temporal Leakage Check
   ↓
Remove 12 Semester Features
   ↓
22 Final Predictors
   ↓
80/20 Stratified Split
   ↓
One-Hot Encoding + StandardScaler
   ↓
Model Comparison
   ↓
5-Fold Stratified Cross-Validation
   ↓
GridSearchCV
   ↓
Logistic Regression
   ↓
Final Test Evaluation
   ↓
Save Pipeline
   ↓
Streamlit Application
   ↓
Student Input
   ↓
Prediction
```

---

## 📌 Conclusion

This project covers the complete Machine Learning workflow from **dataset analysis to deployment**. The final trained pipeline uses 22 enrollment-time predictors and is integrated into a Streamlit application that allows users to enter student information and obtain a predicted academic outcome of **Dropout, Enrolled, or Graduate**.
