# Heart Disease Risk Prediction: Logistic Regression Project

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.4-green.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-3.0-yellow.svg)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-In%20Progress-orange.svg)]()

## 📋 Exercise Summary

This project implements **logistic regression from scratch** for heart disease prediction, covering:

- ✅ **Step 1: Data Loading & Preprocessing** - COMPLETED
- 🔄 **Step 2: Basic Logistic Regression Implementation** - In Progress
- ⏳ **Step 3: Decision Boundary Visualization** - Pending
- ⏳ **Step 4: Regularization (L2)** - Pending
- ⏳ **Step 5: AWS SageMaker Deployment** - Pending

### Key Features
- **Custom implementations** of sigmoid, cost function, and gradient descent (no scikit-learn for core training)
- **Exploratory Data Analysis (EDA)** with comprehensive visualizations
- **Feature engineering** and normalization
- **Decision boundary visualization** for multiple feature pairs
- **Regularization** with hyperparameter tuning
- **Model deployment** simulation on AWS SageMaker

---

## 📊 Dataset Description

### Source
**Kaggle Heart Disease Dataset**  
🔗 https://www.kaggle.com/datasets/neurocipher/heartdisease

### Overview
- **Origin:** UCI Machine Learning Repository
- **Total Samples:** 270 patient records (after cleaning)
- **Features:** 14 clinical and diagnostic features
- **Target:** Binary classification
  - `1` = Presence of heart disease (120 samples, 44.4%)
  - `0` = Absence of heart disease (150 samples, 55.6%)

### Selected Features for Modeling (6 features)

| Feature | Description | Range | Correlation with Target |
|---------|-------------|-------|-------------------------|
| **Age** | Patient age (years) | 29-77 | +0.212 (weak positive) |
| **Cholesterol** | Serum cholesterol (mg/dL) | 126-564 | +0.118 (weak positive) |
| **BP** | Resting blood pressure (mm Hg) | 94-200 | +0.155 (weak positive) |
| **Max HR** | Maximum heart rate achieved | 71-202 | -0.419 (moderate negative) |
| **ST depression** | ST depression induced by exercise | 0.0-6.2 | +0.418 (moderate positive) |
| **Number of vessels fluro** | Major vessels colored by fluoroscopy (0-3) | 0-3 | +0.455 (strong positive) |

### Data Quality
✓ **No missing values**  
✓ **Relatively balanced classes** (44.4% / 55.6%)  
✓ **Outliers preserved** (valid extreme clinical cases)

---

## 🚀 Step 1 Progress: Data Loading & Preparation ✅

### Completed Tasks

#### 1. Data Loading
- [x] Loaded CSV file from Kaggle
- [x] Verified dataset structure (270 rows × 14 columns)
- [x] Displayed column names and data types

#### 2. Exploratory Data Analysis (EDA)
- [x] Generated summary statistics for all features
- [x] Checked for missing values (0 found)
- [x] Identified and visualized outliers using box plots
- [x] Analyzed class distribution (balanced dataset)

#### 3. Target Variable Binarization
- [x] Converted `"Presence"/"Absence"` to `1/0`
- [x] Verified disease prevalence: **44.44%**

#### 4. Feature Selection
- [x] Selected 6 most relevant clinical features
- [x] Computed correlation matrix
- [x] Visualized feature correlations with heatmap
- [x] Identified strongest predictors:
  - **Number of vessels** (+0.455)
  - **ST depression** (+0.418)
  - **Max HR** (-0.419)

#### 5. Data Splitting
- [x] Implemented custom **stratified train-test split** (70/30)
- [x] Training set: 189 samples (70%)
- [x] Test set: 81 samples (30%)
- [x] Verified class distribution preservation

#### 6. Feature Normalization
- [x] Implemented **z-score standardization** from scratch
- [x] Fit on training data only (no data leakage)
- [x] Transformed both train and test sets
- [x] Verified normalization: mean ≈ 0, std ≈ 1

### Key Visualizations Generated
1. **Box plots** for outlier detection (5 numerical features)
2. **Bar chart + Pie chart** for class distribution
3. **Correlation heatmap** showing feature relationships

### Data Insights Summary

📈 **Class Balance:** Dataset is relatively balanced (not severely imbalanced)  
📊 **Feature Relationships:** Number of vessels and ST depression show strongest correlation with disease presence  
🔍 **Data Quality:** Clean dataset with no missing values; outliers retained as valid medical extremes  
✨ **Preprocessing:** Data successfully normalized and split, ready for model training

---

## 🛠️ Technologies Used

### Core Libraries
- **NumPy** 2.4.1 - Numerical computations
- **Pandas** 3.0.0 - Data manipulation
- **Matplotlib** 3.x - Visualization
- **Seaborn** 0.x - Statistical plotting

### Development Environment
- **Python** 3.13.9
- **Jupyter Notebook** - Interactive development
- **VS Code** - Code editor

---

## 📁 Project Structure

```
ClassificationAndLogisticRegression/
│
├── heart_disease_lr_analysis.ipynb    # Main analysis notebook (Step 1 ✅)
├── Heart_Disease_Prediction.csv       # Dataset from Kaggle
├── README.md                          # This file
│
├── week2_classification_hour1_final.ipynb                           # Course materials
├── week2_classification_hour2_regularization_with_derivatives.ipynb # (reference)
└── APENDIX-RidgeVsGradientDescentInRegularizedLinearRegression.ipynb
```

---

## 🔬 Next Steps (Step 2: Logistic Regression Implementation)

### To-Do List
- [ ] Implement sigmoid function: $\sigma(z) = \frac{1}{1 + e^{-z}}$
- [ ] Implement binary cross-entropy cost function
- [ ] Implement gradient computation
- [ ] Implement gradient descent algorithm
- [ ] Train model on full training set (α ≈ 0.01, 1000+ iterations)
- [ ] Plot cost vs. iterations (convergence analysis)
- [ ] Implement prediction function (threshold = 0.5)
- [ ] Evaluate metrics: accuracy, precision, recall, F1-score
- [ ] Generate metrics table for train and test sets

---

## 📝 Deployment Evidence (Step 5 - Pending)

> **Note:** This section will be updated after completing AWS SageMaker deployment.

### Planned Deliverables
1. **Screenshot 1:** SageMaker training job status
2. **Screenshot 2:** Model endpoint configuration
3. **Screenshot 3:** Inference response example
   - Input: `{Age=60, Cholesterol=300, ...}`
   - Output: `Probability=0.68 (High risk)`
4. **Endpoint details:** ARN, latency, instance type

---

## 📊 Evaluation Criteria (100 points)

| Category | Points | Status |
|----------|--------|--------|
| EDA | 10 | ✅ Complete |
| Implementation (sigmoid, cost, GD) | 35 | 🔄 In Progress |
| Visualization & Analysis | 20 | ⏳ Pending |
| Regularization | 15 | ⏳ Pending |
| Deployment & Documentation | 15 | ⏳ Pending |
| Code Clarity & Comments | 5 | 🔄 In Progress |
| **Total** | **100** | **10/100** |

---

## 👨‍💻 Author

**[Your Name]**  
📧 Email: [your.email@example.com]  
🔗 GitHub: [github.com/yourusername](https://github.com/yourusername)

---

## 📜 License

This project is part of a machine learning homework assignment.  
Dataset source: [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/neurocipher/heartdisease) (UCI ML Repository)

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the original dataset
- **Kaggle** for hosting and providing easy access
- **World Health Organization (WHO)** for heart disease statistics
- Course instructors for structured guidance

---

**Last Updated:** January 28, 2026  
**Current Status:** Step 1 Complete ✅ | Step 2 In Progress 🔄
