# lending-club-loan-data-analysis
Lending Club loan analysis and default prediction using Python and deep learning.
# Lending Club Loan Data Analysis & Default Prediction

## 📌 Project Overview

This project analyzes Lending Club loan data and develops a deep learning classification model to predict whether a loan is fully paid or not fully paid.

The project includes data inspection, exploratory data analysis, preprocessing, class balancing using SMOTE, feature scaling, Artificial Neural Network development, and model evaluation.

## 🎯 Objectives

- Analyze Lending Club loan data.
- Understand borrower and loan characteristics.
- Explore factors related to loan repayment.
- Handle class imbalance.
- Prepare data for deep learning.
- Build an Artificial Neural Network for loan classification.
- Evaluate the model using classification metrics and ROC-AUC.

## 📊 Dataset

The dataset contains information about Lending Club loans and borrower characteristics.

Important features include:

- `credit.policy`
- `purpose`
- `int.rate`
- `installment`
- `log.annual.inc`
- `dti`
- `fico`
- `days.with.cr.line`
- `revol.bal`
- `revol.util`
- `inq.last.6mths`
- `delinq.2yrs`
- `pub.rec`

The target variable is:

- `not.fully.paid = 0` → Loan fully paid
- `not.fully.paid = 1` → Loan not fully paid

## 🔍 Exploratory Data Analysis

The project explores:

- Loan purpose distribution
- Borrower financial characteristics
- FICO scores
- Debt-to-income ratio (DTI)
- Interest rates
- Revolving balance and utilization
- Loan repayment/default patterns

A loan-purpose distribution visualization was created using Seaborn. 

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Inspected the dataset structure.
2. Checked for missing values.
3. Prepared features and target variables.
4. Split the data into training and testing sets.
5. Applied SMOTE to address class imbalance.
6. Applied feature scaling using `StandardScaler`.
7. Prepared the processed data for the neural network.

SMOTE was applied only to the training data using `fit_resample()`. 

## 🧠 Deep Learning Model

An Artificial Neural Network (ANN) was developed using TensorFlow/Keras.

### Model Architecture

```text
Input
  ↓
Dense (64 neurons, ReLU)
  ↓
Dense (32 neurons, ReLU)
  ↓
Dropout (0.3)
  ↓
Dense (16 neurons, ReLU)
  ↓
Dense (1 neuron, Sigmoid)
