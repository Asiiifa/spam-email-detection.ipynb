# 📩 Spam Email Detection using Machine Learning

This project focuses on **detecting spam emails** using natural language processing (NLP) and machine learning techniques. We preprocess the data, balance the classes using **SMOTE**, and train a **Naive Bayes classifier** to distinguish between **spam** and **ham (not spam)** messages.

## 🎯 Objective

Build a machine learning model that can accurately classify SMS messages as **spam** or **ham**.

## 📚 Dataset

- **Dataset name**: `spam.csv`
- **Columns**:
  - `v1`: Label (`spam` or `ham`)
  - `v2`: Message text
- The dataset contains some unnecessary unnamed columns that are dropped during preprocessing.

## 🛠️ Project Workflow

1. **Data Loading**:
   - Read the CSV file using pandas.
   - Drop irrelevant columns.

2. **Exploratory Data Analysis (EDA)**:
   - Visualize the class distribution using pie charts and count plots.

3. **Feature Extraction**:
   - Convert text data into numerical vectors using `CountVectorizer`.

4. **Handling Imbalanced Data**:
   - Use **SMOTE** (Synthetic Minority Over-sampling Technique) to balance spam and ham classes.

5. **Label Encoding**:
   - Encode the target labels into numerical values (spam → 1, ham → 0).

6. **Splitting the Data**:
   - Train-test split (80% training, 20% testing).

7. **Model Training**:
   - Train a **Multinomial Naive Bayes** classifier.

8. **Model Evaluation**:
   - Evaluate the model using **Accuracy**, **Confusion Matrix**, **Classification Report** (Precision, Recall, F1-score).

9. **Visualization**:
   - Confusion matrix heatmap using seaborn.

## 🚀 How to Run

1. Install necessary libraries:

```bash
pip install pandas seaborn scikit-learn imbalanced-learn matplotlib
