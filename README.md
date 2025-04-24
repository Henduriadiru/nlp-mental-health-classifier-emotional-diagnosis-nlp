# nlp-mental-health-classifier-emotional-diagnosis-nlp
This notebook presents a sentiment analysis project tailored to identifying potential mental health issues based on textual input. Using various NLP techniques and machine learning models, the goal is to detect patterns that could be associated with mental health conditions like depression, anxiety, or PTSD.

# 🧠 Mental Health Diagnosis via Text using NLP

This project applies Natural Language Processing (NLP) and Machine Learning to analyze textual data for signs of mental health conditions such as **depression**, **anxiety**, and **PTSD**. The notebook combines feature engineering, model building, and class balancing techniques to create a diagnostic model based on textual sentiment and semantics.

## 📌 Objectives

- Build a classifier to detect mental health issues based on written text.
- Compare different feature extraction methods: TF-IDF, Word2Vec, and their combination.
- Evaluate multiple ML models (SVM, Random Forest, XGBoost, Adaboost).
- Apply dimensionality reduction with PCA for better performance and visualization.

## 📁 File Structure

## 🔧 Techniques Used

- **Preprocessing:** Tokenization, Lemmatization, Stopword Removal
- **Vectorization:** TF-IDF, Word2Vec, and Hybrid
- **ML Models:** SVM, Random Forest, XGBoost
- **Processing Calculate:** LIME
- **Dimensionality Reduction:** PCA
- **Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix

## 📊 Results

Evaluation metrics suggest that combining Word2Vec with TF-IDF yields better results for complex emotional understanding. SMOTE improves detection for minority classes such as PTSD.

## 📦 Requirements

` ⚙️ Requirements
- Python 3.8+
- scikit-learn
- nltk
- gensim
- imbalanced-learn
- matplotlib, seaborn
- xgboost
