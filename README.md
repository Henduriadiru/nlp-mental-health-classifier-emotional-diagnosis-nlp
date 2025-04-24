# nlp-mental-health-classifier-emotional-diagnosis-nlp
This notebook presents a sentiment analysis project tailored to identifying potential mental health issues based on textual input. Using various NLP techniques and machine learning models, the goal is to detect patterns that could be associated with mental health conditions like depression, anxiety, or PTSD.

# 🧠 Mental Health Diagnosis via Text using NLP

This project applies Natural Language Processing (NLP) and Machine Learning to analyze textual data for signs of mental health conditions such as **depression**, **anxiety**, and **PTSD**. The notebook combines feature engineering, model building, and class balancing techniques to create a diagnostic model based on textual sentiment and semantics.

## 📌 Objectives

- Build a classifier to detect mental health issues based on written text.
- Compare different feature extraction methods: TF-IDF, Word2Vec, and their combination.
- Evaluate multiple ML models (SVM, Random Forest, XGBoost, Adaboost).
- Evaluate Transformer model (Distilbert)
- Apply dimensionality reduction with PCA for better performance and visualization.

## 📁 File Structure
mental-health-text-diagnosis/
│
├── Sentiment_Analysis_Mental_Helath_Diganostic.ipynb       # Classical ML with TF-IDF + Word2Vec
├── sentiment_analysis_mentalhealth_distilbert.ipynb        # Transformer-based model using DistilBERT
├── README.md                                                # Project documentation
└── requirements.txt                                         # (optional) Python dependencies
## 🔧 Techniques Used

- **Preprocessing:** Tokenization, Lemmatization, Stopword Removal
- **Vectorization:** TF-IDF, Word2Vec, and Hybrid
- **ML Models:** SVM, Random Forest, XGBoost, Adaboost, Distilbert-base-uncased
- **Processing Calculate:** LIME
- **Dimensionality Reduction:** PCA
- **Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix
  All techniques is fine tuning!!
  
This repository now includes two approaches to text classification for mental health:

1. **Traditional ML Approach**  
   Using TF-IDF, Word2Vec, and classical models (SVM, Random Forest, XGBoost)  
   → File: `Sentiment_Analysis_Mental_Helath_Diganostic.ipynb`

2. **Transformer-Based Approach**  
   Fine-tuning DistilBERT using Hugging Face Transformers library for more contextual understanding of language.  
   → File: `sentiment_analysis_mentalhealth_distilbert.ipynb`

The DistilBERT model is trained and evaluated using a stratified dataset split, early stopping, and precision-focused evaluation.

**Why DistilBERT?**  
It provides a lighter, faster alternative to BERT with nearly comparable performance — suitable for fine-tuning on moderate hardware (e.g. Google Colab).

---

## 🧠 Model Comparison

| Approach         | Feature Representation     | Strengths                             |
|------------------|----------------------------|----------------------------------------|
| Traditional ML   | TF-IDF, Word2Vec           | Simpler, interpretable, faster to train|
| DistilBERT       | Transformer Embeddings     | Context-aware, handles nuance better   |

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

📜 License
MIT License

🙋‍♂️ Author
Made with ❤️ by [Hendriadi Dwi Nugraha]. Contributions and feedback are welcome!
