# 📰 Fake News Detection System Using Machine Learning and NLP

<p align="center">
  <b>An NLP-powered machine learning system for classifying news articles as potentially fake or real.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikitlearn" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/NLP-TF--IDF-green" alt="NLP">
  <img src="https://img.shields.io/badge/Interface-Streamlit-red?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Status-Research%20Project-purple" alt="Research Project">
</p>

---

## 📌 Overview

The **Fake News Detection System** is a machine learning project that analyzes the textual content of news articles and predicts whether an article resembles fake-news or real-news examples learned during training.

The project combines Natural Language Processing (NLP), TF-IDF feature extraction, and supervised machine learning algorithms to transform news text into numerical features and classify it automatically.

A Streamlit-based web interface allows users to enter a news headline and article text and obtain a model prediction.

> **Important:** This is a text-classification system, not an independent fact-checking engine. Its predictions do not establish whether a news claim is factually true or false.

## ✨ Key Features

- **Text preprocessing:** Prepares news text for machine learning.
- **TF-IDF vectorization:** Converts text into numerical feature vectors.
- **Model comparison:** Evaluates Logistic Regression, Linear Support Vector Machine, and Multinomial Naive Bayes.
- **News classification:** Uses a trained classifier to predict a news category.
- **Interactive interface:** Provides a Streamlit web application.
- **Model explainability:** Includes a script for examining influential text features.
- **Performance analysis:** Stores experiment results and visualizations for model evaluation.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Natural Language Processing | TF-IDF |
| Machine learning | Scikit-learn |
| Classification algorithms | Logistic Regression, Linear SVM, Multinomial Naive Bayes |
| Visualization | Matplotlib, Seaborn |
| Web application | Streamlit |
| Model serialization | Joblib / Pickle |

## 🧠 Machine Learning Workflow

```text
News Article Dataset
        │
        ▼
Data Inspection and Preprocessing
        │
        ▼
Text Preparation
        │
        ▼
TF-IDF Feature Extraction
        │
        ▼
Train and Evaluate Models
        │
        ├── Logistic Regression
        ├── Linear SVM
        └── Multinomial Naive Bayes
        │
        ▼
Model Comparison and Selection
        │
        ▼
Saved Model and TF-IDF Vectorizer
        │
        ▼
Streamlit Prediction Interface
        │
        ▼
Potentially Fake / Potentially Real
```

## 📊 Experimental Results

The following results were obtained during the project's model-comparison experiment.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 98.99% | 98.46% | 99.69% | 99.07% |
| Linear SVM | **99.59%** | **99.41%** | **99.83%** | **99.62%** |
| Multinomial Naive Bayes | 96.24% | 96.55% | 96.51% | 96.53% |

**Best-performing model in this experiment: Linear SVM**

The Linear SVM achieved the highest accuracy and F1-score among the three evaluated models on the project's test split.

These are dataset-specific experimental results, not a guarantee of performance on unseen news sources or real-world misinformation. A random train/test split can preserve source and writing-style patterns, so further testing on independent datasets and time-based or source-based splits is necessary.

## 📁 Project Structure

```text
fake-news-detection-system/
│
├── app/
│   ├── app.py
│   └── test_streamlit.py
│
├── data/
│   ├── raw/                 # Excluded from GitHub
│   └── processed/           # Excluded from GitHub
│
├── figures/
│   ├── fig1_news_distribution.png
│   ├── fig2_text_length_distribution.png
│   ├── fig3_model_performance_comparison.png
│   ├── fig4_linear_svm_confusion_matrix.png
│   └── fig4_training_time_comparison.png
│
├── models/
│   ├── linear_svm_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── multinomial_naive_bayes_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── tfidf_vectorizer_compare.pkl
│
├── notebooks/
│   └── test_setup.py
│
├── results/
│   ├── class_distribution.png
│   ├── model_comparison.csv
│   ├── text_length_distribution.png
│   └── title_length_distribution.png
│
├── src/
│   ├── eda.py
│   ├── explain_model.py
│   ├── inspect_dataset.py
│   ├── predict.py
│   ├── preprocess_dataset.py
│   ├── train_baseline.py
│   └── train_compare.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## 🚀 Installation and Setup

### Prerequisites

- Python installed on your system
- Git
- A terminal such as PowerShell or Command Prompt

### 1. Clone the repository

```bash
git clone https://github.com/VuppalapatiVedesh/fake-news-detection-system.git
cd fake-news-detection-system
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, use the Python executable directly or follow the appropriate Python environment activation guidance for your system.

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Launch the web application

```bash
streamlit run app/app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the application.

**Note:** The saved model files and matching TF-IDF vectorizer must be present at the paths expected by `app/app.py`. The raw dataset is not required for prediction if the saved model artifacts are available.

## 🖥️ Using the Application

1. Launch the Streamlit application.
2. Enter a news headline.
3. Paste the article text.
4. Submit the article for prediction.
5. Review the predicted category and any explanation provided by the interface.

The result reflects learned textual patterns; it should not be interpreted as proof that a story is true or false.

## 📈 Evaluation and Visualization

The repository includes experiment outputs and plots that support analysis of:

- News class distribution
- Text-length distribution
- Model performance comparison
- Linear SVM confusion matrix
- Training-time comparison

These visualizations can be used when discussing the experimental methodology and results in a project demonstration or research report.

## ⚠️ Limitations and Future Work

### Current limitations

- The classifier learns patterns in its training dataset rather than verifying claims against independent evidence.
- Dataset-specific vocabulary, source characteristics, and writing styles may influence predictions.
- High test-set performance may not generalize to new publishers, topics, or future events.
- The system may misclassify satire, opinion pieces, breaking news, and articles with mixed or incomplete claims.
- Model scores should not be interpreted as calibrated probabilities unless calibration has been evaluated.

### Potential improvements

- Evaluate on independent datasets from different sources.
- Use source-based and time-based train/test splits.
- Explore transformer-based language models.
- Add claim extraction and retrieval of evidence from reliable sources.
- Develop explainability and uncertainty reporting.
- Measure robustness to paraphrasing, new topics, and distribution shifts.

## 🔬 Reproducibility

For reproducible experiments, record the dataset version, preprocessing steps, random seed, train/test split, feature-extraction settings, model parameters, and package versions. The results in this repository should be interpreted in the context of the experiment that generated them.

## 🔒 Dataset and Model Files

Raw and processed datasets are excluded from the repository to keep it lightweight. Users must obtain the dataset from an appropriate source and follow its terms of use if they wish to reproduce training.

The trained model artifacts are included for inference, subject to the applicable dataset and artifact-use terms.

## ⚖️ Disclaimer

This project is intended for educational and research purposes. It is not a substitute for professional journalism, independent fact-checking, or verification using trustworthy primary sources.

A model prediction alone should never be used to label an individual, publisher, or news organization as deceptive.

## 👨‍💻 Author

**Vedesh Vuppalapati**

GitHub: [@VuppalapatiVedesh](https://github.com/VuppalapatiVedesh)

---

If you find this project useful, consider starring the repository ⭐ and sharing constructive feedback.

