# SWYNEX-End-to-End-ML-Project
# Drug Review Sentiment and Side-Effect Mining
# Swynex Internship — Task 4: End-to-End Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

# Project Overview

This project develops an end-to-end Machine Learning pipeline for analyzing drug reviews using Natural Language Processing (NLP) and text classification techniques.

The project uses the *UCI Drug Review dataset*, which contains patient reviews sourced from Drugs.com. It aims to classify the sentiment of drug reviews and explore patient-reported side-effect mentions using text analysis.

This task brings together the complete Machine Learning workflow, from data preprocessing and feature engineering to model training, evaluation, and prediction.

# Objectives

- Load and preprocess the drug review dataset.
- Perform exploratory data analysis (EDA).
- Transform review text into numerical features using TF-IDF.
- Train and optimize suitable Machine Learning classification models.
- Evaluate model performance using standard classification metrics.
- Compare baseline and improved model results.
- Predict sentiment for new, unseen drug reviews.
- Explore potential side-effect mentions in review text.
- Document installation, methodology, results, and usage instructions.

# Dataset

Dataset: Drug Review Dataset

Source: UCI Machine Learning Repository — drug reviews sourced from Drugs.com.

The dataset version used may contain the following columns:

| Column | Description |
| `drugName` | Name of the drug |
| `condition` | Condition associated with the review |
| `review` | Patient-written review |
| `rating` | Numerical rating provided by the reviewer |
| `date` | Date of the review |
| `usefulCount` | Number of users who found the review useful |

Target variable: Sentiment derived from the numerical rating.

For three-class sentiment classification, the following mapping can be used:

- Positive: Rating greater than 6
- Neutral: Rating from 4 to 6
- Negative: Rating less than 4

These labels represent rating-based sentiment categories and do not necessarily capture every emotion expressed in a review.

# Technologies Used

- Python — programming language
- Pandas and NumPy — data manipulation
- Matplotlib and Seaborn — visualization
- Scikit-learn — Machine Learning and evaluation
- TF-IDF Vectorizer** — text feature extraction
- NLTK or regular expressions — optional text preprocessing
- Jupyter Notebook / Google Colab — development environment
- Joblib — saving and loading trained models

 End-to-End Machine Learning Workflow:

# 1. Data Collection and Loading

Load the drug review dataset into a Pandas DataFrame and inspect its structure, column names, and data types.

# 2. Data Preprocessing

- Identify and handle missing values.
- Remove duplicate or invalid records where appropriate.
- Clean review text.
- Create sentiment labels from numerical ratings.
- Inspect the class distribution to identify possible imbalance.

# 3. Exploratory Data Analysis

Explore the dataset to understand:

- Distribution of drug ratings.
- Number of positive, neutral, and negative reviews.
- Review length distribution.
- Frequently occurring words and phrases.
- Relationships between sentiment labels and review characteristics.

# 4. Feature Engineering

Use TF-IDF (Term Frequency–Inverse Document Frequency)** to transform textual reviews into numerical feature vectors.

TF-IDF gives greater importance to informative words while reducing the relative influence of words that occur frequently across many reviews.

The vectorizer must be fitted on the training data only and then applied to validation and test data to avoid data leakage.

# 5. Model Training

Train and compare suitable Machine Learning models, such as:

- Logistic Regression: A linear classification algorithm that works effectively with high-dimensional text features.
- Multinomial Naive Bayes: A probabilistic model commonly used for text classification.

If Task 3 identified a better-performing model, use that model as the primary candidate for the final pipeline.

# 6. Model Optimization and Validation

Apply suitable hyperparameter tuning using techniques such as GridSearchCV or RandomizedSearchCV.

Use cross-validation on the training data to select model parameters. Keep the test set separate until final evaluation.

# 7. Model Evaluation

Evaluate the final model using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Macro-averaged F1-score is useful when sentiment classes are imbalanced because it gives equal weight to each class.

# 8. Prediction on New Reviews

Use the trained pipeline to classify new drug reviews into the defined sentiment categories.

The saved pipeline should contain both the TF-IDF vectorizer and the trained classifier so the same preprocessing and feature extraction steps are applied consistently during prediction.

# 9. Side-Effect Mining

Explore patient review text for possible side-effect mentions using keyword matching, a predefined adverse-effect vocabulary, or an appropriate NLP extraction method.

This component is exploratory. Identifying a symptom in a review does not establish that the medication caused it.

# 10. Model Saving and Reusability

Save the final fitted pipeline using Joblib so it can be loaded later to predict the sentiment of new reviews without retraining the model.

# Project Architecture

Drug Review Dataset
        |
        v
Data Loading and Inspection
        |
        v
Data Cleaning and Preprocessing
        |
        v
Sentiment Label Generation
        |
        v
Train / Test Split
        |
        v
TF-IDF Feature Extraction
        |
        v
Model Training and Hyperparameter Tuning
        |
        v
Model Evaluation and Comparison
        |
        v
Final Trained Pipeline
        |
        v
New Review Sentiment Prediction
        |
        v
Exploratory Side-Effect Mining


# Model Evaluation and Results

The final model should be evaluated on an independent test set that was not used for model selection or hyperparameter tuning.

# Model Comparison

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-score (Macro) |
| Logistic Regression | To be measured | To be measured | To be measured | To be measured |
| Multinomial Naive Bayes | To be measured | To be measured | To be measured | To be measured |
| Optimized Selected Model | To be measured | To be measured | To be measured | To be measured |

*Replace the placeholders with the actual values produced by your executed notebook. Include only models that you trained and evaluated.*

# Result Analysis

The final analysis should explain:

- Which model achieved the best test-set performance.
- Whether hyperparameter tuning improved the selected model.
- How precision, recall, and F1-score compare across models.
- Which sentiment classes were most frequently misclassified.
- Whether the final model generalizes well to unseen reviews.

Do not assume that model optimization will always improve test performance. Report the measured results, including any trade-offs.

# Example Prediction

Once the saved pipeline is available, a new review can be classified as follows:

```python
import joblib

# Load the saved fitted pipeline
model = joblib.load("models/drug_sentiment_pipeline.joblib")

# Example patient review
review = [
    "The medication helped improve my symptoms, "
    "but I experienced some unwanted effects."
]

# Predict sentiment
prediction = model.predict(review)

print("Predicted sentiment:", prediction[0])
```

The example demonstrates how to use the saved pipeline. The predicted output depends on the model trained in the project.

# Repository Structure

```text
Drug-Review-Sentiment-and-Side-Effect-Mining/
│
├── Drug_Review_Sentiment_Analysis.ipynb
├── README.md
├── requirements.txt
│
├── data/
│   └── README.md
│
├── models/
│   └── drug_sentiment_pipeline.joblib
│
├── results/
│   ├── model_comparison.csv
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── sentiment_distribution.png
│
└── src/
    ├── preprocess.py
    ├── train.py
    └── predict.py
```

*This is a suggested structure. Keep only files and folders that actually exist in your repository. Large datasets and model files may be better stored outside GitHub, with download instructions provided instead.*

# Future Improvements

- Build an interactive web application for sentiment prediction.
- Experiment with advanced NLP models such as transformer-based classifiers.
- Improve side-effect extraction using medical entity recognition.
- Evaluate performance across different medical conditions.
- Address class imbalance where necessary.
- Add automated testing and a reproducible training pipeline.

# Disclaimer

This project is intended for educational and research purposes only. The sentiment classifier does not provide medical advice or determine whether a drug is safe or effective. Extracted side-effect mentions are patient-reported information and do not establish causation.


⭐ If you find this project interesting, feel free to explore the repository and share feedback!

