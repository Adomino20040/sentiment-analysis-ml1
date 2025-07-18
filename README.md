# Twitter Sentiment Analysis using Machine Learning

## Project Overview

This project implements a machine learning pipeline to classify the sentiment of Twitter messages related to U.S. airlines. The goal is to automatically determine whether a tweet expresses a positive, neutral, or negative sentiment toward an airline. Sentiment analysis like this is a key task in natural language processing (NLP) with real-world applications in customer feedback analysis, brand monitoring, and social media insights.

## Motivation
With the vast amount of user-generated content on social media platforms like Twitter, companies need scalable automated tools to understand public opinion and respond accordingly. Manual analysis is impractical due to data volume and velocity. This project aims to build a reliable model that can help airlines (or any brand) monitor customer sentiment and improve their services based on real-time feedback.

## Dataset
The dataset consists of approximately 14,640 tweets mentioning U.S. airlines, each labeled with one of three sentiment classes:

positive

neutral

negative

Along with the tweet text, the dataset contains metadata such as tweet ID, airline name, and confidence scores for labels.

## Methodology
### 1. Data Preprocessing
Data Cleaning:
Tweets were cleaned to remove noise such as URLs, mentions (@username), hashtags, punctuation, special characters, and extra whitespace. Text was also lowercased to maintain uniformity.

Stopword Removal:
Common English stopwords (e.g., “the”, “is”, “and”) were removed to reduce noise and improve model focus on meaningful words.

### 2. Feature Extraction
TF-IDF Vectorization:
The cleaned tweets were transformed into numeric feature vectors using Term Frequency-Inverse Document Frequency (TF-IDF). This technique weighs words by how important they are to a document relative to the entire corpus, highlighting discriminative terms.

Parameter Tuning:
To optimize the feature set, the maximum number of features (max_features) was tuned, with 2000 features providing the best performance balance.

### 3. Label Encoding
The sentiment labels were converted from categorical strings to integer labels (0, 1, 2) using label encoding to prepare for supervised classification.

### 4. Model Training and Evaluation
Train-Test Split:
The dataset was split into 80% training and 20% testing subsets to evaluate model generalization.

Classifier:
Logistic Regression was chosen as the baseline model due to its effectiveness and interpretability for text classification tasks.

Performance Metrics:
The model was evaluated using accuracy, precision, recall, and F1-score for each sentiment class.

## Results
The final model achieved an accuracy of approximately 80% on the test set. Precision and recall were highest for the negative sentiment class, indicating strong ability to detect complaints or problems. The neutral class showed lower recall, reflecting the inherent difficulty in classifying more ambiguous sentiments.

Sentiment	Precision	Recall	F1-Score
Negative	0.82	0.94	0.88
Neutral	0.67	0.50	0.57
Positive	0.78	0.60	0.68

## Future Work and Applications
### Model Improvements:
Explore more advanced models such as Support Vector Machines (SVM), Random Forests, or deep learning architectures like LSTM or Transformer-based models (e.g., BERT) to capture contextual nuances better.

### Data Augmentation and Balancing:
Techniques like oversampling underrepresented classes or gathering more balanced data can improve model fairness and recall, especially for the neutral class.

### Real-Time Deployment:
Deploying the model as an API or integration with social media platforms can provide live sentiment monitoring dashboards for airline customer service teams.

### Broader Applications:
The methodology extends to any domain where understanding user sentiment is critical, such as product reviews, political analysis, or brand reputation management.

## How to Use This Project
Clone the repository.

Install dependencies (e.g., pandas, scikit-learn).

Run the notebook or script to preprocess data, train the model, and evaluate performance.

Use the provided prediction function to classify new tweets.

Technologies Used
Python 3.x

Pandas

scikit-learn

Jupyter Notebook / Google Colab

## Acknowledgments
This project was inspired by the practical needs of sentiment analysis and the availability of open Twitter airline sentiment datasets. Special thanks to open-source contributors whose libraries made this work possible.

## Contact
Feel free to reach out for questions or collaboration opportunities!
