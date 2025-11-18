# 💬 AI Sentiment Analysis Using NLP & Machine Learning Advanced Project
A complete end-to-end sentiment classification project using Twitter US Airline Sentiment Dataset, TF-IDF, multiple ML models, hyperparameter tuning, and advanced RoBERTa-based emotion detection.

📂 Project Overview :- 
This project analyzes airline-related tweets to classify them as positive, negative, or neutral.
It includes a full NLP workflow:
- Cleaning tweets (punctuation removal, lowercasing, stopwords, etc.)
- Vectorizing using TF-IDF with optimal parameters
- Training multiple classification models
- Hyperparameter tuning for best performance
- Building a reusable prediction pipeline
- Adding an advanced emotion analysis model powered by RoBERTa, giving detailed emotional insights with confidence scores and emojis.

🎯 Objectives :-
- Load, clean, and preprocess raw text data
- Convert text into numerical features using TF-IDF vectorization
- Train and compare multiple ML models (Logistic Regression, Naive Bayes, SVM, etc.)
- Perform hyperparameter tuning to achieve the best accuracy
- Evaluate models using accuracy, classification report, and confusion matrix
- Build a custom sentiment prediction function
- Perform advanced emotion detection using RoBERTa (GoEmotions)

🧰 Tech Stack & Libraries :-
- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib / Seaborn
- Transformers (HuggingFace)
- RoBERTa (GoEmotions model)

🧮 Key Steps in the Workflow
1️⃣ Load Dataset
- Twitter US Airline Sentiment dataset (Kaggle)
- Load CSV → inspect shape and structure

2️⃣ Data Cleaning
- Remove punctuation
- Lowercase text
- Remove stopwords
- Tokenization and text normalization

3️⃣ Text Vectorization (TF-IDF)
- Convert tweets into TF-IDF vectors
- Tune parameters such as:
- max_features
- ngram_range
- min_df

4️⃣ Train-Test Split
- 80/20 split for training and evaluation

5️⃣ Model Comparison
- Trained multiple models:
- Logistic Regression
- Multinomial Naive Bayes
- Random Forest
- SVM (LinearSVC)
- KNN
  
Each model evaluated for:
- Accuracy
- Precision, Recall, F1-score
- Overall performance ranking

6️⃣ Hyperparameter Tuning (Best Model: SVM)
- Used GridSearchCV to tune:
- Regularization parameters
- Kernel optimization
- Loss functions

7️⃣ Model Comparison Summary
- SVM achieved highest accuracy (≈ 95%)
- Naive Bayes performed well for speed
- Logistic Regression showed strong baseline performance

8️⃣ Evaluation Report
- Classification report for each sentiment class
- Confusion matrix for detailed error analysis

9️⃣ Custom Prediction Function
- Enter a sentence → get predicted sentiment
- Automatically applies TF-IDF & trained model
- Includes 10 pre-loaded example predictions

🔟 Advanced Emotion Analysis (RoBERTa — GoEmotions)
Includes:
- Deep emotion detection
- 27 emotion classes (e.g., joy, anger, love, worry)
- Confidence scores
- Emoji-enhanced interpretation

📊 Major Insights :-
📌 1. SVM achieves the best sentiment classification accuracy
📌 2. TF-IDF + Linear models outperform tree-based models
📌 3. Negative tweets dominate airline-related feedback
📌 4. RoBERTa reveals richer emotions beyond sentiment polarity

🗂️ Project Structure
├── Sentimental Analysis Advanced ML Project.ipynb
├── dataset/
│   └── Twitter US Airline Sentiment Dataset.csv
├── README.md

👨‍💻 Developed By
-- Ayush
Data Science & Analytics | Machine Learning | NLP | Web Scraping & APIs
- 🔗 GitHub: https://github.com/ayush13-0
- 🔗 LinkedIn: https://www.linkedin.com/in/ayush130

📜 License
- This project is licensed under the **MIT License**.

