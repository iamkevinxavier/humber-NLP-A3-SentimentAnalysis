# NLP - Assignment 3 - Sentiment Analysis - Vader

## Project Overview

This project focuses on performing sentiment analysis on Amazon product reviews using two different machine learning classifiers: Logistic Regression and KNN Classifier. The project also compares the performance of these classifiers with VADER (Valence Aware Dictionary and sEntiment Reasoner), a lexicon-based sentiment analysis tool.

## Dataset

The project utilizes the `amazonreviews.tsv` dataset, which contains a collection of Amazon product reviews labeled as either "pos" (positive) or "neg" (negative). The dataset is loaded and preprocessed to handle any missing values.

## Methodology

1. **Data Loading and Preprocessing:** The `amazonreviews.tsv` dataset is read into a pandas DataFrame, and any missing values are handled.

2. **Sentiment Analysis using Machine Learning:**
   - The dataset is split into training and testing sets using `train_test_split`.
   - Two classification pipelines are created: one using Logistic Regression and the other using KNN Classifier.
   - Both pipelines utilize TF-IDF vectorization to convert text data into numerical features.
   - The models are trained on the training data and evaluated on the testing data.

3. **Sentiment Analysis using VADER:**
   - The VADER sentiment analysis tool is used to analyze the sentiment of random reviews.
   - VADER provides sentiment scores (positive, negative, neutral, and compound) for each review.
   - The compound score is used to classify the overall sentiment as positive, negative, or neutral.

4. **Evaluation and Comparison:**
   - The performance of Logistic Regression, KNN Classifier, and VADER is compared based on various metrics such as accuracy, precision, recall, and F1-score.
   - A confusion matrix is generated for each classifier to visualize the classification results.

## Results

- **Logistic Regression:** Achieved higher accuracy and more balanced precision and recall compared to KNN Classifier.

- **KNN Classifier:** Performed decently in identifying negative sentiment but had lower accuracy and precision in identifying positive sentiment.

- **VADER:** Provided detailed sentiment scores, including individual sentiment categories and an overall compound score.

## Conclusion

Based on the analysis, Logistic Regression is a more appropriate choice for sentiment classification in this specific scenario. VADER offers a comprehensive sentiment analysis approach, providing valuable insights into the sentiment distribution of text.


## Usage

1. Upload the 'amazonreviews.tsv' to your Colab environment.
2. Install the necessary libraries: `!pip install vaderSentiment`.
3. Execute the code cells in the notebook sequentially.
4. You can modify the random reviews in the `random_reviews` list to test the performance of the models on different text samples.
