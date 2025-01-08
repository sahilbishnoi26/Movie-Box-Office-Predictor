# Predicting Box Office Performance from Movie Trailer Comments

- The project aimed to predict the opening week box office revenue of movies using Reddit comments on movie trailer posts prior to their release. Analyzed Reddit comments with BERTopic for topic extraction and a BERT-based Transformer from Hugging Face to generate sentiment scores.

- Reduced dimensions for topic clustering and refinement, achieving a silhouette score of 0.61, indicating well-formed clusters.

- Features such as sentiment scores, topic relevance, genre, Reddit engagement, production scale, and budget were used to train regression models like XGBoost and Random Forest.

- Achieved a mean squared error (MSE) of $5M and an R² of 74%, demonstrating strong predictive performance given that the average opening week box office revenue is around $20M.


Code Files:

`/code/Unsupervised.ipynb`: BERTopic topic modeling implementation

`/code/data_collection.ipynb`: Full data collection & scraping methods for TMDb, OMDb, BoxOffice and Reddit

`/code/data_cleaning_validation.ipynb`: Data cleaning methods (deduplication, removing new comments, etc)

`/code/Data_Prep_for_supervised_model.ipynb`: Additional data cleaning for supervised model

`/code/data_processing_and_merging.ipynb`: Preprocessing data and merging different data sources

'`/code/Random Forest and XGBoost.ipynb`: construction and evaluation of tree-based models

`/code/Sentiment_analysis_scoring.ipynb`: sentiment analysis scoring

`/code/Supervised_model_Linear_Regression.ipynb`: Multiple Linear Regression models

`/code/Supervised_model_MLR.ipynb`: Multiple Linear Regression models

`/code/Sentiment.ipynb`: Sentiment Analysis for unsupervised
