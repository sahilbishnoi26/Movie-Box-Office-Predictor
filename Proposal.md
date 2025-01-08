---
layout: default
title: Predicting Box Office Performance from Movie Trailer Comments
description: Project Proposal
---
# Proposal Video:

[![YouTube Video](https://img.youtube.com/vi/oifzktR09SY/0.jpg)](https://www.youtube.com/watch?v=oifzktR09SY)

# Introduction and Background

In the film industry, word of mouth, defined as “informal communication among consumers about products and services,” plays a significant role in consumers’ decision-making process. Because the power of word of mouth is often cited as a key factor in the success of certain movies, movie studios and distributors have a clear incentive to understand what consumers are saying [1]. Fortunately, user-generated online text or “electronic word of mouth” has proliferated in recent decades, representing a rich dataset for understanding consumer sentiment [2, 3]. 

Analyses of user-generated content for movies appears in the literature, such as a Naïve Bayes sentiment analysis conducted by Novendri and colleagues [4] or box office success predictions based on movie reviews [2]. Our project seeks to build on these analyses by analyzing comments on trailers posted to the r/movies subreddit to predict movie success, defined as money made during a movie’s opening weekend. To do so, we will use a custom dataset from the APIs of Reddit, TMDb, and OMDb containing movie metadata and user comments on trailers.

# Problem Definition

For both movie creators and viewers, differentiating one film from the massive amount released each year presents a challenge. From a studio’s perspective, determining which user sentiments predict financial success and / or attention could lead to a more optimized or profitable movie promotion strategy. Meanwhile, from a moviegoer’s perspective, a model that sorts through public comments on movies and clusters them into topic could give an idea of what the notable aspects of a movie’s trailer might be.

# Methods

Since it involves unstructured text data, preprocessing will be key to this project’s success. In addition to general data cleaning and matching across datasets, preprocessing steps will include standard NLP techniques such as case folding, tokenization, stemming and stopword removal [3, 4].  Additionally, TF-IDF will be used to measure relative word importance [3,4]. 

Once the data has been processed, our unsupervised learning will center around LDA and NMF since they are the industry standard for this type of project. Additionally, we will attempt to build both classification and regression models to determine the predictive power of trailer comments on Reddit when controlling for factors such as genre. Our classification will begin with Naïve Bayes for its simplicity and move upwards in complexity as necessary. For regression, we plan to use a random forest model with classifications drawn from our clustering and classification models to predict box office performance and critical and viewer response.  


# Potential Results

To measure our model performance, we will use the silhouette coefficient for our clustering model, F1 score for classification, and BIC for our regression. The project’s end goal is a model that uses trailer sentiments along with other features to predict movie performance during release weekend.

### [View Gantt Chart](gantt_chart.png)

### Contribution Table

|    Name     |   Proposal Contributions     |
| ----------- | ----------- | 
| Nathan Popper      | Slides, GitHub Pages, Dataset Research       | 
| Chandler Schneider      | Slides, GitHub Pages, Proposal Video, Proposal Writeup      |
| Oliver Hewett   | Dataset Research        | 
| Sahil Bishnoi   | Model Research       | 
| Rishidhar Duvvuru   | Dataset and Model Research        | 

# References

[1] Y. Liu, “Word of Mouth for Movies: Its Dynamics and Impact on Box Office Revenue,” _Journal of Marketing_, vol. 70, no. 3, pp. 74-89, 2006. [Online]. Available: Sage Journals, www.journals.sagepub.com. [Accessed Feb. 18, 2024].  

[2] Y. Kim, M. Kang, S.R. Jeong, “Text Mining and Sentiment Analysis for Predicting Box Office Success”, _KSII Transactions on Internet and Information Systems_, vol. 12, no. 8, pp. 4090-4102, 2018. [Online]. Available www.itiis.org. [Accessed Feb. 16, 2024].  

[3] S. Sun, C. Luo, and J. Chen, “A review of natural language processing techniques for opinion mining systems,” _Information Fusion_, vol. 36, pp. 10-25, 2018.. [Online]. Available: ScienceDirect, www.sciencedict.com. [Accessed Feb. 18, 2024].  

[4] R.  Novendri, A.S. Callista, D.N. Pratama, and E.E. Puspita, “Sentiment Analysis of YouTube Movie Trailer Comments Using Naïve Bayes,” _Bulletin of Computer Science and Electrical Engineering_, vol. 1, no. 1 , pp. 26-32, 2020. [Online]. Available: www.bscee.org. [Accessed Feb. 18, 2024].

[Home](./)
