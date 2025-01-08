---
layout: default
title: Predicting Box Office Performance from Movie Trailer Comments
description: Midterm Checkpoint
---

# Introduction and Background

In the film industry, word of mouth, defined as “informal communication among consumers about products and services,” plays a significant role in consumers’ decision-making process. Because the power of word of mouth is often cited as a key factor in the success of certain movies, movie studios and distributors have a clear incentive to understand what consumers are saying [1]. Fortunately, user-generated online text or “electronic word of mouth” has proliferated in recent decades, representing a rich dataset for understanding consumer sentiment [2, 3]. 

Analyses of user-generated content for movies appears in the literature, such as a Naïve Bayes sentiment analysis conducted by Novendri and colleagues [4] or box office success predictions based on movie reviews [2]. Our project seeks to build on these analyses by analyzing comments on trailers posted to the r/movies subreddit to predict movie success, defined as money made during a movie’s opening weekend. To do so, we will use a custom dataset from the APIs of Reddit, TMDb, and OMDb containing movie metadata and user comments on trailers.

# Problem Definition

For both movie creators and viewers, differentiating one film from the massive amount released each year presents a challenge. From a studio’s perspective, determining which user sentiments predict financial success and / or attention could lead to a more optimized or profitable movie promotion strategy. *Meanwhile, from a moviegoer’s perspective, a model that sorts through public comments on movies and clusters them into topic could give an idea of what the notable aspects of a movie’s trailer might be.*

# Unsupervised Model Implementation

To gain insight into the general landscape of conversations that occur on r/movies, we decided to implement an unsupervised clustering method known as topic modeling. By understanding what conversation topics are occuring on social media, as well as which of them are correlated with high performing movies, we may be able to extract useful features that could help in our predictions. Text already presents unique challenges in modeling, and social media likely adds even more complexity. As a medium, social media text is almost exclusively short and unstructured. In addition, since so many different people are contributing text, there is naturally more variation in the vocabulary than if there was a single writer. 

A key resource for this step was a [study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9120935/) [5] by Egger and Yu in which they evaluated and compared four topic modeling techniques: latent Dirichlet allocation (LDA), non-negative matrix factorization (NMF), Top2Vec, and BERTopic. Their research was applied to Twitter posts, but we felt that this should still be applicable to another text based social media site like Reddit. Based on the suggestions from the paper and our own experimentation, we decided to implement BERTopic. 

## BERTopic

> BERTopic is a topic modeling technique that leverages 🤗 transformers and c-TF-IDF to create dense clusters allowing for easily interpretable topics whilst keeping important words in the topic descriptions. [Documentation](https://maartengr.github.io/BERTopic/)

### Text Preprocessing Methods

Since BERTopic relies on text embeddings, it is best to keep the structure of the comments as close to their original form as possible. This allowed us to skip many of the usual NLP preprocessing methods and minimize the time spent on text cleaning. However, we did need to remove duplicate comments, remove deleted comments, remove links and remove any other non-word text. After that, the only preprocessing needed is to create the embeddings. 

### The Algorithm

![BERTopic](/BERTopic.jpeg)

1. Embed documents using sentence-transformers
2. Dimensionality reduction using UMAP
3. Clustering comments using HDBScan
4. CountVectorizer to determine word frequencies within clusters
5. Class-based TF-IDF to determine words most important to topic

Each of these steps has its own benefits which are worth mentioning. The big benefit of embeddings is the ability to identify semantic similarity. For example, our topic model would be able to determine that "song" and "music" have more similar meanings than "song" and "graphics". These representations are far too large and require dimensionality reduction. Using UMAP can preserve some local and global structure of the comments dataset. HDBScan doesn't require you to predetermine the number of clusters, and is able to allocate comments to clusters without requiring that every comment be part of a group. This allows for outliers and will improve overall coherence of the topics. Finally, by defining our clusters in terms of the most important words using TF-IDF, we can immediately extract topics. 

# Results and Discussion

Overall, we were happy with the results of the model. While there is certainly more fine-tuning that could be done, these preliminary results are promising. Our model was able to achielve a silhouette score of 0.61 which indicates the clustering is reasonably good. 

We were also able to visualize the clusters using the BERTopic package:

![Topic_Clustering](/topic_clustering.png)

A visualization of topics. Note that we reduced the clustering to 2 dimensions for visualization purposes, so clusters may not be as on top of each other as they appear here. Overall, we are happy with the separation.
![Topic_Words](/topic_words.png)

A sampling of words in some of our clusters. As you can see some topics are similar (i.e. actor and actress), and some are quite general (i.e. movie, movies), but others could be indicative of topics that are important to audiences.

![topic_heatmap](/topic_heatmap.png)

It appears none of the topics are overly similar to each other.

**Next Steps**\
There are a few steps that we'd like to take next to improve the performance. First, there are methods to reduce the number of outlier comments. Right now, nearly half of the comments are considered outliers. While this could be accurate, it may be more helpful to merge those into the closest cluster. Also, we'd like to analyze which movies these topics take place on and how well those movies go on to perform in the theaters. If there are certain topics that predict a movie will perform well, we may want to include it somehow in our supervised model. 

### [View Gantt Chart](gantt_chart.png)

### Contribution Table

|    Name     |   Proposal Contributions     |
| ----------- | ----------- | 
| Nathan Popper      | Unsupervised Model Construction, Preprocessing, and Writeup      | 
| Chandler Schneider      | Writeup Assistance, Supervised Learning    |
| Oliver Hewett   | Focused on Supervised Learning Model during this time        | 
| Sahil Bishnoi   | Focused on Supervised Learning Model during this time       | 
| Rishidhar Duvvuru   | Focused on Supervised Learning Model during this time        | 

**Note:** As a group we divided labor in such a way that it made sense for one person to be chiefly in charge of this midterm progress report while other group members focused on models that will be presented in our final writeup.

# References

[1] Y. Liu, “Word of Mouth for Movies: Its Dynamics and Impact on Box Office Revenue,” _Journal of Marketing_, vol. 70, no. 3, pp. 74-89, 2006. [Online]. Available: Sage Journals, www.journals.sagepub.com. [Accessed Feb. 18, 2024].  

[2] Y. Kim, M. Kang, S.R. Jeong, “Text Mining and Sentiment Analysis for Predicting Box Office Success”, _KSII Transactions on Internet and Information Systems_, vol. 12, no. 8, pp. 4090-4102, 2018. [Online]. Available www.itiis.org. [Accessed Feb. 16, 2024].  

[3] S. Sun, C. Luo, and J. Chen, “A review of natural language processing techniques for opinion mining systems,” _Information Fusion_, vol. 36, pp. 10-25, 2018.. [Online]. Available: ScienceDirect, www.sciencedict.com. [Accessed Feb. 18, 2024].  

[4] R.  Novendri, A.S. Callista, D.N. Pratama, and E.E. Puspita, “Sentiment Analysis of YouTube Movie Trailer Comments Using Naïve Bayes,” _Bulletin of Computer Science and Electrical Engineering_, vol. 1, no. 1 , pp. 26-32, 2020. [Online]. Available: www.bscee.org. [Accessed Feb. 18, 2024].

[5] Egger R, Yu J. A Topic Modeling Comparison Between LDA, NMF, Top2Vec, and BERTopic to Demystify Twitter Posts. Front Sociol. 2022 May 6;7:886498. doi: 10.3389/fsoc.2022.886498. PMID: 35602001; PMCID: PMC9120935.

[Home](./)
