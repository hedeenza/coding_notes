# Machine Learning in 2024 - Beginner's Course
https://www.youtube.com/watch?v=bmmQA8A-yUA
~ 4.5 Hours

Contents
00:00:00 Introduction
00:03:13 Machine Learning Roadmap for 2024
00:10:39 Must Have Skill Set for Career in Machine Learning
00:38:54 Machine Learning Common Career Paths
00:45:48 Machine Learning Basics
01:00:59 Bias-Variance Trade-Off
01:08:04 Overfitting and Regularization
01:23:38 Linear Regression Basics - Statistical Version
01:36:56 Linear Regression Model Theory
02:00:20 Logistic Regression Model Theory
02:15:37 Case Study with Linear Regression
02:33:44 Loading and Exploring Data
02:39:54 Defining Independent and Dependent Variables
02:45:59 Data Cleaning and Preprocessing 
02:54:39 Descriptive Statistics and Data Visualization
03:03:39 InterQuantileRange for Outlier Detection
03:14:00 Correlation Analysis 
03:32:14 Splitting Data into Train/Test with sklearn
03:34:31 Running Linear Regression - Causal Analysis
04:01:24 Checking OLS Assumptions of Linear Regression Model
04:10:10 Running Linear Regression for Predictive Analytics
04:15:54 Closing: Next Steps and Resources

#1 What is machine learning?
a branch of ai that allows you to build models based on data, learn from
that data, and then implement those insights to make decisions.

Math - Linear Algebra
    Vectors
    Matrix
    Do Product
    Matrix Multiplication
    Identity/Diagonal Matrix
    Transposition
    Inverse
    Determinant

Math - Calculus
    Differentiation rules
    integration
    sum rule
    constant rule
    chain rule
    gradients
    hessian

Math - Discrete math 
    complexity ("Big O Notation")

Statistics
    descriptive
        central limit theorem / llm
        sample/population
    inferential
    causal analysis
    multivariate
    probability theory
        conditional probabilities
        bernoulli distribution
        binomial distribution
        uniform distribution
        normal distribtion
        exponential distribution
        expectation/variand oc pdfs
    bayesian theory
        prior probability
        posterior pobability

Machine Learning
    Supervised vs Unsupervised
    Classification / Regression
    Clustering
    Time Series Analysis
    Linear Regression
    Logistic Regression
    LDA
    Models
        KNN
        Decision Trees
        Bagging
        Boosting (LightGBM, GBM)
        Boosting (XGBoost, AdaBoost)
        Random Forest
        K-Means / DBSCAN
        Hierarchical Clustering

Machine Learnin - Training Models
    Training / Validating / Testing
    Hyperparameter Tuning
    Optimization Algorithms
    GD, SGD, SGD Momentum, Adam, AdamW, RMSProp
    Bootstrapping
    LOOCV
    K-Fold Cross Validation
    
    F1-Score
    Precision/Recall
    MSE/RMSE/MAE
    R-Squared/Adj. R-Squared
    Silhouette Score
    RSS

Python
    
Projects
    Recommender system - movies, etc.
        demonstrates usage of text and numeric data
        similarity between ev
    Regression based model
        predictive analytics
        estimate salaries by the description of teh job
    Classification
        emails as spam or not spam 
        from publicly available data (0 = not spam, 1 = spam)
        can you train a model for classification purposes
    Supervised learning 
        sort customers into good/better/best by their shopping information
    LLM
        demonstrates pre-processing
        GPT creation

# Machine Learning Basics
Supervised vs Unsupervised
Regression vs Classificaion 
Training and Evaluating ML Models


Supervised = Independent + Dependent variables
    - regression models
        continuous values
        prediction tests

        linear regression, fixed effects regression, xgboos

    - classification models
        categorical values
        decision-making tests

        logistic regression, random forest regression

Unsupervised = ONLY Independent variables
    - Clustering models
    - Outlier detection models

Performance metrics (Regression)
    RSS - residual sum of squares
    MSE - mean squared error (average of RSS)
    RMSE - Root mean squared error (easier to interpret)
    MAE - lower = better fit

    Determine how well the model did by comparing the predicted values and actual values

Performance metrics (Classification)
    Accuracy - correct / (correct + incorrect)
    Precision - true positive / (true positive + false positive)
    Recall - true positive / (true positive + false negative)
    F-1 Score - 2* (recall*precision / recall+precision)

Performance metrics (Clustering)
    homogeneity
    silhouette score
    completeness

Paused @ 00:58:07
