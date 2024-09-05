Machine Learning for Everybody - Full Course
https://www.youtube.com/watch?v=i_LwzRVP7bg

Contents
00:00:00 Intro
00:00:58 Data/Colab Intro
00:08:45 Intro to Machine Learning
00:12:26 Features
00:17:23 Classification/Regression
00:19:57 Training Model
00:30:57 Preparing Data
00:44:43 K-Nearest Neighbors
00:52:42 KNN Implementation
01:08:43 Naive Bayes
01:17:30 Naive Bayes Implementation
01:19:22 Logistic Regression
01:27:56 Log Regression Implementation
01:29:13 Support Vector Machine
01:37:54 SVM Implementation
01:39:44 Neural Networks
01:47:57 Tensorflow
01:49:50 Classification NN using Tensorflow
02:10:12 Linear Regression
02:34:54 Lin Regression Implementation
02:57:44 Lin Regression using a Neuron
03:00:15 Regression NN using Tensorflow
03:13:13 K-Means Clustering
03:23:46 Principal Component Analysis
03:33:54 K-Means and PCA Implementations

Contents
00:00:00 Intro
    supervised / unsupervised

00:00:58 Data/Colab Intro
    MAGIC gamma telescope dataset

    # Packages to import 
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Dataset
    https://archive.ics.uci.edu/ml
        it's the magic04.data

    # reading the data set
    cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", 
            "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
    df = pd.read_csv('magic-gamma-telescope/magic04.data', names=cols)"
    df.head()

    # df["class"].unique()
        # array(['g', 'h'], dtype=object)
        # g = gammas
        # h = hadrons
        # converting these from g's and h's to 0's and 1's
    df["class"] = (df["class"] == "g").astype(int) # if it is a g, will make it 1, if it's not, will make it a 0

    # We're building a CLASSIFICATION MODEL
        # Determining whether future samples are g or h based on their features
        # features = things we pass into our model to try and predict which label it is
            # for us, our features are fLength:fDist, and our label = class

        # We know the label for our inputs, making this SUPERVISED learning

00:08:45 Intro to Machine Learning

Machine Learning = algorithms which help a computer learn from data without explicit programming
                 = solve specific problems and make predictions using data
                 = a subset of the AI field

Supervised Learning = labeled data is used to help make predictions on future, unlabeled data

Unsupervised Learning = unlabeled data is used to help learn about patterns in the data
                      = can't tell you what it is, but can group things together

Reinforcement Learning = an agent is learning based on an interactive environment with rewards and penalties


00:12:26 Features
    features = our inputs that the model uses to predict the label for those inputs
        types: qualitative (categorical data)
               nominal data = NO INHERENT ORDER TO THE DATA
                    ONE-HOT ENCODING
                        the method we feed nominal data into a computer
                            if it matches, it's a 1
                            if it doesn't, it's a 0
                    e.g. [USA, India, Canada, France]
                    USA = [1,0,0,0]  # if the input is from the US, give it a 1 here
                    India = [0,1,0,0]  # if the input is from the India, give it a 1 here
                    Canada = [0,0,1,0]  # if the input is from the Canada, give it a 1 here
                    France = [0,0,0,1]  # if the input is from the France, give it a 1 here

               ordinal data = INHERENT ORDER TO THE DATA
                    we can just mark these 1-5 or whatever our scale calls for
        types: quantitative (numerical, discrete, or continuous)


00:17:23 Classification/Regression
    stopped here


00:19:57 Training Model
00:30:57 Preparing Data
00:44:43 K-Nearest Neighbors
00:52:42 KNN Implementation
01:08:43 Naive Bayes
01:17:30 Naive Bayes Implementation
01:19:22 Logistic Regression
01:27:56 Log Regression Implementation
01:29:13 Support Vector Machine
01:37:54 SVM Implementation
01:39:44 Neural Networks
01:47:57 Tensorflow
01:49:50 Classification NN using Tensorflow
02:10:12 Linear Regression
02:34:54 Lin Regression Implementation
02:57:44 Lin Regression using a Neuron
03:00:15 Regression NN using Tensorflow
03:13:13 K-Means Clustering
03:23:46 Principal Component Analysis
03:33:54 K-Means and PCA Implementations

