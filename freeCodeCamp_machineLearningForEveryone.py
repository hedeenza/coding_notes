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
    df["class"] = (df["class"] == "g").astype(int) # if it is a g, will make it 1, if it's not, aka it's an h, will make it a 0

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
    Supervised Learing Tasks
        Classification - predict discrete classes 
            multi-class classification - many possible classes, like plant species from a list
            binary-class classification - positive/negative, cat/dog, spam/not spam
        Regression - trying to come up with a number that's on some sort of scale
            gas prices, temperature tomorrow, house prices

00:19:57 Training Model
    Each row = a different data sample, the "feature vector" is all of the features in that row 
            - all of the features in all of the rows is the "feature matrix"
    Each column = a different feature
    ONE column = the label, the "target vector" is the label for that feature vector
            - all of the labels in all of the rows is the "target/labels matrix"

    Training = tinkering with the model so its output gets closer and closer to the known values
            - we don't use *all* of the data in training. If we did, we wouldn't know whether the model could handle new data
                TRAINING DATASET
                VALIDATION DATASET
                    - used during or after training to make sure the model can handle unseen data
                    - the loss from this data set NEVER gets fed back into the model, like the trainig dataset does
                TESTING DATASET
                    might be a 60-20-20 or 80-10-10 split between them
            - "loss" = the difference between the prediction by the model and the true values
                    - used at the end so we can see how generalizable our model is
                    - the TESTING DATA is where the final performance of your model is reported
    
    Performance Metrics
        "L1 Loss" = sum( |real - predicted| )
        "L2 Loss" = sum( (real - predicted)^2 ) # quadratic - so if it's close, the penalty is small, if it's off by a lot, there's a large penalty
        "Binary Cross-Entropy Loss" = -1/N * sum(real * log(predicted) + (1-real) * log((1-predicted)))
        LOSS DECREASES AS PERFORMANCE GETS BETTER

        "Accuracy" = how many times did it get it right out of how many times it made a prediction

00:30:57 Preparing Data
    
    # Back to the MAGIC data
    for label in cols[:-1]:  # going up to the last item in the list
        plt.hist(df[df["class"]==1][label], color="blue", label='gamma', alpha=0.7, density=True) 
        plt.hist(df[df["class"]==0][label], color="red", label='hadron', alpha=0.7, density=True) 

            # [df["class"]==1] means "get me everything from the table where the class is equal to 1"
            # ][label] means "now that you've done that, just get me the column labels
            # color = the histogram will be blue for the 1's, red for the 0's
            # label = the label for that group of data
            # alpha = transparency
            # density = True means the histograms will be normalized, and more visually comparable even though they may contain a different number of values (there's probably not 50% 1's and 50% 0's)

        plt.title(label)  # the title of the graph will be the column label
        plt.ylabel("Probability")  # the y-axis will be labeled probability
        plt.xlabel(label)  # the x-axis will also be the column label
        plt.legend()  # we're adding a legend
        plt.show()  # we want to show the plot!

    train, validate, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df)) ])
        # splitting the dataset into our "train", "validate", and "test" groups
        # np to call numpy
        # .split to split the data into sections
        # df to bring in the dataframe we assigned to 'df'
        # .sample to randomly split it 
        # (frac=1) you want 100% of the data to get sampled
        # [int(0.6*len(df),  will assign 60% of the length of the dataframe to the "train" group
        # int(0.8*len(df) ])  will assign from 60% to 80% of the length of the dataframe (the next 20%) to the "validate" group
        # the last 20% gets automatically assigned

    # Scaling the dataset so the magnitude of the values in each different column doesn't throw anything off
        # having values in one column be like 1000 and in another column be like 1 may weight the predictive capabilities in ways you don't want????
    def scale_dataset(dataframe): 
           x = dataframe[dataframe.cols[:-1]].values

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

