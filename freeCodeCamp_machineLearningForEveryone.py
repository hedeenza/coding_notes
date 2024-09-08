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
        x = dataframe[dataframe.cols[:-1]].values  # get all of the values from all of the columns up to the last item (-1)
        y = dataframe[dataframe.cols[-1]].values  # get the values from the last column, which is our "labels" column

    from sklearn.preprocessing import StandardScaler  # a module that will help us transform the data
                                                            # put this at the top with everything else

        scaler = StandardScaler()
        x = scaler.fit_transform(x)  # take x, fir the standard scaler to x, and transform x, and that will be our new x

        data = np.hstack((x, np.reshape(y, (-1, 1))))  # take two arrays and horizontally stack them together
                                                       # python is particular about the dimensions, so we need to "reshape()" y, which is just a vector, into the 2d object
                                                       # reshape is from the NumPy module, so we need to call that
                                                       # the -1 value tells python to infer the number of rows, which ends up being the length of the vector
                                                                # this is just like if we had put len(y) there instead
                                                       # the 1 value tells python that the 2d shape will be 1 column wide
        return data, x, y


    # BUT WAIT THERE'S A PROBLEM!!!
    print(len(train[train["class"]==1]))  # looking at how man gammas there are in the gamma's training set
    print(len(train[train["class"]==0]))  # looking at how man hadrons there are in the gamma's training set
                        # They don't match very well, because we randomly grabbed from our original data frame
                            # We can "make more hadron observations" by utilizing another module we can import

    from imblearn.over_sampling import RandomOverSampler  # put this at the top with the other stuff

    # Now we can modify what we did before to include tools that will help us with the group count disparity
    def scale_dataset(dataframe, oversample=False):  # include a way to control whether we want to oversample or not
        x = dataframe[dataframe.columns[:-1]].values  # fixing a typo "cols" > "columns"
        y = dataframe[dataframe.columns[-1]].values  

        scaler = StandardScaler()
        x = scaler.fit_transform(x)  

        if oversample:  # if we decide we want to oversample
           ros = RandomOverSampler()
           x, y = ros.fit_resample(x, y) # this is basically saying "keep sampling more from the group that there is less of until its count better matches the group that there is more of"

        data = np.hstack((x, np.reshape(y, (-1, 1))))  
                                                       
        return data, x, y

    # Now that our function is complete, we need to call it
    train, x_train, y_train = scale_dataset(train, oversample=True)  
                    # assigning the variables "train", "x_train", and "y_train"
                    # running our "train" dataset through our scaling function
                    # we *are* oversampling to make up for the discrepancy between the number of gammas and hadrons
                    
                    # Now, within our training chunk of the data, our training dataset, our "features / x set" and our "labels / y set" should all be normalized 

    # Checking if our normalization worked
    len(y_train)  # example results in the tutorial is 14,838
    sum(y_train == 1)  # the number of gammas is 7,419 in the example from the video
    sum(y_train == 0)  # the number of hadrons is 7,419 in the example from the video
                # it worked!!!

    # Next, we need to run our normalizing function on our validate and test data set chunks
    validate, x_validate, y_validate = scale_dataset(validate, oversample=False)  
    test, x_test, y_test = scale_dataset(test, oversample=False)  
                # We DON'T want to oversample with our validate or test groups
                # We want to know how well our model performs when its totally new data, which can be of any amount of any group
                # If we oversampled it, we would be making the groups equal, which may end up giving us more confidence in the model's performance than we should have in the case it didn't perform as well with data it was totally blind to

00:44:43 K-Nearest Neighbors

    For a new data point, take the label of the majority of things that are around that new data point, and guess that the new data point probably has the same label     

    Things we need to define!
        1. Distance - a.k.a "Euclidian Distance" in a 2D plot
                    - "the straight line distance on a 2D plot"
                    - "how far away from the new data point do we look?"
                    - d = sqrt( (x1-x2)**2 + (y1-y2)**2 )  # The basic geometry equation for distance between two points
        2. K - "how many neighbors do we use to determine the label of our new data point?"

        # We can do this concept for any number of factors, not just 2 like on a 2d plot

00:52:42 KNN Implementation

# Using the kNN Method!!!
from sklearn.neighbors import KNeighborsClassifier # bringning in our module

knn_model = KNeighborsClassifier(n_neighbors=1) # assigning our model to a variable, and choosing just 1 neighbor to start

knn_model.fit(x_train, y_train)  # calling our model to use our x training group and our y training group

y_predicted = knn_model.predict(x_test) # running the model on the x_test group and getting a set of predictions
y_predicted  # this will show the predicted values in an array

y_test  # this will show the *real* values, and we can compare these real values to our predicted values

# But we can do a little better than counting and comparing
from sklearn.metrics import classification_report # bring in a module that gives us a nicer report of how our model did

print(classification_report(y_test, y_predicted)) # print out our nice prediction report, given that the y_test is our *real* values and our y_predicted is our *we did our best to guess* values

# The chart it displays 
                precision   recall  f1-score    support

            0       0.77    0.68    0.72        1305
            1       0.84    0.89    0.87        2499

    accuracy                        0.82        3804
    macro avg       0.80    0.79    0.79        3804
weighted avg        0.82    0.82    0.82        3804

        # accuracy / f1-score intersection = "how many did we get right out of our number of guesses?"
        # precision = "out of all of the ones we've labeled as positive, how many are true positives?"
                        # DEALS WITH FALSE POSITIVES
                # "we labeled some a number of the data points as being 0; 77% of the ones we said were 0 were actually 0
                # "we labeled some b number of the data points as being 1; 84% of the ones we said were 1 were actually 0
        # recall = "out of all the ones we KNOW are positive, how many did we label as positive?"
                        # DEALS WITH FALSE NEGATIVES
                # "We know some c number of the data points are 0; we labeled 68% of them as being 0"
                # "We know some d number of the data points are 1; we labeled 89% of them as being 1"
        # f1-score = "like a combination of the precision and recall"
                # "what we're going to look at primarily because we have an unbalanced data set"

        # For our data, predicting the 1's went quite a bit better than predicting the 0's

        # Our 0.72 and 0.87 scores aren't too bad, but this model is pretty simple, so we might be able to get more accuracy with a more complex model!

00:58:43 Naive Bayes

# Before we can undersand the "Naive Bayes" model, we need to understand...
    Conditional Probability
    Base Rates
                Covid Test Result
                +           -
Has     Yes     531         6
Covid?  No      20          9443

        We have 6 false negatives (the test said no, but they have covid)
        We have 20 false positives (the test said yes, but they don't have covid)

1. What is the probability of haing covid, given a positive test?
    
        P (covid|+test)
        531 +test +covid / 551 +test ~96.4%

2. Bayes' Rule

        P(A|B) = [ P(B|A) * P(A) ] / P(B)  # we can tell the probability of A, given B, if we have
                                                # the probability of A
                                                # the probability of B
                                                # the probability of B, given A
3. Using Bayses' Rule on an Example

    P(false positive) = 0.05
    P(false negative) = 0.01
    P(disease) = 0.1 
    P(disease|+test) = ?

              + test      - test
    disease     0.99        0.01
    no dis      0.05        0.95

    P(A|B) = [ P(B|A) * P(A) ] / P(B)  

    P(disease|+test) = [ P(+test|disease) * P(disease) ] / P(+test)  
    P(disease|+test) = [ 0.99 * 0.1 ] / P(+test) 
            
            Probability you got a postitive test is 
                P(+test|disease) * P(disease) +
                P(+test|no disease) * P(no disease) 

                0.99 * 0.1 + 0.05 * 0.9 # we're basically assigning weights to the + testing scenarios
    
    P(disease|+test) ~ 0.6875 ~ 68.75%

# In Naive Bayes, ***we're assuming all of the features, the x's, are INDEPENDENT***

# argmax = picking the k that maximizes....... (need to rewatch ~01:17:00)

# MAP = maximum a posteriori
        # pick the k that minimizes the chance of mis-classification


01:17:30 Naive Bayes Implementation
# Getting our module
from sklearn.naive_bayes import GaussianNB

# Assigning the model function to a variable
nb_model = GaussianNB()

# Fitting the model to our training data
nb_model = nb_model.fit(x_train, y_train)

# Making our Prediction
y_predicted = nb_model.predict(x_test)

# Making our nice classification table
print(classification_report(y_test, y_predicted)

This model performed a little worse across all metrics, but it was important to test it out

01:19:22 Logistic Regression

Changing our model from y being probability to y being odds (P/1-P) can help
    P must be 0 - 1
    Odds can be any value

In our regression, the slope, x, and b can all be negative, and may result in a negative probability
      taking the natural log of the odds ratio will prevent a negative probability from occuring(???) # I don't know how you can just do that to one side of the equation in this context without further explanation
      we can do some other manipulation to eventually arrive at a *SIGMOID FUNCTION*

# This is the form of a sigmoid function
S(y) = 1 / (1 + e**-y)

# With only one feature / x = "simple logistic regression"
# With many features / x's = "multiple logistic regression"

01:27:56 Log Regression Implementation

# Loading in our Logistic Regression Module
from sklearn.linear_model import LogisticRegression

# Calling our model
lr_model = LogisticRegression()

# Fitting out model to our data
lr_model = lr_model.fit(x_train, y_train)
            # you can change the "penalty" in this model
            # default is "L2", which is that quadratic penalty from before

# Testing our model 
y_predicted = lr_model.predict(x_test)

# Make our nice classification table
print(classification_report(y_test, y_predicted))

This model's results were better than Naive Bayes but not quite as good as kNN

01:29:13 Support Vector Machine (SVM)

The last model for classification for this course

The goal is to find the line that best DIVIDES the data into its groups
      but it could also be a plane in 3d, or "hyperplane" in more than 3 dimensions

We also care about the "margin" in SVM. That's how far the existing points are from the hyperplane we creat
            # we want the hyperplane with the largest margin
    The data points that are ON the margin and help define the margin are called "support vectors"

***SVM's are NOT always very robust to outliers***

"The Kernel Trick" - when the data cannot easily be divided by a single line, you can go x --> x**2, and potentially find a better line
            # you may be able to do this with even higher powers as well???

01:37:54 SVM Implementation

# Getting out model module loaded
from sklearn.svm import SVC  # this is our "support vector classifier"

# Assign our model to a variable
svc_model = SVC()

# Fit our model to our data
svc_model = svc_model.fit(x_train, y_train)
            # There are a lot of options you can change here as well!

# Test our model 
y_predicted = svc_model.predict(y_test)

# Print our nice results table
print(classification_report(y_test, y_predicted))

With the SVC model, accuracy went up even higher than it was with the kNN model!

01:39:44 Neural Networks

3 Layers to a Neural Networkd
    Input - these are our features, which get multiplied by some weight, and the sum of those products gets fed into the "neuron"
            - we can also add a "bias" term, to shift the results a little
            - the sum of the inputs and the bias go into the neuron, then go to the "activation function" which then goes to the output
    Hidden
    Output

Without an "activation function", a neural net is just a linear model
    Some activation functions
        Sigmoid - like earlier! runs from 0 - 1
        Tanh - runs from -1 to 1
        RELU - any input < 0 is 0; any input >= 0 is linear

When training a model, we can use "Backpropogation" to determine how much the loss related to each feature is affecting the accuracy of the model
            # so we can create a new weight for each of our features to reduce the loss in the future
    Learning Rate = how big the steps are that we're adjusting the old weight by to get the new weight
            # We update all of our weights by the same amount for one round of training

01:47:57 Tensorflow

We've used some Machine Learning Libraries already:
    sklearn

Tensorflow makes it easy to define these models and have control over what we're feeding into them
e.g.

    model = tf.keras.Sequential([  # Let's create a sequential neural net
        tf.keras.layers.Dense(16, activation='relu'), # let's create a dense layer, where they're all interconnected, made up of 16 nodes and relu activation
        tf.keras.layers.Dense(16, activation='relu'), # let's make *another* layer that's like the previous one
        tf.keras.layers.Dense(1) # our output node will just be one node
    ])

01:49:50 Classification NN using Tensorflow

# importing our module
import tensorflow as tf

# Creating our Tensorflow neural network model
nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)), # dense layer, 32 nodes, relu activation, need to define the shape at first
    tf.keras.layers.Dense(32, activation='relu'), # dense layer, 32 nodes, relu activation, need to define the shape at first
    tf.keras.layers.Dense(1, activation='sigmoid') # one dense node as the output layer, sigmoid activation to have the final classiciation be 0 or 1 based on the sigmoid shape we saw earlier
    ])

nn_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001), # learning rate is 0.001
    loss='binary_crossentropy'  # our loss type
    metrics=['accuracy']) # so we can see accuracy in a plot later

# Adding in tensor-flow pre-defined plots for loss and accuracy
            # These will need to go ABOVE our nn_model.compile() section
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Binary crossentropy')
    plt.legend()
    plt.grid(True)
    plt.show()
            # This one is plotting the loss over each of the Epochs
            # an "Epoch" is a training cycle

def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
            # This one is plotting the accuracy over each of the Epochs
            # an "Epoch" is a training cycle

# tensorflow keeps track of all the history, allowing you to plot it later on
history = nn_model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,  # fraction of the training data to be used as the validation data
        verbose=0  # don't print anything as you run
)
        # Running this step will take some time. Let it cook while it trains.

# Now that it's cooked, plot the loss and history throughout the training
plot_loss(history)
plot_accuracy(history)

# The graphs reveal that loss is decreasing and accuracy is increasing over time, which is good!
        # Note it will say it's performing better on the training data than on the validation data
        # This is because it is adapting specifically to the training data

# Re-writing the neural network to use a "grid seach", allowing us to test multiple hyper-parameters like number of nodes in each layer
        # Adding a drop-out layer too, which randomly chooses nodes not to train on certain cycles, which REDUCES over-fitting of the model
def train_model(x_train, y_train, num_nodes, dropout_prob, learning_rate, batch_size, epochs):
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(10,)), # number of nodes is set to the num_nodes variable
        tf.keras.layers.Dropout(dropout_prob), # added a dropout node, the probability it doesn't train a node is set to the dropout_prob variable
        tf.keras.layers.Dense(num_nodes, activation='relu'), 
        tf.keras.layers.Dropout(dropout_prob), # added a dropout node, the probability it doesn't train a node is set to the dropout_prob variable
        tf.keras.layers.Dense(1, activation='sigmoid') 
        ])

    nn_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate), # changed to the learning_rate variable
            loss='binary_crossentropy'  
            metrics=['accuracy']
            ) 

    history = nn_model.fit(
        x_train, y_train,
        epochs=epochs, # number of epochs set to epochs variable
        batch_size=batch_size, # batch size changed to the batch_size variable
        validation_split=0.2,  
        verbose=0  
        )

    return nn_model, history

# Re-writing our plotting functions so they're side by side for easier viewing
def plot_loss(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))  # we want one row, two columns for that row, and those will be my plots; figure is 10 units wide and 4 units tall (size ratio)
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary crossentropy')
    ax1.grid(True)
    plt.show()

    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('accuracy')
    ax2.grid(True)
    plt.show()


# Setting our variables and our nested for loops in order to test out every combination of these variables
least_val_loss = float('inf')  # to help us keep track of the model with the least validation loss
least_loss_model = None

epochs=100
for num_nodes in [16, 32, 64]:
    for dropout_prob in [0, 0.2]:
        for learning_rate in [0.01, 0.005, 0.001]:
            for batch_size in [32, 64, 128]:
                print(f"{num_nodes} nodes, dropout {dropout_prob}, learning rate {learning_rate}, batch size {batch_size}") # printing out all of our parameters
                model, history = train_model(x_train, y_train, num_nodes, dropout_prob, learning_rate, batch_size, epoch)
                plot_history(history)
                val_loss = mode.evaluate(x_valid, y_valid)  # figuring out our loss for the validation set from our original data split

                if val_loss < least_val_loss:
                    least_val_loss = val_loss  # if something beats what's the least so far, make that what we keep track of in the least variable
                    least_loss_model = model  # the model with the least loss is whatever model just set the record for the least amount of loss


# !!! one option is, in the "history =", section...
            # to pass in the "validation_data=valid" instead of the 
            # "validation_split=0.2"

# as is, the model is taking 20% of the training data and validating against that
# and also validating against the validation data
# so the accuracy in the plot, and the accuracy value reported under the plot will be slightly different

Now that we have our least loss model and it's assigned to least_loss_model, we can use that to predict on our testing data!

y_predicted = least_lost_model.predict(x_test)
y_predicted = (y_predicted > 0.5).astype(int).reshape(-1,)
            # will initially report floats that are not quite 0 or 1
            # this pushes anything above 0.5 to be 1, anything below to be 0
            # reshapes it into a one-dimensional array
y_predicted

# And now the moment we've been waiting for, seeing how well we did with our nice table!
print(classification_report(y_test, y_predicted))

performed pretty well, but similarly to the SVF() from before.

02:10:12 Linear Regression




02:34:54 Lin Regression Implementation
02:57:44 Lin Regression using a Neuron
03:00:15 Regression NN using Tensorflow
03:13:13 K-Means Clustering
03:23:46 Principal Component Analysis
03:33:54 K-Means and PCA Implementations

