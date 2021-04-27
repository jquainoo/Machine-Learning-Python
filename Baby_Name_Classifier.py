#!/usr/bin/env python
# coding: utf-8

# <h2>About this Project</h2>
# <p>In this project, you will use naive bayes to build a baby name classifier. Your classifier will use the features commonly found in names to distinguish whether a name is likely to be given to a baby girl or boy. You'll train your classifier to identify these certain qualities of names and classify new examples based on those qualities.</p>
# 
# <h3>Evaluation</h3>
# 
# <p><strong>This project must be successfully completed and submitted in order to receive credit for this course. Your score on this project will be included in your final grade calculation.</strong><p>
#     
# <p>You are expected to write code where you see <em># YOUR CODE HERE</em> within the cells of this notebook. Not all cells will be graded; code input cells followed by cells marked with <em>#Autograder test cell</em> will be graded. Upon submitting your work, the code you write at these designated positions will be assessed using an "autograder" that will run all test cells to assess your code. You will receive feedback from the autograder that will identify any errors in your code. Use this feedback to improve your code if you need to resubmit. Be sure not to change the names of any provided functions, classes, or variables within the existing code cells, as this will interfere with the autograder. Also, remember to execute all code cells sequentially, not just those you’ve edited, to ensure your code runs properly.</p>
#     
# <p>You can resubmit your work as many times as necessary before the submission deadline. If you experience difficulty or have questions about this exercise, use the Q&A discussion board (found in the Live Labs section of this course) to engage with your peers or seek assistance from the instructor.<p>
# 
# <p>Before starting your work, please review <a href="https://s3.amazonaws.com/ecornell/global/eCornellPlagiarismPolicy.pdf">eCornell's policy regarding plagiarism</a> (the presentation of someone else's work as your own without source credit).</p>
# 
# <h3>Submit Code for Autograder Feedback</h3>
# 
# <p>Once you have completed your work on this notebook, you will submit your code for autograder review. Follow these steps:</p>
# 
# <ol>
#     <li><strong>Save your notebook —</strong> Click <strong>Save and Checkpoint</strong> in the "File" menu.</li>
#   <li><strong>Mark as Completed —</strong> In the blue menu bar along the top of this code exercise window, you’ll see a menu item called <strong>Education</strong>. In the <strong>Education</strong> menu, click <strong>Mark as Completed</strong> to submit your code for autograder/instructor review. This process will take a moment and a progress bar will show you the status of your submission.</li>
# 	<li><strong>Review your results —</strong> Once your work is marked as complete, the results of the autograder will automatically be presented in a new tab within the code exercise window. You can click on the assessment name in this feedback window to see more details regarding specific feedback/errors in your code submission.</li>
#   <li><strong>Repeat, if necessary —</strong> The Jupyter notebook will always remain accessible in the first tabbed window of the exercise. To reattempt the work, you will first need to click <strong>Mark as Uncompleted</strong> in the <strong>Education</strong> menu and then proceed to make edits to the notebook. Once you are ready to resubmit, follow steps one through three. You can repeat this procedure as many times as necessary.</li>
# </ol>

# <h2>Getting Started</h2>
# <h3>Prepare Text for Machine Learning </h3>
# 
# <p> If we are to create a classifier for text, we'll first need to think about the format of our data. Take a look at the files <code>girls.train</code> and <code>boys.train</code>. For example with the unix command: <pre>cat girls.train</pre> 
# <pre>
# ...
# Addisyn
# Danika
# Emilee
# Aurora
# Julianna
# Sophia
# Kaylyn
# Litzy
# Hadassah
# </pre>
# This file contains names that are more or less commonly used for girls. The problem with the current data in this file is that the names are in plain text, which is not a format our machine learning algorithm can work with effectively. You need to transform these plain text names into some vector format, where each name becomes a vector that represents a point in some high dimensional input space. </p>
# 
# <p>That is exactly what the following Python function <code>name2features</code> does, by arbitrarily chunking and hashing different string extractions from each baby name inputted, thus transforming the string into a quantitative feature vector:</p>

# <h3>Python Initialization</h3>
# <p>Please run the following code to initialize your Python kernel. You should be running a version of Python 3.x.<p>

# In[1]:


import numpy as np
import sys
sys.path.append('/home/codio/workspace/.guides/hf')
from helper import *

get_ipython().run_line_magic('matplotlib', 'inline')
print('You\'re running python %s' % sys.version.split(' ')[0])


# <h3>The <code>hashfeatures</code> and <code>name2features</code> Functions</h3>
# <p>Below, the <code>hashfeatures</code> and <code>name2features</code> functions will take the plain text names and convert them to feature vectors so that you'll be able to work with the data effectively.  

# In[2]:


def hashfeatures(baby, B, FIX):
    """
    Input:
        baby : a string representing the baby's name to be hashed
        B: the number of dimensions to be in the feature vector
        FIX: the number of chunks to extract and hash from each string
    
    Output:
        v: a feature vector representing the input string
    """
    v = np.zeros(B)
    for m in range(FIX):
        featurestring = "prefix" + baby[:m]
        v[hash(featurestring) % B] = 1
        featurestring = "suffix" + baby[-1*m:]
        v[hash(featurestring) % B] = 1
    return v


# In[3]:


def name2features(filename, B=128, FIX=3, LoadFile=True):
    """
    Output:
        X : n feature vectors of dimension B, (nxB)
    """
    # read in baby names
    if LoadFile:
        with open(filename, 'r') as f:
            babynames = [x.rstrip() for x in f.readlines() if len(x) > 0]
    else:
        babynames = filename.split('\n')
    n = len(babynames)
    X = np.zeros((n, B))
    for i in range(n):
        X[i,:] = hashfeatures(babynames[i], B, FIX)
    return X


# <p>In the code cell above, <code>name2features</code> reads every name in the given file and converts it into a 128-dimensional feature vector by first assembling substrings (based on the parameter 'FIX'), then hashing these assembled substrings and modifying the feature vector index (the modulo of the number of dimensions) that corresponds to this hash value. </p> 
# 
# <p>Can you see how the feature vector for each name changes with different parameters? (Understanding how these features are constructed will help you later on in the challenge.)<br></p>

# <h3>The <code>genTrainFeatures</code> Function</h3>
# <p>We have provided you with a python function <code>genTrainFeatures</code>, which transforms the names into features and loads them into memory. 

# In[4]:


def genTrainFeatures(dimension=128):
    """
    Input: 
        dimension: desired dimension of the features
    Output: 
        X: n feature vectors of dimensionality d (nxd)
        Y: n labels (-1 = girl, +1 = boy) (n)
    """
    
    # Load in the data
    Xgirls = name2features("girls.train", B=dimension)
    Xboys = name2features("boys.train", B=dimension)
    X = np.concatenate([Xgirls, Xboys])
    
    # Generate Labels
    Y = np.concatenate([-np.ones(len(Xgirls)), np.ones(len(Xboys))])
    
    # shuffle data into random order
    ii = np.random.permutation([i for i in range(len(Y))])
    
    return X[ii, :], Y[ii]


# <p>You can call the following command to return two vectors, one holding all the concatenated feature vectors and one holding the labels of all boys and girls names.</p>

# In[5]:


X, Y = genTrainFeatures(128)


# <h2> The Na&iuml;ve Bayes Classifier </h2>
# 
# <p> The Na&iuml;ve Bayes classifier is a linear classifier based on Bayes Rule. The following cells will walk you through steps and ask you to finish the necessary functions in a pre-defined order. <strong>As a general rule, you should avoid tight loops at all costs.</strong></p>

# <h3>Part One: Class Probability [Graded]</h3>
# <i>
# <p>Estimate the class probability $P(y)$ in 
# <b><code>naivebayesPY</code></b>. This should return the probability that a sample in the training set is positive or negative, independent of its features.
# </p>
# </i>

# In[18]:


def naivebayesPY(X, Y):
    """
    naivebayesPY(Y) returns [pos,neg]

    Computation of P(Y)
    Input:
        X : n input vectors of d dimensions (nxd)
        Y : n labels (-1 or +1) (nx1)

    Output:
        pos: probability p(y=1)
        neg: probability p(y=-1)
    """
    
    # add one positive and negative example to avoid division by zero ("plus-one smoothing")
    Y = np.concatenate([Y, [-1,1]])
    n = len(Y)
    
    # Pre-configuring the size of matrix X
    "We do not need to preconfigure the matrix - we can just skip it and calculate the probabilities

    ## fill in code here
    pos = np.count_nonzero(Y == 1) / n
    neg = np.count_nonzero(Y == -1) / n
    
    return pos,neg

pos, neg = naivebayesPY(X,Y)

# In[19]:

# The following tests will check that the probabilities returned by your function sum to 1 (test1) and return the correct probabilities for a given set of input vectors (tests 2-4)

# Check that probabilities sum to 1
def naivebayesPY_test1():
    pos, neg = naivebayesPY(X,Y)
    return np.linalg.norm(pos + neg - 1) < 1e-5

# Test the Naive Bayes PY function on a simple example
def naivebayesPY_test2():
    x = np.array([[0,1],[1,0]])
    y = np.array([-1,1])
    pos, neg = naivebayesPY(x,y)
    pos0, neg0 = .5, .5
    test = np.linalg.norm(pos - pos0) + np.linalg.norm(neg - neg0)
    return test < 1e-5

# Test the Naive Bayes PY function on another example
def naivebayesPY_test3():
        x = np.array([[0,1,1,0,1],
            [1,0,0,1,0],
            [1,1,1,1,0],
            [0,1,1,0,1],
            [1,0,1,0,0],
            [0,0,1,0,0],
            [1,1,1,0,1]])    
        y = np.array([1,-1, 1, 1,-1,-1, 1])
        pos, neg = naivebayesPY(x,y)
        pos0, neg0 = 5/9., 4/9.
        test = np.linalg.norm(pos - pos0) + np.linalg.norm(neg - neg0)
        return test < 1e-5

# Tests plus-one smoothing
def naivebayesPY_test4():
    x = np.array([[0,1,1,0,1],[1,0,0,1,0]])    
    y = np.array([1,1])
    pos, neg = naivebayesPY(x,y)
    pos0, neg0 = 3/4., 1/4.
    test = np.linalg.norm(pos - pos0) + np.linalg.norm(neg - neg0)
    return test < 1e-5    
        
    
runtest(naivebayesPY_test1, 'naivebayesPY_test1')
runtest(naivebayesPY_test2,'naivebayesPY_test2')
runtest(naivebayesPY_test3,'naivebayesPY_test3')
runtest(naivebayesPY_test4,'naivebayesPY_test4')


# In[ ]:


# Autograder test cell- worth 1 point
# runs naivebayesPY_test1


# In[ ]:


# Autograder test cell- worth 1 points
# runs naivebayesPY_test2


# In[ ]:


# Autograder test cell- worth 1 points
# runs naivebayesPY_test3


# In[ ]:


# Autograder test cell- worth 1 points
# runs naivebayesPY_test4


# <h3>Part Two: Conditional Probability [Graded]</h3>
# <p>Estimate the conditional probabilities $P([\mathbf{x}]_{\alpha}|y)$ in 
# <b><code>naivebayesPXY</code></b>. Notice that by construction, our features are binary categorical features. Use a <b>categorical</b> distribution as model and return the probability vectors for each feature being 1 given a class label.  Note that the result will be two vectors of length d (the number of features), where the values represent the probability that feature i is equal to 1.
# </p> 

# In[7]:


def naivebayesPXY(X,Y):
    """
    naivebayesPXY(X, Y) returns [posprob,negprob]
    
    Input:
        X : n input vectors of d dimensions (nxd)
        Y : n labels (-1 or +1) (n)
    
    Output:
        posprob: probability vector of p(x_alpha = 1|y=1)  (d)
        negprob: probability vector of p(x_alpha = 1|y=-1) (d)
    """
    
    # add one positive and negative example to avoid division by zero ("plus-one smoothing")
    n, d = X.shape
    X = np.concatenate([X, np.ones((2,d)), np.zeros((2,d))])
    Y = np.concatenate([Y, [-1,1,-1,1]])
    n, d = X.shape
    
    # YOUR CODE HERE
    posprob = np.mean(X[Y == 1], axis=0)
    negprob = np.mean(X[Y == -1], axis=0)
    
    return posprob, negprob
    

posprob, negprob = naivebayesPXY(X,Y)


# In[8]:


# The following tests check that your implementation of naivebayesPXY returns the same posterior probabilities as the correct implementation, in the correct dimensions

# test a simple toy example with two points (one positive, one negative)
def naivebayesPXY_test1():
    x = np.array([[0,1],[1,0]])
    y = np.array([-1,1])
    pos, neg = naivebayesPXY(x,y)
    pos0, neg0 = naivebayesPXY_grader(x,y)
    test = np.linalg.norm(pos - pos0) + np.linalg.norm(neg - neg0)
    return test < 1e-5

# test the probabilities P(X|Y=+1)
def naivebayesPXY_test2():
    pos, neg = naivebayesPXY(X,Y)
    posprobXY, negprobXY = naivebayesPXY_grader(X, Y)
    test = np.linalg.norm(pos - posprobXY) 
    return test < 1e-5

# test the probabilities P(X|Y=-1)
def naivebayesPXY_test3():
    pos, neg = naivebayesPXY(X,Y)
    posprobXY, negprobXY = naivebayesPXY_grader(X, Y)
    test = np.linalg.norm(neg - negprobXY)
    return test < 1e-5


# Check that the dimensions of the posterior probabilities are correct
def naivebayesPXY_test4():
    pos, neg = naivebayesPXY(X,Y)
    posprobXY, negprobXY = naivebayesPXY_grader(X, Y)
    return pos.shape == posprobXY.shape and neg.shape == negprobXY.shape

runtest(naivebayesPXY_test1,'naivebayesPXY_test1')
runtest(naivebayesPXY_test2,'naivebayesPXY_test2')
runtest(naivebayesPXY_test3,'naivebayesPXY_test3')
runtest(naivebayesPXY_test4,'naivebayesPXY_test4')


# In[30]:


# Autograder test cell- worth 1 point
# runs naivebayesPXY_test1


# In[31]:


# Autograder test cell- worth 1 point
# runs naivebayesPXY_test2


# In[32]:


# Autograder test cell- worth 1 point
# runs naivebayesPXY_test3


# In[33]:


# Autograder test cell- worth 1 point
# runs naivebayesPXY_test4


# <h3>Part Three: Log Lokelihood [Graded]</h3>
# 
# <i>
# <p>Calculate the log likelihood $\log P(\mathbf{x}|y)$ for each point in X_test in 
# <b><code>loglikelihood</code></b> and label Y_test. Recall that the likelihood is given by the product of the conditional probabilities of each feature and that $\log(ab) = \log a + \log b$.
# </p> 
# <i>

# In[34]:


def loglikelihood(posprob, negprob, X_test, Y_test):
    """
    loglikelihood(posprob, negprob, X_test, Y_test) returns loglikelihood of each point in X_test
    
    Input:
        posprob: conditional probabilities for the positive class (d)
        negprob: conditional probabilities for the negative class (d)
        X_test : features (nxd)
        Y_test : labels (-1 or +1) (n)
    
    Output:
        loglikelihood of each point in X_test (n)
    """
    n, d = X_test.shape
    loglikelihood = np.zeros(n)
    
    positive = (Y_test == 1)
    negative = (Y_test == -1)
  
    loglikelihood[positive] = X_test[positive]@np.log(posprob) + (1 - X_test[positive])@np.log(1 - posprob)
    loglikelihood[negative] = X_test[negative]@np.log(negprob) + (1 - X_test[negative])@np.log(1 - negprob)

    return loglikelihood

# compute the loglikelihood of the training set
posprob, negprob = naivebayesPXY(X,Y)
loglikelihood(posprob,negprob,X,Y) 


# In[35]:


# The following tests check that your implementation of loglikelihood returns the same values as the correct implementation for three different datasets

X, Y = genTrainFeatures(128)
posprobXY, negprobXY = naivebayesPXY_grader(X, Y)

# test if the log likelihood of the training data are all negative
def loglikelihood_testneg():
    ll=loglikelihood(posprob,negprob,X,Y);
    return all(ll<0)

# test if the log likelihood of the training data matches the solution
def loglikelihood_test0():
    ll=loglikelihood(posprob,negprob,X,Y);
    llgrader=loglikelihood_grader(posprob,negprob,X,Y);
    return np.linalg.norm(ll-llgrader)<1e-5

# test if the log likelihood of the training data matches the solution
# (positive points only)
def loglikelihood_test0a():
    ll=loglikelihood(posprob,negprob,X,Y);
    llgrader=loglikelihood_grader(posprob,negprob,X,Y);
    return np.linalg.norm(ll[Y==1]-llgrader[Y==1])<1e-5

# test if the log likelihood of the training data matches the solution
# (negative points only)
def loglikelihood_test0b():
    ll=loglikelihood(posprob,negprob,X,Y);
    llgrader=loglikelihood_grader(posprob,negprob,X,Y);
    return np.linalg.norm(ll[Y==-1]-llgrader[Y==-1])<1e-5


# little toy example with two data points (1 positive, 1 negative)
def loglikelihood_test1():
    x = np.array([[0,1],[1,0]])
    y = np.array([-1,1])
    posprobXY, negprobXY = naivebayesPXY_grader(X, Y)
    loglike = loglikelihood(posprobXY[:2], negprobXY[:2], x, y)
    loglike0 = loglikelihood_grader(posprobXY[:2], negprobXY[:2], x, y)
    test = np.linalg.norm(loglike - loglike0)
    return test < 1e-5

# little toy example with four data points (2 positive, 2 negative)
def loglikelihood_test2():
    x = np.array([[1,0,1,0,1,1], 
        [0,0,1,0,1,1], 
        [1,0,0,1,1,1], 
        [1,1,0,0,1,1]])
    y = np.array([-1,1,1,-1])
    posprobXY, negprobXY = naivebayesPXY_grader(X, Y)
    loglike = loglikelihood(posprobXY[:6], negprobXY[:6], x, y)
    loglike0 = loglikelihood_grader(posprobXY[:6], negprobXY[:6], x, y)
    test = np.linalg.norm(loglike - loglike0)
    return test < 1e-5


# one more toy example with 5 positive and 2 negative points
def loglikelihood_test3():
    x = np.array([[1,1,1,1,1,1], 
        [0,0,1,0,0,0], 
        [1,1,0,1,1,1], 
        [0,1,0,0,0,1], 
        [0,1,1,0,1,1], 
        [1,0,0,0,0,1], 
        [0,1,1,0,1,1]])
    y = np.array([1, 1, 1 ,1,-1,-1, 1])
    posprobXY, negprobXY = naivebayesPXY_grader(X, Y)
    loglike = loglikelihood(posprobXY[:6], negprobXY[:6], x, y)
    loglike0 = loglikelihood_grader(posprobXY[:6], negprobXY[:6], x, y)
    test = np.linalg.norm(loglike - loglike0)
    return test < 1e-5


runtest(loglikelihood_testneg, 'loglikelihood_testneg (all log likelihoods must be negative)')
runtest(loglikelihood_test0, 'loglikelihood_test0 (training data)')
runtest(loglikelihood_test0a, 'loglikelihood_test0a (positive points)')
runtest(loglikelihood_test0b, 'loglikelihood_test0b (negative points)')
runtest(loglikelihood_test1, 'loglikelihood_test1')
runtest(loglikelihood_test2, 'loglikelihood_test2')
runtest(loglikelihood_test3, 'loglikelihood_test3')


# In[36]:


# Autograder test cell- worth 1 point
# runs loglikelihood_testneg


# In[37]:


# Autograder test cell- worth 1 point
# runs loglikelihood_test0


# In[38]:


# Autograder test cell- worth 1 point
# runs loglikelihood_test0a


# In[39]:


# Autograder test cell- worth 1 point
# runs loglikelihood_test0b


# In[40]:


# Autograder test cell- worth 1 point
# runs loglikelihood_test1


# In[41]:


# Autograder test cell- worth 1 point
# runs loglikelihood_test2


# In[42]:


# Autograder test cell- worth 1 point
# runs loglikelihood_test3


# <h3>Part Four: Naive Bayes Prediction [Graded]</h3>
# 
# 
# <p>Observe that for a test point $\mathbf{x}_{test}$, we should classify it as positive if the log ratio $\log\left(\frac{P(y=1 | \mathbf{x} = \mathbf{x}_{test})}{P(y=-1|\mathbf{x} = \mathbf{x}_{test})}\right) > 0$ and negative otherwise. Implement the <b><code>naivebayes_pred</code></b> by first calculating the log ratio $\log\left(\frac{P(y=1 | \mathbf{x} = \mathbf{x}_{test})}{P(y=-1|\mathbf{x} = \mathbf{x}_{test})}\right)$ for each test point in $\mathbf{x}_{test}$ using Bayes' rule and predict the label of the test points by looking at the log ratio. When calculating the log likelihood, think carefully how you can use the fact $\log \left(\frac{a}{b}\right) = \log{a} - \log{b}$ to simplify your calculations.
# </p>
# 
# 
# 

# In[43]:


def naivebayes_pred(pos, neg, posprob, negprob, X_test):
    """
    naivebayes_pred(pos, neg, posprob, negprob, X_test) returns the prediction of each point in X_test
    
    Input:
        pos: class probability for the negative class
        neg: class probability for the positive class
        posprob: conditional probabilities for the positive class (d)
        negprob: conditional probabilities for the negative class (d)
        X_test : features (nxd)
    
    Output:
        prediction of each point in X_test (n)
    """
    n, d = X_test.shape
    
    # YOUR CODE HERE
    ratio1 = loglikelihood_grader(posprob, negprob, X_test, np.ones(n)) - loglikelihood_grader(posprob, negprob, X_test, -np.ones(n))
    ratio2 = np.log(pos) - np.log(neg)
    loglikelihood_ratio = ratio1 + ratio2
  
    prediction = - np.ones(n)
    prediction[loglikelihood_ratio > 0] = 1
    return prediction


# In[44]:


# The following tests check that your implementation of naivebayes_pred returns only 1s and -1s (test 1), and that it returns the same predicted values as the correct implementation for three different datasets (tests 2-4)

X,Y = genTrainFeatures_grader(128)
posY, negY = naivebayesPY_grader(X, Y)

# check whether the predictions are +1 or neg 1
def naivebayes_pred_test1():
    preds = naivebayes_pred(posY, negY, posprobXY, negprobXY, X)
    return np.all(np.logical_or(preds == -1 , preds == 1))

def naivebayes_pred_test2():
    naivebayesPXY_grader(X, Y)
    x_test = np.array([[0,1],[1,0]])
    preds = naivebayes_pred_grader(posY, negY, posprobXY[:2], negprobXY[:2], x_test)
    student_preds = naivebayes_pred(posY, negY, posprobXY[:2], negprobXY[:2], x_test)
    acc = analyze_grader("acc", preds, student_preds)
    return np.abs(acc - 1) < 1e-5

def naivebayes_pred_test3():
    x_test = np.array([[1,0,1,0,1,1], 
        [0,0,1,0,1,1], 
        [1,0,0,1,1,1], 
        [1,1,0,0,1,1]])
    naivebayesPXY_grader(X, Y)
    preds = naivebayes_pred_grader(posY, negY, posprobXY[:6], negprobXY[:6], x_test)
    student_preds = naivebayes_pred(posY, negY, posprobXY[:6], negprobXY[:6], x_test)
    acc = analyze_grader("acc", preds, student_preds)
    return np.abs(acc - 1) < 1e-5

def naivebayes_pred_test4():
    x_test = np.array([[1,1,1,1,1,1], 
        [0,0,1,0,0,0], 
        [1,1,0,1,1,1], 
        [0,1,0,0,0,1], 
        [0,1,1,0,1,1], 
        [1,0,0,0,0,1], 
        [0,1,1,0,1,1]])
    naivebayesPXY_grader(X, Y)
    preds = naivebayes_pred_grader(posY, negY, posprobXY[:6], negprobXY[:6], x_test)
    student_preds = naivebayes_pred(posY, negY, posprobXY[:6], negprobXY[:6], x_test)
    acc = analyze_grader("acc", preds, student_preds)
    return np.abs(acc - 1) < 1e-5

runtest(naivebayes_pred_test1, 'naivebayes_pred_test1')
runtest(naivebayes_pred_test2, 'naivebayes_pred_test2')
runtest(naivebayes_pred_test3, 'naivebayes_pred_test3')
runtest(naivebayes_pred_test4, 'naivebayes_pred_test4')


# In[ ]:


# Autograder test cell- worth 1 point
# runs naivebayes_pred_test1


# In[ ]:


# Autograder test cell- worth 1 point
# runs naivebayes_pred_test2


# In[ ]:


# Autograder test cell- worth 1 points
# runs naivebayes_pred_test3


# In[ ]:


# Autograder test cell- worth 1 points
# runs naivebayes_pred_test4


# You can now test your code with the following interactive name classification script:

# In[ ]:


DIMS = 128
print('Loading data ...')
X,Y = genTrainFeatures(DIMS)
print('Training classifier ...')
pos, neg = naivebayesPY(X, Y)
posprob, negprob = naivebayesPXY(X, Y)
error = np.mean(naivebayes_pred(pos, neg, posprob, negprob, X) != Y)
print('Training error: %.2f%%' % (100 * error))

while True:
    print('Please enter a baby name>')
    yourname = input()
    if len(yourname) < 1:
        break
    xtest = name2features(yourname,B=DIMS,LoadFile=False)
    pred = naivebayes_pred(pos, neg, posprob, negprob, xtest)
    if pred > 0:
        print("%s, I am sure you are a baby boy.\n" % yourname)
    else:
        print("%s, I am sure you are a baby girl.\n" % yourname)


# <h2> Challenge: Feature Extraction</h2>
# 
# <p>Let's test how well your Na&iuml;ve Bayes classifier performs on a secret test set. If you want to improve your classifier modify <code>name2features2</code> below.   The automatic reader will use your Python script to extract features and train your classifier on the same names training set by calling the function with only one argument--the name of a file containing a list of names.  The given implementation is the same as the given <code>name2features</code> above.
# </p>
#   

# In[ ]:


def hashfeatures(baby, B, FIX):
    v = np.zeros(B)
    for m in range(FIX):
        featurestring = "prefix" + baby[:m]
        v[hash(featurestring) % B] = 1
        featurestring = "suffix" + baby[-1*m:]
        v[hash(featurestring) % B] = 1
    return v

def name2features2(filename, B=128, FIX=3, LoadFile=True):
    """
    Output:
        X : n feature vectors of dimension B, (nxB)
    """
    # read in baby names
    if LoadFile:
        with open(filename, 'r') as f:
            babynames = [x.rstrip() for x in f.readlines() if len(x) > 0]
    else:
        babynames = filename.split('\n')
    n = len(babynames)
    X = np.zeros((n, B))
    for i in range(n):
        X[i,:] = hashfeatures(babynames[i], B, FIX)
        
    # YOUR CODE HERE
    raise NotImplementedError()
    return X


# In[ ]:


# Autograder test cell- competition


# (Hint: You should be able to get >80% accuracy just by changing some of the default hyperparameters in the function argument.  If you'd like to try something more sophisticated, you can add to `name2features2`)
# 
# <h4>Credits</h4>
#  The name classification idea originates from <a href="http://nickm.com">Nick Montfort</a>.
