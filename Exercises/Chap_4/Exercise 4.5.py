# 5. We now examine the differences between LDA and QDA.
#
# (a) If the Bayes decision boundary is linear, do we expect LDA or
# QDA to perform better on the training set? On the test set?

# QDA on training as it is more flexible it will yield to a closer fit.
# LDA for test as QDA may overfit the linearity of the bayes decision boundary.


# (b) If the Bayes decision boundary is non-linear, do we expect LDA
# or QDA to perform better on the training set? On the test set?

# QDA on both.

# (c) In general, as the sample size n increases, do we expect the test
# prediction accuracy of QDA relative to LDA to improve, decline,
# or be unchanged? Why?

# Since QDA is a duadratic model, more data should improve the model faster than LDA

# (d) True or False: Even if the Bayes decision boundary for a given
# problem is linear, we will probably achieve a superior test error
# rate using QDA rather than LDA because QDA is flexible
# enough to model a linear decision boundary. Justify your answer.

# False, QDA will overfit by finding a different variance for each class when in reality the variance for each class are the same