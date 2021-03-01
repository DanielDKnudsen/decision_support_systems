# 4. When the number of features p is large, there tends to be a deterioration
# in the performance of KNN and other local approaches that
# perform prediction using only observations that are near the test observation
# for which a prediction must be made. This phenomenon is
# known as the curse of dimensionality, and it ties into the fact that
# parametric approaches often perform poorly when p is large. We mension ality
# will now investigate this curse.


# (a) Suppose that we have a set of observations, each with measurements
# on p = 1 feature, X. We assume that X is uniformly
# (evenly) distributed on [0, 1]. Associated with each observation
# is a response value. Suppose that we wish to predict a test observation’s
# response using only observations that are within 10% of
# the range of X closest to that test observation. For instance, in
# order to predict the response for a test observation with X = 0.6,
# we will use observations in the range [0.55, 0.65]. On average,
# what fraction of the available observations will we use to make
# the prediction?

# It's easiest to think in terms of each X in a range of 0 - 100 a) 10 / 100 = 10%

# (b) Now suppose that we have a set of observations, each with
# measurements on p = 2 features, X1 and X2. We assume that
# (X1,X2) are uniformly distributed on [0, 1] × [0, 1]. We wish to
# predict a test observation’s response using only observations that
# are within 10% of the range of X1 and within 10% of the range
# of X2 closest to that test observation. For instance, in order to
# predict the response for a test observation with X1 = 0.6 and
# X2 = 0.35, we will use observations in the range [0.55, 0.65] for
# X1 and in the range [0.3, 0.4] for X2. On average, what fraction
# of the available observations will we use to make the prediction?

# 10 x 10 / (100 x 100) = 1%

# (c) Now suppose that we have a set of observations on p = 100 features.
# Again the observations are uniformly distributed on each
# feature, and again each feature ranges in value from 0 to 1. We
# wish to predict a test observation’s response using observations
# within the 10% of each feature’s range that is closest to that test
# observation. What fraction of the available observations will we
# use to make the prediction?
# $10^{-100}$. The fraction of nearest neighbors is $10^{-p}$

# (d) Using your answers to parts (a)–(c), argue that a drawback of
# KNN when p is large is that there are very few training observations
# “near” any given test observation.
# even in 2 dimensions only 1% of neighbors will be within 5% on either side


# (e) Now suppose that we wish to make a prediction for a test observation
# by creating a p-dimensional hypercube centered around
# the test observation that contains, on average, 10% of the training
# observations. For p = 1, 2, and 100, what is the length of
# each side of the hypercube? Comment on your answer.
# 10% for p=1. Generalizing we get $.1^{1/p}$