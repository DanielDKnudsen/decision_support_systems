# We now review k-fold cross-validation.
#
# (a) Explain how k-fold cross-validation is implemented.

# K-fold CV works by taking the dataset given and randomly splitting it into k non-overlapping datasets.
# You can shuffle the data first and then just split at regular intervals. Train K models.
# For each model, use the kth region as the validation set and build on the other k-1 sets.
# Take the mean of the k errors found to estimate the true test error.

# (b) What are the advantages and disadvantages of k-fold crossvalidation relative to:

# i. The validation set approach?
# Advantage to validation set is that there are more test sets to validate on which should reduce the bias of what the overall error actually is.
# Variance should also decrease as the validation set approach is just one split of the data and that split could not represent the test data well.
# Disadvantage is training more models.


# ii. LOOCV?
# Advantage to LOOCV is a decrease in variance as the k models are not as highly correlated as the each LOOCV model is.
# Also, K-folds is computationally less expensive.
