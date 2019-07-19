import numpy as np
from sklearn.svm import LinearSVC


### Functions for you to fill in ###

def one_vs_rest_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for binary classifciation

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (0 or 1) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (0 or 1) for each test data point
    """
    clf = LinearSVC(random_state=0, C=0.1)
    clf.fit(train_x, train_y)
    ret = clf.predict(test_x)
    return ret


def multi_class_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for multiclass classifciation using a one-vs-rest strategy

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
    """
    clf = LinearSVC(random_state=0, C=0.1)
    clf.fit(train_x, train_y)
    ret = clf.predict(test_x)
    return ret

### You can easily copy the last line of "compute_test_error_linear" function and change the name of the variables.
### or
### Hello, implement zero-one loss in the function.
### You can also import the function from the sklearn lib: from sklearn.metrics import zero_one_loss
def compute_test_error_svm(test_y, pred_test_y):
    return 1 - np.mean(test_y == pred_test_y)
