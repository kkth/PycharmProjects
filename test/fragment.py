import numpy as np
from sklearn.svm import LinearSVC


def closed_form(X, Y, lambda_factor):
    """
    Computes the closed form solution of linear regression with L2 regularization

    Args:
        X - (n, d + 1) NumPy array (n datapoints each with d features plus the bias feature in the first dimension)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        lambda_factor - the regularization constant (scalar)
    Returns:
        theta - (d + 1, ) NumPy array containing the weights of linear regression. Note that theta[0]
        represents the y-axis intercept of the model and therefore X[0] = 1
    """
    the_i = np.eye(X.shapep[1])
    return (np.transpose(X)*X + lambda_factor*the_i).I*np.transpose(X)*Y



train_x = [[0.12776022, 0.19765322],
 [0.59225675, 0.37678803],
 [0.13230964, 0.7247785 ],
 [0.42213388, 0.38857987],
 [0.73405887, 0.32101689],
 [0.17749338, 0.02589688],
 [0.03903776, 0.09311586],
 [0.21799272, 0.19997887],
 [0.33638151, 0.99184474],
 [0.45105624, 0.20921742]]
train_y = [0, 0, 1, 0, 1, 1, 0, 0, 1, 1]
test_x = [[0.22373521, 0.75253547],
[0.55796528, 0.27457326],
[0.60261045, 0.2933018 ],
[0.30765521, 0.43781043],
[0.17800156, 0.68400566],
[0.5144091 , 0.99165093],
[0.08753217, 0.01916129],
[0.12347731, 0.29519974],
[0.53449037, 0.33921037],
[0.78679386, 0.81747222],
[0.79803498, 0.13772139],
[0.01215544, 0.00234388],
[0.26409408, 0.03648458],
[0.61676343, 0.97747782]]


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



#print(one_vs_rest_svm(train_x,train_y,test_x))




def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    n = X.shape[0]
    k = theta.shape[0]
    H = []
    for ni in range(n):
        Xi = X[ni, :]
        exp_dot_array = []
        c = 0

        #get the c for this x data point
        for ti in range(k):
            thetai = theta[ti, :]
            cur_c = np.dot(np.dot(thetai, Xi), 1 / temp_parameter)
            exp_dot_array.append(cur_c)
            if cur_c > c:
                c = cur_c

        x_array = []

        for ti in range(k):
            exp_dot_array[ti] = exp_dot_array[ti]- c

        for ti in range(k):
            x_array.append(np.exp(exp_dot_array[ti]))

        sum = 0
        for ti in range(k):
            sum = sum + x_array[ti]

        hn = x_array
        hn = hn / sum
        H.append(hn)

    HT = np.transpose(H)
    Harray = np.asarray(HT)
    return Harray



X = np.array([[1, 2], [3, 5], [1, 6], [2, 4]])
theta = np.array([[0, 1], [2, 3]])
t = 1.2
print(compute_probabilities(X, theta, t))


print(np.log10(1/7))


def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    itemp=1./temp_parameter
    num_examples = X.shape[0]
    num_labels = theta.shape[0]
    probabilities = compute_probabilities(X, theta, temp_parameter)
    # M[i][j] = 1 if y^(j) = i and 0 otherwise.
    M = sparse.coo_matrix(([1]*num_examples, (Y,range(num_examples))), shape=(num_labels,num_examples)).toarray()
    non_regularized_gradient = np.dot(M-probabilities, X)
    non_regularized_gradient *= -itemp/num_examples
    return theta - alpha * (non_regularized_gradient + lambda_factor * theta)



