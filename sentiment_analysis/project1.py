from string import punctuation, digits
import numpy as np
import random

# Part I


#pragma: coderesponse template
def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices
#pragma: coderesponse end


#pragma: coderesponse template
def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    # Your code here
    val = (np.dot(feature_vector, theta) + theta_0) * label
    if val >= 1:
        return 0
    else:
        return 1 - val
#pragma: coderesponse end


#pragma: coderesponse template
def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    # Your code here
    rows = len(feature_matrix)
    results = []
    for r in range(0, rows):
        # print(r)
        theRow = feature_matrix[r, :]
        # print(theRow)
        theLabel = labels[r]
        # print(theLabel)
        val = (np.dot(theRow, theta) + theta_0) * theLabel
        # print("===>", val)
        if val >= 1:
            val = 0
        else:
            val = 1 - val
        results.append(val)
    # print(results)
    final = np.average(results)
    # final = final[0, 0]
    # print(final)
    return final
#pragma: coderesponse end


#pragma: coderesponse template
def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code here
    ret = []
    val = (np.dot(feature_vector, current_theta) + current_theta_0) * label
    if val <= 0:
        ret.append(current_theta + label * feature_vector)
        ret.append(current_theta_0 + label)
    else:
        ret.append(current_theta)
        ret.append(current_theta_0)
    return ret
#pragma: coderesponse end


#pragma: coderesponse template
def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    # Your code here
    theta = []
    ret = []
    rc = feature_matrix.shape
    for i in range(0, rc[1]):
        theta.append(0)
    theta_0 = 0

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
        #for i in range(0, rc[0]):
            # Your code here
            ite = perceptron_single_step_update(np.squeeze(np.asarray(feature_matrix[i, :])), labels[i], theta, theta_0)
            theta = ite[0]
            theta_0 = ite[1]
    ret.append(theta)
    ret.append(theta_0)
    return ret
#pragma: coderesponse end


#pragma: coderesponse template
def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])


    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    # Your code here
    theta = []
    theta_sum = []

    rc = feature_matrix.shape
    for i in range(0, rc[1]):
        theta.append(0)
        theta_sum.append(0)
    theta_0 = 0
    theta_0_sum = 0

    final_result = []
    last = []

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
        #for i in range(0, rc[0]):
            # Your code here
            ite = perceptron_single_step_update(np.squeeze(np.asarray(feature_matrix[i, :])), labels[i], theta, theta_0)

            theta = ite[0]
            theta_0 = ite[1]

            theta_sum = theta_sum + theta
            theta_0_sum = theta_0_sum + theta_0

    final_result.append(theta_sum/(T*rc[0]))
    final_result.append(theta_0_sum/(T*rc[0]))
    last.append(theta)
    last.append(theta_0)

    return final_result
#pragma: coderesponse end


#pragma: coderesponse template
def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code here
    ret = []
    val = (np.dot(feature_vector, current_theta) + current_theta_0) * label
    if val <= 1:
        ret.append(np.dot((1 - L * eta), current_theta) + np.dot(eta * label, feature_vector))
        ret.append(current_theta_0 + eta * label)
    else:
        ret.append(np.dot((1 - L * eta), current_theta))
        ret.append(current_theta_0)
    return ret
#pragma: coderesponse end


#pragma: coderesponse template
def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    # Your code here
    theta = []
    ret = []
    rc = feature_matrix.shape
    for i in range(0, rc[1]):
        theta.append(0)
    theta_0 = 0
    count = 1

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
        #for i in range(0, rc[0]):
            # Your code here
            ite = pegasos_single_step_update(np.array(np.squeeze(np.asarray(feature_matrix[i, :]))), labels[i], L,
                                             1 / (count ** 0.5),
                                             np.array(theta), theta_0)
            theta = ite[0]
            theta_0 = ite[1]
            count = count + 1
    ret.append(theta)
    ret.append(theta_0)
    return ret
#pragma: coderesponse end

# Part II


#pragma: coderesponse template
def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    # Your code here
    row = feature_matrix.shape[0]
    ret = []

    for i in range(row):
        data_point = np.squeeze(feature_matrix[i,:])
        val = np.dot(data_point,theta) + theta_0
        if val>0:
            ret.append(1)
        else:
            ret.append(-1)
    return np.asarray(ret)

#pragma: coderesponse end


#pragma: coderesponse template
def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.
    The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """
    # Your code here
    ret = []
    thetas = classifier(train_feature_matrix, train_labels, **kwargs)
    theta = thetas[0]
    theta_0 = thetas[1]
    #print('theta for', classifier, 'is', ', '.join(map(str, list(thetas[0]))))
    #print('theta_0 for', classifier, 'is', str(thetas[1]))

    model_results1 = classify(train_feature_matrix, theta, theta_0)
    model_results2 = classify(val_feature_matrix, theta, theta_0)

    acc1 = accuracy(train_labels, model_results1)
    acc2 = accuracy(val_labels, model_results2)

    ret.append(acc1)
    ret.append(acc2)

    return tuple(ret)

#pragma: coderesponse end


#pragma: coderesponse template
def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()
#pragma: coderesponse end


#pragma: coderesponse template
def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    stopwords = []
    file = open("stopwords.txt")
    while 1:
        line = file.readline()
        if not line:
            break
        else:
            stopwords.append(line.replace("\r\n", ""))
    file.close()


    # Your code here
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in stopwords:
                if word not in dictionary:
                    dictionary[word] = len(dictionary)
    return dictionary
#pragma: coderesponse end


#pragma: coderesponse template
def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9
    """
    # Your code here

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                #Original(Changed by Kun)     feature_matrix[i, dictionary[word]] = 1
                feature_matrix[i, dictionary[word]] = 1 + feature_matrix[i, dictionary[word]]
    return feature_matrix
#pragma: coderesponse end


#pragma: coderesponse template
def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
#pragma: coderesponse end
