import numpy as np;


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
    # Your code here


def run_hinge_loss_full():
    m1 = np.mat([[1, 2, 3], [4, 1, 5], [3, 6, 8], [3, 6, 2]])
    lab = [1, -1, -1, 1]
    t = [1, 9, 4]
    t0 = 0.5
    print(hinge_loss_full(m1, lab, t, t0))


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
        # for i in get_order(feature_matrix.shape[0]):
        for i in range(0, rc[0]):
            # Your code here
            ite = perceptron_single_step_update(np.squeeze(np.asarray(feature_matrix[i, :])), labels[i], theta, theta_0)
            theta = ite[0]
            theta_0 = ite[1]
    ret.append(theta)
    ret.append(theta_0)
    return ret


def run_perceptron():
    feature_matrix = np.mat([[-0.23535179, 0.06280089, 0.13733581, 0.02678303, -0.35521418, -0.47064226, 0.34671033,
                              -0.33245066, -0.21902541, 0.24041887],
                             [-0.18895702, -0.39473969, -0.39201588, -0.49886023, 0.15727126, 0.40587019, 0.25617022,
                              -0.256695, -0.06175288, -0.27950495],
                             [0.3882822, 0.38009288, -0.19456588, 0.15446576, -0.3033161, 0.35938678, -0.27703366,
                              -0.44149021, -0.45248198, 0.32797633],
                             [-0.34217736, -0.31863244, -0.14678309, 0.38695453, 0.28717654, 0.25732277, 0.25008403,
                              -0.15954429, -0.21926052, -0.11126569],
                             [-0.15477171, -0.39609997, -0.17679138, -0.45808239, -0.3631214, 0.00388449, -0.49911652,
                              0.03295782, 0.43805333, -0.18929941]])
    labels = [-1, 1, 1, 1, -1]
    T = 5
    ret = perceptron(feature_matrix, labels, T)
    print(ret)


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
        # for i in get_order(feature_matrix.shape[0]):
        for i in range(0, rc[0]):
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


def run_average_perceptron():
    feature_matrix = np.mat([[ 0.11509094, 0.41337047, 0.41657219, 0.30912177, -0.20642844, 0.0657247 , 0.09510344, 0.46457421, -0.21455199, -0.40272524]
    , [ 0.15177321, 0.36232757, -0.00461355, -0.49464711, -0.42801048, -0.07516229 , -0.0656667, -0.12668676, -0.48596048, 0.23253797]
    , [ 0.08156486, 0.36887038, 0.47620553, -0.43505672, 0.21717488, 0.3340048 ,  0.18740848, 0.25761652, 0.00859504, 0.14148984]
    , [-0.10515537, -0.09040305, 0.10901287, 0.20407979, 0.06010366, -0.271336 ,  0.19466121, -0.16540974, 0.00827818, 0.45744145]
    , [-0.2763828, -0.29177211, 0.35735031, -0.26406558, 0.04554595, -0.31520974,  -0.13182654, -0.03309603, 0.37555702, 0.05427657]])
    labels = [-1, 1, 1, 1, 1]
    T = 5
    ret = average_perceptron(feature_matrix, labels, T)
    print(ret)


"""
Code for Pegasos Algorithm
"""
def pegasos_single_step_update_origin(
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
        ret.append((1 - L * eta) * current_theta + eta * label * feature_vector)
        ret.append(current_theta_0 + eta * label)
    else:
        ret.append((1 - L * eta) * current_theta)
        ret.append(current_theta_0)
    return ret


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
        #for i in get_order(feature_matrix.shape[0]):
        for i in range(0, rc[0]):
            # Your code here
            ite = pegasos_single_step_update(np.array(np.squeeze(np.asarray(feature_matrix[i, :]))), labels[i], L,
                                             1/np.sqrt(count),
                                             np.array(theta), theta_0)
            theta = ite[0]
            theta_0 = ite[1]
            count = count + 1
    ret.append(theta)
    ret.append(theta_0)
    return ret


def run_pegasos():
        feature_matrix=np.mat([[0.1837462, 0.29989789, -0.35889786, -0.30780561, -0.44230703, -0.03043835, 0.21370063, 0.33344998, -0.40850817, -0.13105809],
        [0.08254096, 0.06012654, 0.19821234, 0.40958367, 0.07155838, -0.49830717, 0.09098162, 0.19062183, -0.27312663, 0.39060785],
        [-0.20112519, -0.00593087, 0.05738862, 0.16811148, -0.10466314, -0.21348009, 0.45806193, -0.27659307, 0.2901038, -0.29736505],
        [-0.14703536, -0.45573697, -0.47563745, -0.08546162, -0.08562345, 0.07636098, -0.42087389, -0.16322197, -0.02759763, 0.0297091],
        [-0.18082261, 0.28644149, -0.47549449, -0.3049562, 0.13967768, 0.34904474, 0.20627692, 0.28407868, 0.21849356, -0.01642202]])
        labels=[-1, -1, -1, 1, -1]
        T=10
        L=0.1456692551041303

        ret = pegasos(feature_matrix,labels,T,L)
        print(ret)


run_pegasos()



def test():
    arr = [1,2,3]
    mul = 4
    print(arr*mul)
    print(mul*arr)
    print(np.dot(arr,mul))



#test()