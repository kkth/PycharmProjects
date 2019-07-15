import numpy as np

def hinge_loss(z):
   if z>=1:
       return 0
   else:
       return 1-z


def square_loss(z):
   return (z**2)/2


def cal_empirical_risk(loss):
    theta = [0, 1, 2]
    training_features = ([1, 0, 1], [1, 1, 1], [1, 1, -1], [-1, 1, 1])
    training_labels = (2, 2.7, -0.7, 2)
    sum = 0
    for i in range(len(training_labels)):
       sum += loss(training_labels[i] - np.dot(training_features[i], theta))

    return sum/len(training_labels)


#Emperical risk question 1
print(cal_empirical_risk(hinge_loss))

#Emperical risk question2
print(cal_empirical_risk(square_loss))

