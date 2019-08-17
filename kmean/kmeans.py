from copy import deepcopy
import numpy as np # linear algebra

# Set three centers, the model should predict similar results
center_1 = np.array([-5, 2])
center_2 = np.array([0, -6])

# Generate random data and center it to the three centers
data_1 = np.array([])
data_2 = np.array()
data_3 = np.array()

data = np.concatenate((data_1, data_2, data_3), axis = 0)