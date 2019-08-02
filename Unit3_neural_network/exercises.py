import numpy as np

def f(z):
    if z > 0:
        return z
    else:
        return 0


x = np.array([1,0])
w_0 = -3
w = np.array([1, -1]).reshape(2, -1)
z = np.dot(np.transpose(x), w) + w_0
print(z)
print(z.item())
print(f(z.item()))

"""
Code for "Non-linear Activation Functions question" in 
 https://courses.edx.org/courses/course-v1:MITx+6.86x+1T2019/courseware/unit_3/lec9_neuralnets2/?child=first
"""

def linear(z):
    return 5 * z - 2


def ReLu(z):
    if z > 0:
        return z
    else:
        return 0

def tanhz(z):
    return np.tanh(z)


def yourself(z):
    return z


active_functions = (linear, ReLu, tanhz, yourself)
# Activation function
X = np.array([-1, -1, 1, -1, -1,1, 1, 1]).reshape(-1, 2)
print(X)
Y = np.array([1, -1, -1, 1]).reshape(-1, 1)
print(Y)


def getResults(x, y):
   w = np.array([1, -1, -1, 1]).reshape(2, -1)
   w0 = np.array([1, 1]).reshape(-1, 1)
   print(w)
   print(w0)

   for m in range(len(active_functions)):
        print("For active function ", active_functions[m])
        for i in range(X.shape[0]):
            xi = x[i, :]
            print("data point ", xi)
            xi.reshape([2, -1])
            fa = (np.dot(w, xi)).reshape(2, -1) + w0
            for j in range(fa.shape[0]):
                val = fa.item(j, 0)
                print(active_functions[m](val))
            print("y=", y[i])

getResults(X, Y)


"""
End of code for "Non-linear Activation Functions question" in 
 https://courses.edx.org/courses/course-v1:MITx+6.86x+1T2019/courseware/unit_3/lec9_neuralnets2/?child=first
"""


"""
Code for https://www.cnblogs.com/charlotte77/p/5629865.html
"""
print("Begin https://www.cnblogs.com/charlotte77/p/5629865.html")
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


# Initial values
w0 = np.array([0.15, 0.2, 0.25, 0.3]).reshape(-1, 2)
b0 = np.array([0.35, 0.35]).reshape(2, -1)
w1 = np.array([0.4, 0.45, 0.5, 0.55]).reshape(-1, 2)
b1 = np.array([0.6, 0.6]).reshape(2, -1)

init_param = ((w0, b0), (w1, b1))

# Input values
i = np.array([0.05, 0.1]).reshape(2, -1)
print(i)
# Desired output
o = np.array([0.01, 0.99]).reshape(2, -1)


# Forward
def layer_cal(input, layer):
    layer_param = init_param[layer]
    w = layer_param[0]
    print(w)
    b = layer_param[1]
    print(b)
    z = np.dot(w, input) + b
    ret = []
    for i in range(z.shape[0]):
        ret.append(sigmoid(z[i, 0]))
    return np.array(ret).reshape(2, -1)


ret1 = layer_cal(i, 0)
print(ret1)
ret2 = layer_cal(ret1, 1)
print(ret2)


print("End of https://www.cnblogs.com/charlotte77/p/5629865.html")
"""
End of code for https://www.cnblogs.com/charlotte77/p/5629865.html
"""
