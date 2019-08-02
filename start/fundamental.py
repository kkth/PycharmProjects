import numpy as np
import math


##Array operations

def genArraies():
    # 1-d array
    arr1 = np.array((1, 3, 5, 7))
    print(arr1)
    print(arr1.size)
    print(arr1.shape)

    arr1a = np.array([1, 3, 5, 7])
    print(arr1a)
    print(arr1a.size)
    print(arr1a.shape)

    arr1b = np.array([[1, 3, 5, 7]])
    print(arr1b)
    print(arr1b.size)
    print(arr1b.shape)

    # 2-d array
    arr2 = np.array([(1, 3, 5, 7), (2, 4, 6, 8)])
    print(arr2)
    print(arr2.size)
    print(arr2.shape)

    # 2-d array
    arr3 = np.array([[1, 3, 5, 7], [2, 4, 6, 8]])
    print(arr3)
    print(arr3.size)
    print(arr3.shape)

    # 2-d array
    arr4 = np.array(([1, 3, 5, 7], [2, 4, 6, 8]))
    print(arr4)
    print(arr4.size)
    print(arr4.shape)


def matrixMultiply():
    m1 = np.mat([[1, 2, 3], [4, 5, 6]])
    m2 = np.mat([[4, 5], [1, 8], [2, 3]])

    result1 = m1 * m2
    print(result1)

    v1 = [0, 1]
    v2 = [1, 0]

    try:
        result2 = v1 * v2
        print(result2)
    except:
        print("exception!")

    result2 = np.dot(v1, v2)
    print(result2)

    angle = math.acos(result2/(np.linalg.norm(v1) * np.linalg.norm(v2)))
    angle = angle/np.pi
    print(angle, "Pi")

    result3 = np.multiply(m1, m2.T)
    print(result3)



def ndArries():
    a = np.arange(15).reshape(3, 5)
    print(a)
    print(a.shape)
    print(a.ndim)
    print(a.size)
    print(type(a))


# matrixMultiply()
# ndArries()

print("numpy array operation")
# generate an np array with 24 elements=[0,23]
a = np.arange(0, 24, 1)
print("a=", a)

# reshape it to 3*8 2-d array
b = a.reshape(3, -1)
print("b=", b)
# print(a[:3])
print(b[(0, 2), ])


# reshape it to 4*6 2-d array
c = a.reshape(4, -1)
print("c=", c)
print("right corner=", c[0:3, 3:6])
print("corner lines=", c[[0, 1, 2], [5, 4, 3]])
print("column=", c[0:3, 2])
print("column=", c[0:3, ])
print("column=", c[0:3, :])




