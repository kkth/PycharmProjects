import numpy as np
# Begin of Question 1 (Neural Network)
beta = 1

X = np.array([3, 14]).reshape(2, -1)
print("X=", X)
W = np.array([1, 0, -1, 0, 1, -1, -1, 0, -1, 0, -1, -1]).reshape(4, -1)
print("W=", W)
V = np.array([1, 1, 1, 1, 0, -1, -1, -1, -1, 2]).reshape(2, -1)
print("V=", V)


def ReLu(z):
    if z > 0:
        return z
    else:
        return 0


def softmax(u1, u2):
    o1 = np.exp(beta * ReLu(u1)) / (np.exp(beta*ReLu(u1)) + np.exp(beta * ReLu(u2)))
    o2 = np.exp(beta * ReLu(u2)) / (np.exp(beta*ReLu(u1)) + np.exp(beta * ReLu(u2)))
    return o1, o2


def main():
    W1 = W[:, 0: 2]
    print("W1=", W1)
    W0 = W[:, 2].reshape(-1, 1)
    print("W0=", W0)

    z = np.dot(W[:, 0:2], X)
    print("z=", z)
    z += W0
    print("z=", z)
    fz = np.array([ReLu(zi) for zi in z[:, 0]]).reshape(-1, 1)
    print("fz=", fz)

    u = np.dot(V[:, 0:4], fz) + V[:, 4].reshape(-1, 1)
    print("u=", u)
    fu = np.array([ReLu(ui) for ui in u[:, 0]]).reshape(-1, 1)[:, 0]
    print("fu=", fu)

    ret = softmax(fu[0], fu[1])

    return ret


print(main())

print(softmax(1, 1))
print(softmax(0, 2))
print(softmax(3, 0))

# End of Question 1 (Neural Network)

# Begin of Question 2 (LSTM)
# End of Question 2 (LSTM)
