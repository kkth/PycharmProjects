import numpy as np
y = np.array([1, -1, 1])
x = np.array([[-1, -1], [1, 0], [-1, 10]])
matrix = np.zeros((2,))
matrix_record = []
count = 0
for j in range(100):
    print(j)
    for i in range(len(y)):
        if y[i] * (np.dot(matrix.T, x[i])) <= 0:
            matrix = matrix + y[i] * x[i]
            matrix_record.append(matrix)
            count += 1
print(count)
print(matrix)
print(matrix_record)