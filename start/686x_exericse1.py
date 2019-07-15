import numpy as np;
def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    A = np.mat(np.random.rand(n,1));
    return A;
    #raise NotImplementedError

print(randomization(5));

def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    #Your code here
    A = np.mat(np.random.rand(h,w));
    B = np.mat(np.random.rand(h,w));
    s = A+B;


    return A,B,s;

print(operations(2,3));

print("==========>");
i=0;
for i in range(1,11,3):
    print(i);
    i=i+1;
    print(i);


print("finish!");

def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """
    s = np.linalg.norm(A+B)

    return s;

def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    #Your code here
    m1 = np.mat(inputs)
    m2 = np.mat(weights)
    val = m1.T*m2
    print(val)
    inner = np.tanh(val)
    out = np.array(inner)
    return out;


theout = neural_network(np.array([[2],[1]]),np.array([[3],[4]]))
print(theout)

