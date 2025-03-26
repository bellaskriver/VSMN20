import numpy as np

def main():
    A = np.matrix([[1, 2, 3], [3, 7, 4], [2, 5, 3]])
    b = np.matrix([[1], [2], [3]])
    calculations(A, b)

def calculations(A, b):
    print('A invers:', np.linalg.inv(A))
    print(' Matris multiplikation A*b:', A @ b)
    print('A transponat:', np.transpose(A))
    print('x från Ax=b:',np.linalg.solve (A, b))

main()

