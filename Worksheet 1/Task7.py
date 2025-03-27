import numpy as np

def main():
    A = np.matrix([[1, 2, 3], [3, 7, 4], [2, 5, 3]])
    b = np.matrix([[1], [2], [3]])
    calculations(A, b)

def calculations(A, b):
    print()
    print('Inversen av A är:')
    print(np.linalg.inv(A))
    print()
    print('Matrismultiplikationen A*b ger:')
    print(A @ b)
    print()
    print('Transponatet av A är:')
    print(np.transpose(A))
    print()
    print('Beräknat x från ekvationssystemet Ax=b är:')
    print(np.linalg.solve(A, b))
    print()

main()

