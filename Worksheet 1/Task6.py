import numpy as np
import matplotlib.pyplot as plt

def main():
    x = np.linspace(0, 2 * np.pi, 1000)
    y = function(x)
    plt.plot(x, y)
    plt.suptitle('Function: $y = sin(x^2)$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.autoscale(tight=True)
    plt.grid()
    plt.show()

def function(x):
        return np.sin(x ** 2)

main()