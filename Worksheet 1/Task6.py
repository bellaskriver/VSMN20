import numpy as np
import matplotlib.pyplot as plt

def main():
    x = np.linspace(0, 2 * np.pi, 1000)
    y = function(x)
    plt.plot(x, y)
    plt.show()

def function(x):
        return np.sin(x ** 2)

main()