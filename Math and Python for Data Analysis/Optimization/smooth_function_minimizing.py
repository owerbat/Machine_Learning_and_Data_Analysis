from scipy.optimize import minimize
from matplotlib import pyplot as plt
from numpy import arange, sin, exp


def f(x):
    return sin(x/5.)*exp(x/10.) + 5.*exp(-x/2.)


def main():
    print(minimize(f, 2, method='BFGS'))
    print('\n\n\n')
    print(minimize(f, 30, method='BFGS'))

    x = arange(0., 30., 0.1)
    y = f(x)
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    main()
