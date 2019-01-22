from scipy.optimize import minimize, differential_evolution
from matplotlib import pyplot as plt
from numpy import array, arange, sin, exp


def h(x):
    return int(sin(x / 5.) * exp(x / 10.) + 5. * exp(-x / 2.))


def main():
    print(minimize(h, 29., method='BFGS'))
    print('\n\n\n')
    print(differential_evolution(h, [(1., 30.)]))

    x = arange(0., 31., 0.1)
    y = array([h(i) for i in x])
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    main()
