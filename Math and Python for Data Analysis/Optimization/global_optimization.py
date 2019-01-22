from scipy.optimize import differential_evolution
from numpy import sin, exp


def f(x):
    return sin(x/5.)*exp(x/10.) + 5.*exp(-x/2.)


def main():
    print(differential_evolution(f, [(1., 30.)]))


if __name__ == '__main__':
    main()
