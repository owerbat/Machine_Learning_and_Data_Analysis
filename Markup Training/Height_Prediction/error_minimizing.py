import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def error(w0, w1, data):
    return sum((w0 + w1*data['Weight'] - data['Height'])**2)


def error_vector(w0, w1, data):
    result = 0.
    for i, item in enumerate(data['Weight']):
        result += (w0 + w1*item - data['Height'][i+1])**2
    return result


def first_try(data):
    data.plot(x='Weight', y='Height', kind='scatter')
    x = np.linspace(70, 180, 200)
    y = 60 + .05*x
    plt.plot(x, y, color='red')
    y = 50 + .16 * x
    plt.plot(x, y, color='orange')
    plt.show()


def show_plot(data, w0, w1):
    data.plot(x='Weight', y='Height', kind='scatter')
    x = np.linspace(70, 180, 200)
    y = w0 + w1*x
    plt.plot(x, y, color='orange')
    plt.show()


def w1_dependence(data):
    x = np.linspace(-100, 100, 100)
    y = np.zeros(len(x))
    for i, item in enumerate(x):
        y[i] = error(50, item, data)
    plt.plot(x, y)
    plt.show()


def w1_optimizing(data):
    opt = minimize_scalar(lambda w1: error(50., w1, data))
    return opt.x


def optimizing(data):
    opt = minimize(lambda w: error(w[0], w[1], data), x0=np.array([0., 0.]),
                   bounds=((-100., 100.), (-5., 5.)), method='L-BFGS-B')
    return opt.x


def show_error_plot(data):
    x = np.linspace(-5., 5., 100)
    y = np.linspace(-5., 5., 100)
    x, y = np.meshgrid(x, y)
    z = error_vector(x, y, data)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def main():
    data = pd.read_csv('weights_heights.csv', index_col='Index')
    w1_opt = w1_optimizing(data)
    # show_plot(data, 50., w1_opt)
    # show_error_plot(data)
    opt = optimizing(data)
    show_plot(data, opt[0], opt[1])


if __name__ == '__main__':
    main()

