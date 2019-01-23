import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def show_histogramm(data):
    data.plot(y='Height', kind='hist')
    data.plot(y='Weight', kind='hist', facecolor='red')
    plt.show()


def make_bmi(height_inch, weight_pound):
    METER_TO_INCH, KILO_TO_POUND = 39.37, 2.20462
    return (weight_pound / KILO_TO_POUND) / \
           (height_inch / METER_TO_INCH) ** 2


def show_pair_plot(data):
    sns.pairplot(data)
    plt.show()


def weight_category(weight):
    if weight < 120:
        return 1
    elif weight >= 150:
        return 3
    else:
        return 2


def show_box_plot(x, y, data):
    sns.boxplot(x=data[x], y=data[y])
    plt.show()


def show_plot(x, y, data):
    data.plot(x=x, y=y, kind='scatter')
    plt.show()


def main():
    data = pd.read_csv('weights_heights.csv', index_col='Index')
    # show_histogramm(data)
    data['BMI'] = data.apply(lambda row: make_bmi(row['Height'], row['Weight']), axis=1)
    # show_pair_plot(data)
    data['W_Category'] = data['Weight'].apply(weight_category)
    # show_box_plot('W_Category', 'Height', data)
    show_plot('Weight', 'Height', data)


if __name__ == '__main__':
    main()
