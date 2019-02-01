from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import naive_bayes

import pandas as pd


def get_error(classifier, train_data, test_data, train_labels, test_labels):
    clf = classifier()
    clf.fit(train_data, train_labels)
    # return metrics.mean_squared_error(test_labels, clf.predict(test_data))
    return metrics.accuracy_score(test_labels, clf.predict(test_data))


def best_distribution(data, target):
    train_data, test_data, train_labels, test_labels = train_test_split(data, target, shuffle=True, test_size=.3)
    bern_error = get_error(naive_bayes.BernoulliNB, train_data, test_data, train_labels, test_labels)
    multi_error = get_error(naive_bayes.MultinomialNB, train_data, test_data, train_labels, test_labels)
    gauss_error = get_error(naive_bayes.GaussianNB, train_data, test_data, train_labels, test_labels)
    return str(bern_error), str(multi_error), str(gauss_error)


def main():
    data = datasets.load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    print(df)

    data, target = datasets.load_digits(return_X_y=True)
    print('\nDigits: ' + ', '.join([*best_distribution(data, target)]))

    data, target = datasets.load_breast_cancer(return_X_y=True)
    print('Breast Cancer: ' + ', '.join([*best_distribution(data, target)]))


if __name__ == '__main__':
    main()
