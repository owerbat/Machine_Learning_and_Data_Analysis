from sklearn import model_selection
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

from numpy import sqrt


def task1(data, target):
    clf = DecisionTreeClassifier()
    return model_selection.cross_val_score(clf, data, y=target, cv=10, n_jobs=8).mean()


def task2(data, target):
    clf = BaggingClassifier(DecisionTreeClassifier(), 100, n_jobs=8)
    return model_selection.cross_val_score(clf, data, y=target, cv=10, n_jobs=8).mean()


def task3(data, target):
    clf = BaggingClassifier(DecisionTreeClassifier(), 100, n_jobs=8, max_features=int(sqrt(data.shape[1])),
                            bootstrap_features=True)
    return model_selection.cross_val_score(clf, data, y=target, cv=10, n_jobs=8).mean()


def task4(data, target):
    clf = BaggingClassifier(DecisionTreeClassifier(max_features='sqrt'), 100, n_jobs=8, bootstrap_features=True)
    return model_selection.cross_val_score(clf, data, y=target, cv=10, n_jobs=8).mean()


def task5(data, target):
    clf = RandomForestClassifier(n_estimators=100, n_jobs=8, max_depth=None, max_features='sqrt')
    return model_selection.cross_val_score(clf, data, y=target, cv=10, n_jobs=8).mean()


def main():
    data, target = datasets.load_digits(return_X_y=True)

    # print('task1: {}'.format(task1(data, target)))
    # print('task2: {}'.format(task2(data, target)))
    # print('task3: {}'.format(task3(data, target)))
    # print('task4: {}'.format(task4(data, target)))
    print('task5: {}'.format(task5(data, target)))


if __name__ == '__main__':
    main()
