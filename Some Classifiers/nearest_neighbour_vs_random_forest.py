from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def nearest_neighbour(train_data, test_data, train_labels, test_labels):
    clf = KNeighborsClassifier(n_neighbors=1, weights='distance', n_jobs=8)
    clf.fit(train_data, train_labels)
    return metrics.accuracy_score(test_labels, clf.predict(test_data))


def random_forest(train_data, test_data, train_labels, test_labels):
    clf = RandomForestClassifier(n_estimators=1000, n_jobs=8)
    clf.fit(train_data, train_labels)
    return metrics.accuracy_score(test_labels, clf.predict(test_data))


def main():
    data, target = load_digits(return_X_y=True)
    train_data, test_data, train_labels, test_labels = train_test_split(data, target, shuffle=True, test_size=.3)
    print(nearest_neighbour(train_data, test_data, train_labels, test_labels))
    print(random_forest(train_data, test_data, train_labels, test_labels))


if __name__ == '__main__':
    main()
