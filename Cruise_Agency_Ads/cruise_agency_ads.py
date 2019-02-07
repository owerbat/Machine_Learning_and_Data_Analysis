import pandas as pd
import csv
import numpy as np
from sklearn.cluster import MeanShift
from numpy.linalg import norm
from itertools import product


def prepare_csv():
    with open('checkins.dat', 'r') as data_file, open('checkins.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file)

        csv_writer.writerow([item.strip() for item in data_file.readline().split('|')])
        for i in range(3):
            data_file.readline()

        for line in data_file:
            row = [item.strip() for item in line.split('|')]
            try:
                if row[3] != '' and row[4] != '':
                    csv_writer.writerow(row)
            except IndexError:
                print(row)


def nearest_center(centers, offices):
    distance = np.inf
    coordinates = None
    for center, office in product(np.array(centers), offices):
        if norm(center-office) < distance:
            distance = norm(center-office)
            coordinates = center
    return distance, list(coordinates)


def main():
    df = pd.read_csv('checkins.csv')
    print(df.keys())

    X = df.loc[:10**4, ['latitude', 'longitude']]
    cls = MeanShift(.1, n_jobs=8)
    cls.fit(X)
    print(cls.labels_)

    counts = np.array([list(cls.labels_).count(i) for i in range(max(cls.labels_)+1)])
    true_counts = np.array([item if item > 15 else -1 for item in counts])
    clusters = []
    for i, item in enumerate(true_counts):
        if item > -1:
            clusters.append(i)
    centers = [list(x) for x in cls.cluster_centers_[clusters]]

    offices = np.array([[33.751277, -118.188740],
                       [25.867736, -80.324116],
                       [51.503016, -0.075479],
                       [52.378894, 4.885084],
                       [39.366487, 117.036146],
                       [-33.868457, 151.205134]])

    result = []
    for i in range(20):
        distance, coordinates = nearest_center(centers, offices)
        result.append((distance, coordinates))
        centers.remove(coordinates)

    print(result)


if __name__ == '__main__':
    main()
