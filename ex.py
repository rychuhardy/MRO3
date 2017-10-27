import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import random as rn
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import CondensedNearestNeighbour

data_points=[]
n_neighbors = 1
samples_in_class = 50

mean = [7, 7]
cov_matrix = np.matrix('12, 5; 5, 6')
x, y = np.random.multivariate_normal(mean, cov_matrix, samples_in_class).T
data_points += list(zip(x, y, it.repeat(0)))

mean = [-1, -1]
cov_matrix = np.matrix('5, 4; 4, 4')
x, y = np.random.multivariate_normal(mean, cov_matrix, samples_in_class).T
data_points += list(zip(x, y, it.repeat(1)))

rn.shuffle(data_points)

X = np.array(list(map(lambda point: [point[0], point[1]], data_points)))
y = np.array(list(map(lambda point: point[2], data_points)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=19)

h = .02  # step size in the mesh

# Create color maps
cmap_light = ['#FFAAAA', '#AAFFAA', '#AAAAFF']
cmap_bold = ['#FF0000', '#00FF00', '#0000FF']
cmap_light2 = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold2 = ListedColormap(['#FF0000', '#0000FF'])

# plt.figure(1)
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold2, edgecolor='k', s=20)

for weights in ['uniform']:



    #CNN, k =1
    cnn = CondensedNearestNeighbour(return_indices=True, ratio='all', n_neighbors=n_neighbors)
    X_resampled, y_resampled, idx_resampled = cnn.fit_sample(X_train, y_train)

    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X_resampled, y_resampled)

    fig = plt.figure(1)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

  # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light2)

    idx_samples_removed = np.setdiff1d(np.arange(X_train.shape[0]),
                                       idx_resampled)

    idx_class_0 = y_resampled == 0
    plt.scatter(X_resampled[idx_class_0, 0], X_resampled[idx_class_0, 1],
                alpha=.8, label='Class #0', color=cmap_bold[0])
    plt.scatter(X_resampled[~idx_class_0, 0], X_resampled[~idx_class_0, 1],
                alpha=.8, label='Class #1', color=cmap_bold[2])
    plt.scatter(X_train[idx_samples_removed, 0], X_train[idx_samples_removed, 1],
                alpha=.8, label='Removed samples', color=cmap_bold[1])

    print(clf.score(X_test, y_test))
    print('CNN reduction ', len(X_resampled)/len(X_train))
    plt.title('CNN k=1')
    plt.legend()




   # Euclidean k=1
    clf = neighbors.KNeighborsClassifier(3, weights=weights)
    clf.fit(X_train, y_train)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(4)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light2)

    # Plot also the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold2,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Euclidean k=1")
    print(clf.score(X_test, y_test))











    # CNN, k =3
    cnn = CondensedNearestNeighbour(return_indices=True, ratio='all', n_neighbors=3)
    X_resampled, y_resampled, idx_resampled = cnn.fit_sample(X_train, y_train)

    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(3, weights=weights)
    clf.fit(X_resampled, y_resampled)

    fig = plt.figure(3)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light2)

    idx_samples_removed = np.setdiff1d(np.arange(X_train.shape[0]),
                                       idx_resampled)

    idx_class_0 = y_resampled == 0
    plt.scatter(X_resampled[idx_class_0, 0], X_resampled[idx_class_0, 1],
                alpha=.8, label='Class #0', color=cmap_bold[0])
    plt.scatter(X_resampled[~idx_class_0, 0], X_resampled[~idx_class_0, 1],
                alpha=.8, label='Class #1', color=cmap_bold[2])
    plt.scatter(X_train[idx_samples_removed, 0], X_train[idx_samples_removed, 1],
                alpha=.8, label='Removed samples', color=cmap_bold[1])

    plt.title('CNN k=3')
    plt.legend()
    print(clf.score(X_test, y_test))
    print('CNN reduction ', len(X_resampled)/len(X_train))





    # Euclidean k=3
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X_train, y_train)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(2)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light2)

    # Plot also the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold2,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Euclidean k=3")

    print(clf.score(X_test, y_test))

    plt.show()