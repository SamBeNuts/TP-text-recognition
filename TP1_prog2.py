from sklearn import datasets, neighbors, model_selection
import matplotlib.pyplot as plt
import numpy as np
import time

if __name__ == '__main__':
    mnist = datasets.fetch_openml('mnist_784')

    data = []
    target = []
    for i in np.random.randint(len(mnist.data), size=5000):
        data.append(mnist.data[i])
        target.append(mnist.target[i])

    '''
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=0.8)
    
    clf = neighbors.KNeighborsClassifier(5)
    clf.fit(xtrain, ytrain)
    clf.predict(xtest)
    print("Score: ", clf.score(xtest, ytest))
    
    images = np.array(xtest).reshape((-1, 28, 28))
    results = clf.predict(xtest)
    for i in range(5):
        plt.imshow(images[i], cmap=plt.cm.gray_r, interpolation="nearest")
        print("Predict n°", i + 1, ": ", results[i])
        plt.show()
    '''

    '''
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=0.8)
    for i in range(2, 16):
        clf = neighbors.KNeighborsClassifier(i)
        clf.fit(xtrain, ytrain)
        print("Score (k=", i, "): ", clf.score(xtest, ytest))
    '''

    '''
    clf = neighbors.KNeighborsClassifier(5)
    for i in range(1, 10):
        xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=1-i/10)
        clf.fit(xtrain, ytrain)
        print("Score (train_size=", 1-i/10, "): ", clf.score(xtest, ytest))
    '''

    '''
    data = []
    target = []
    for i in np.random.randint(len(mnist.data), size=10000):
        data.append(mnist.data[i])
        target.append(mnist.target[i])

    clf = neighbors.KNeighborsClassifier(5)
    for i in range(1, 11):
        xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data[:(i*1000)], target[:(i*1000)], train_size=0.8)
        clf.fit(xtrain, ytrain)
        print("Score (data_size=", i * 1000, "): ", clf.score(xtest, ytest))
    '''

    '''
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=0.8)
    for i in range(2, 10):
        clf = neighbors.KNeighborsClassifier(5, p=i)
        clf.fit(xtrain, ytrain)
        print("Score (p=", i, "): ", clf.score(xtest, ytest))
    '''

    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=0.8)

    start_time = time.time()
    clf = neighbors.KNeighborsClassifier(5, n_jobs=1)
    clf.fit(xtrain, ytrain)
    clf.predict(xtest)
    print("Execution time (n_jobs=1): ", time.time() - start_time, "s")

    start_time = time.time()
    clf = neighbors.KNeighborsClassifier(5, n_jobs=-1)
    clf.fit(xtrain, ytrain)
    clf.predict(xtest)
    print("Execution time (n_jobs=-1): ", time.time() - start_time, "s")
