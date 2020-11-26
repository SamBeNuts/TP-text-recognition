from sklearn import datasets, svm, model_selection
import numpy as np
import time
from plot import Plot

if __name__ == '__main__':
    # Charger le jeu de données mnist
    mnist = datasets.fetch_openml('mnist_784')

    # Diviser la base de données à 70% pour l’apprentissage (training) et à 30% pour les tests
    data = []
    target = []
    for i in np.random.randint(len(mnist.data), size=5000):
        data.append(mnist.data[i])
        target.append(mnist.target[i])

    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(
        data,
        target,
        train_size=0.7
    )

    # Construire un modèle de classification ayant comme paramètres un noyau
    # linear: clsvm = svm.SVC(kernel=’linear’)
    clf = svm.SVC(kernel='linear')
    clf.fit(xtrain, ytrain)
    results = clf.predict(xtest)
    print(ytest[0])
    print(results[0])
    score = clf.score(xtest, ytest)
    print("Score: ", score)

    # Tentez d’améliorer les résultats en variant la fonction noyau : ‘poly’, ‘rbf’,
    # ‘sigmoid’, ‘precomputed
    plt = Plot()
    start_time = time.time()
    clf = svm.SVC(kernel='linear')
    clf.fit(xtrain, ytrain)
    plt.add('linear', time.time()-start_time, ytest, clf.predict(xtest))
    start_time = time.time()
    clf = svm.SVC(kernel='poly')
    clf.fit(xtrain, ytrain)
    plt.add('poly', time.time()-start_time, ytest, clf.predict(xtest))
    start_time = time.time()
    clf = svm.SVC(kernel='rbf')
    clf.fit(xtrain, ytrain)
    plt.add('rbf', time.time()-start_time, ytest, clf.predict(xtest))
    start_time = time.time()
    clf = svm.SVC(kernel='sigmoid')
    clf.fit(xtrain, ytrain)
    plt.add('sigmoid', time.time()-start_time, ytest, clf.predict(xtest))
    plt.figure('Kernel')

    # Faites varier le paramètre de tolérance aux erreurs C (5 valeurs entre 0.1 et
    # 1)
    plt = Plot()
    for i in range(5):
        start_time = time.time()
        clf = svm.SVC(C=(1-0.1)/4*i+0.1)
        clf.fit(xtrain, ytrain)
        plt.add((1-0.1)/4*i+0.1, time.time()-start_time, ytest, clf.predict(xtest))
    plt.figure('C')

    plt.show()
