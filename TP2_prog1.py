from sklearn import datasets, neural_network, model_selection, metrics
import numpy as np
import time
import random
from plot import Plot

if __name__ == '__main__':
    # Charger le jeu de données MNIST
    mnist = datasets.fetch_openml('mnist_784')

    # Diviser la base de données en 49000 lignes pour l’apprentissage (training) et
    # le reste pour les tests
    data = []
    target = []
    for i in np.random.randint(len(mnist.data), size=5000):
        data.append(mnist.data[i])
        target.append(mnist.target[i])

    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(
        data,
        target,
        train_size=49000/len(mnist.data)
    )

    # Construire un modèle de classification ayant comme paramètre :
    # hidden_layer_sizes = (50), puis calculez la précession du classifieur
    clf = neural_network.MLPClassifier(hidden_layer_sizes=50)
    clf.fit(xtrain, ytrain)
    results = clf.predict(xtest)
    print(ytest[0])
    print(results[0])
    score = clf.score(xtest, ytest)
    print("Score: ", score)
    score = metrics.precision_score(results, ytest, average='micro')
    print("Score: ", score)

    # Varier le nombre de couches de 1 entre (2 et 100) couches, et recalculer la
    # précision du classifieur
    plt = Plot()
    hidden_layer_sizes = (100,)
    for i in range(2, 101):
        start_time = time.time()
        hidden_layer_sizes += (101-i,)
        clf = neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
        clf.fit(xtrain, ytrain)
        plt.add(i, time.time()-start_time, ytest, clf.predict(xtest))
    plt.figure('Increment number of hidden layers')

    # Construire cinq modèles de classification des données mnist, avec des
    # réseaux qui ont respectivement de 1 à 10 couches cachées, et des tailles de
    # couches entre 10 et 300 neurones au choix d’une façon aléatoire
    plt = Plot()
    hidden_layer_sizes = ()
    for i in range(1, 11):
        start_time = time.time()
        hidden_layer_sizes += (random.randint(10, 300),)
        clf = neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
        clf.fit(xtrain, ytrain)
        plt.add(i, time.time()-start_time, ytest, clf.predict(xtest))
    plt.figure('Increment number of hidden layers and random number of neurons')

    # Étudier la convergence des algorithmes d’optimisation disponibles : L-BFGS, SGD et Adam
    plt = Plot()
    start_time = time.time()
    clf = neural_network.MLPClassifier(hidden_layer_sizes=(50,)*10, solver="lbfgs")
    clf.fit(xtrain, ytrain)
    plt.add("lbfgs", time.time()-start_time, ytest, clf.predict(xtest))
    start_time = time.time()
    clf = neural_network.MLPClassifier(hidden_layer_sizes=(50,)*10, solver="sgd")
    clf.fit(xtrain, ytrain)
    plt.add("sgd", time.time()-start_time, ytest, clf.predict(xtest))
    start_time = time.time()
    clf = neural_network.MLPClassifier(hidden_layer_sizes=(50,)*10, solver="adam")
    clf.fit(xtrain, ytrain)
    plt.add("adam", time.time()-start_time, ytest, clf.predict(xtest))
    plt.figure('Solver')

    # Varier les fonctions d’activation {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
    plt = Plot()
    start_time = time.time()
    clf = neural_network.MLPClassifier(hidden_layer_sizes=(50,)*10, activation="identity")
    clf.fit(xtrain, ytrain)
    plt.add("identity", time.time()-start_time, ytest, clf.predict(xtest))
    start_time = time.time()
    clf = neural_network.MLPClassifier(hidden_layer_sizes=(50,)*10, activation="logistic")
    clf.fit(xtrain, ytrain)
    plt.add("logistic", time.time()-start_time, ytest, clf.predict(xtest))
    start_time = time.time()
    clf = neural_network.MLPClassifier(hidden_layer_sizes=(50,)*10, activation="tanh")
    clf.fit(xtrain, ytrain)
    plt.add("tanh", time.time()-start_time, ytest, clf.predict(xtest))
    start_time = time.time()
    clf = neural_network.MLPClassifier(hidden_layer_sizes=(50,)*10, activation="relu")
    clf.fit(xtrain, ytrain)
    plt.add("relu", time.time()-start_time, ytest, clf.predict(xtest))
    plt.figure('Activation')

    # Changer la valeur de la régularisation L2 (paramètre α)
    plt = Plot()
    for i in range(1, 11):
        start_time = time.time()
        clf = neural_network.MLPClassifier(hidden_layer_sizes=(50,)*10, alpha=5/100000 * i)
        clf.fit(xtrain, ytrain)
        plt.add(5/100000 * i, time.time()-start_time, ytest, clf.predict(xtest))
    plt.figure('Alpha')

    plt.show()
