from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import pandas as pd
import seaborn as sn


class Plot:
    def __init__(self):
        self.label = []
        self.time = []
        self.precision = []
        self.rappel = []
        self.error = []
        self.cm = []

    def add(self, label, time, ytest, ypred):
        self.label.append(label)
        self.time.append(time)
        self.precision.append(metrics.precision_score(ytest, ypred, average='micro') * 100)
        self.rappel.append(metrics.recall_score(ytest, ypred, average='micro') * 100)
        self.error.append(metrics.zero_one_loss(ytest, ypred) * 100)
        self.cm.append(metrics.confusion_matrix(ytest, ypred))
        print('Values added to plot!')

    def subplot(self, id, xlabel, ylabel, x, y, min, max):
        plt.subplot(id)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.scatter(x, y)
        plt.ylim(bottom=min, top=max)

    def figure(self, title):
        plt.figure(title)
        self.subplot(221, 'Time', 's', self.label, self.time, 0, None)
        self.subplot(222, 'Precision', '%', self.label, self.precision, 80, 100)
        self.subplot(223, 'Rappel', '%', self.label, self.rappel, 80, 100)
        self.subplot(224, 'Error', '%', self.label, self.error, 0, 20)
        plt.tight_layout()
        self.subplot_cm(title)
        print('Figure created!')

    def subplot_cm(self, title):
        best = self.precision[0]
        best_i = 0
        for i in range(1, len(self.precision)):
            if best < self.precision[i]:
                best = self.precision[i]
                best_i = i
        df_cm = pd.DataFrame(self.cm[best_i], range(len(self.cm[0])), range(len(self.cm[0])))
        plt.figure("Confusion matrix for %s=%s" % (title, self.label[best_i]))
        sn.heatmap(df_cm, annot=True, fmt='d')

    def show(self):
        plt.show()
