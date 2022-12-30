import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

class logistic_regression:

    predicted_reults = []
    losses=[]
    def __init__(self,X_train, X_test, y_train, y_test,alpha=0.3,epochs=20000):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.alpha = alpha
        self.epochs = epochs


    def sigmoid(self,z):
        return 1.0 / (1 + np.exp(-z))

    def loss(self,h):
        return np.sum(-self.y_train * np.log(h) - (1 - self.y_train) * np.log(1 - h))/self.X_train.shape[0]

    def graient_descent(self,h): # 1/m *(h-y)*x for every element of matrix
        return np.dot(h-self.y_train,self.X_train)/self.X_train.shape[0]

    def train(self):
        self.theta=np.zeros(self.X_train.shape[1],dtype=np.float128)

        for i in range(self.epochs):
            z = np.dot(self.X_train,self.theta)
            h=self.sigmoid(z)
            self.theta=self.theta-(self.alpha*self.graient_descent(h))

            self.losses.append(self.loss(h))

    def test(self):
        z=np.dot(self.X_test,self.theta)
        h=self.sigmoid(z)
        for element in h:
            if element>=0.5:
                self.predicted_reults.append(1)
            else:
                self.predicted_reults.append(0)

    def evaluate_results(self):

        error=0
        for i in range (len(self.y_test)):
            if self.y_test[i]!=self.predicted_reults[i]:
                error+=1

        print('error rate of logistic regression : ' ,"{:.3%}".format(error/len(self.X_test)))

        #plot loss function in epochs
        levels=np.arange(self.epochs)

        plt.scatter(levels, self.losses,alpha=0.5)
        plt.title('loss based on iteration')
        plt.xlabel('iteration number')
        plt.ylabel('loss')
        plt.show()


if __name__ == '__main__':

    dataframe=pd.read_csv('spambase.data',header=None)
    y = dataframe.iloc[:, -1].values
    X=dataframe.iloc[:,:-1]
    X_norm=(X-X.mean())/(X.max()-X.min())

    X_train, X_test, y_train, y_test = train_test_split(X_norm.values, y, test_size=0.20, random_state=5)

    logistic_regression=logistic_regression(X_train, X_test, y_train, y_test)

    logistic_regression.train()
    logistic_regression.test()

    logistic_regression.evaluate_results()
