import pandas as pd
from sklearn.model_selection import train_test_split
import logistic_regression
from nb import NaiveBayes

if __name__ == '__main__':

    dataframe = pd.read_csv('spambase.data', header=None)
    y = dataframe.iloc[:, -1].values
    X = dataframe.iloc[:, :-1]
    X_norm = (X - X.mean()) / (X.max() - X.min())

    print("Evaluating Logistic Regression :")
    X_train, X_test, y_train, y_test = train_test_split(X_norm.values, y, test_size=0.20, random_state=5)
    logistic_regression = logistic_regression.logistic_regression(X_train, X_test, y_train, y_test)
    logistic_regression.train()
    logistic_regression.test()
    logistic_regression.evaluate_results()

    print()
    print("Evaluating Naive Bayes classifier ;")
    X = dataframe.iloc[:, :-10].values
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.20, random_state=5)
    nb = NaiveBayes()
    nb.train_model(X_train, y_train)
    nb.evaluate_classifier_accuracy(X_test, y_test, 0.5)

