from __future__ import division

import math
import random

import pandas as pd

from collections import defaultdict
from sklearn.model_selection import train_test_split

NOT_SPAM = 0
SPAM = 1

class NaiveBayes:
    """A Naive Bayes model for text classification."""
    def __init__(self):

        self.class_word_frequencies={"SPAM":defaultdict(float),"NOT SPAM": defaultdict(float)}

        self.vocab = ['make','address','all','3d','our','over','remove','internet','order','mail',
                      'receive','will','people','report','addresses','free','business','email','you',
                      'credit','your','font','000','money','hp','hpl','george','650','lab','labs',
                      'talent','857','data','415','85','technology','1999','parts','pm','direct',
                      'cs','meeting','original','project','re','edu','table','conference']

        # class_total_doc_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of documents in the trainning set of that class
        self.class_total_doc_counts = { SPAM: 0.0,
                                        NOT_SPAM: 0.0 }

        # class_total_word_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of words in the training set in documents of that class
        self.class_total_word_frequency = { SPAM: 0.0,
                                         NOT_SPAM: 0.0 }

        # class_word_counts is a dictionary of dictionaries. It maps a class (i.e.,
        # pos/neg) to a dictionary of word counts. For example:
        #    self.class_word_counts[POS_LABEL]['awesome']
        # stores the number of times the word 'awesome' appears in documents
        # of the positive class in the training documents.
        self.class_word_frequencies = { SPAM: defaultdict(float),
                                   NOT_SPAM: defaultdict(float) }


    def bow(self,X):
        bow = defaultdict(float)
        for i in range(len(X)):
            bow[self.vocab[i]]=X[i]

        return bow

    def train_model(self,X_train,y_train):
        for (row,label) in zip(X_train,y_train):
            bow=self.bow(row)
            self.update_model(bow,label)

    def update_model(self, bow, label):

        for word,frequency in bow.items():
            self.class_word_frequencies[label][word] = self.class_word_frequencies[label][word] + frequency

        self.class_total_doc_counts[label] = self.class_total_doc_counts[label] + 1
        self.class_total_word_frequency[label] = self.class_total_word_frequency[label] + sum(bow.values())

    def p_word_given_label_and_psuedocount(self, word, label,alpha):
        return (self.class_word_frequencies[label][word]+alpha)/(self.class_total_word_frequency[label]+(len(self.vocab)*alpha))

    def log_likelihood(self, bow, label, alpha):

        ln_likelihood = 0.0
        for word in bow.keys():
            ln_likelihood += math.log(self.p_word_given_label_and_psuedocount(word, label, alpha))

        return ln_likelihood

    def log_prior(self, label):
        return math.log(self.class_total_doc_counts[label]/(self.class_total_doc_counts[SPAM]+self.class_total_doc_counts[NOT_SPAM]))


    def unnormalized_log_posterior(self, bow, label, alpha):
        return self.log_likelihood(bow,label,alpha)+self.log_prior(label)

    def classify(self, bow, alpha):
        spam_posteriori = self.unnormalized_log_posterior(bow,SPAM,alpha)
        not_spam_posteriori = self.unnormalized_log_posterior(bow,NOT_SPAM,alpha)

        if spam_posteriori>not_spam_posteriori:
            return SPAM
        elif spam_posteriori<not_spam_posteriori:
            return NOT_SPAM
        else:
            return random.choice([SPAM,NOT_SPAM])

    def evaluate_classifier_accuracy(self, X_test,y_test,alpha):

        error=0
        for (test_data,label) in zip(X_test,y_test):
            bow = self.bow(test_data)
            classified_label=self.classify(bow,alpha)
            if classified_label!=label:
                error+=1

        print('error rate of Naive Bayes classifier  : ' ,"{:.3%}".format(error/len(X_test)))


# if __name__ == '__main__':
#     dataframe = pd.read_csv('spambase.data', header=None)
#     y = dataframe.iloc[:, -1].values
#     X = dataframe.iloc[:, :-10].values
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)
#
#     nb = NaiveBayes()
#     nb.train_model(X_train,y_train)
#     nb.evaluate_classifier_accuracy(X_test,y_test,0.5)
#

