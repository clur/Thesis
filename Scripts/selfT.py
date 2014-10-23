__author__ = 'claire'
import cPickle
from utility import load_twitter_2class, load_amazon
from sklearn.naive_bayes import BernoulliNB as nb
from sklearn.svm import LinearSVC as svc
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as log
from sklearn.feature_extraction.text import TfidfVectorizer as tf
import random
import codecs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline


class Selftraining():
    def __init__(self, learner, iters=5, pool=100):
        self.learner = learner()
        self.iters = iters
        self.pool = pool

    def fit(self, X_train, y_train, U_size):

        L_size = X_train.shape[0] - U_size * 2
        print 'L_size', L_size
        weights = [True] * L_size + [False] * U_size * 2
        idxs = range(X_train.shape[0])
        print X_train[idxs[:2]].shape[0]

        def M_step():
            self.learner.fit(X_train, y_train)

        def E_step():
            for i in np.random.randint(L_size, X_train.shape[0] - U_size, size=self.pool):
                if float(self.learner.predict(X_train[i, :1])) == 1:
                    weights[i] = 1
                else:
                    weights[i + U_size] = 1

        M_step()
        for _ in range(self.iters):
            E_step()
            M_step()

    def score(self, X, y):
        return self.learner.score(X, y)


train, y_train = load_twitter_2class('Data/twitter/twitter.train')
unlabeled = load_unlabeled('Data/amazon_embeddings/vshort.txt')
test, y_test = load_twitter_2class('Data/twitter/twitter.dev')

train = train + unlabeled + unlabeled  # add two copies of unlabeled to train to pool from

# pipeline for extraction and training
clf = svc()
vec = tf()

X_train = vec.fit_transform(train)

trainer = Selftraining(svc)
print trainer.learner
trainer.fit(X_train, y_train, len(unlabeled))

'''

for threshold in np.arange(0.1, 1.5, 0.1):
    # Load datasets
    train, y_train = load_twitter_2class('Data/twitter/twitter.train')
    test, y_test = load_twitter_2class('Data/twitter/twitter.dev')
    unlabeled = 'Data/amazon_embeddings/vshort.txt'
    raw = codecs.open(unlabeled, 'r', 'utf8')  # load and split data into reviews
    unlabeled = []
    for r in raw:
        try:
            unlabeled.append(' '.join(r.split('\t')[6:]).strip())
        except:
            pass
    raw.close()
    # random.shuffle(X_U)
    U_total = unlabeled
    # Fit vectorizer with training data and transform datasets
    vec = tf()
    X_train = vec.fit_transform(train)
    X_test = vec.transform(test)
    X_U = vec.transform(unlabeled)


    # train classifier on labeled data
    clf = svc()
    clf.fit(X_train, y_train)
    # print clf.score(X_test,y_test)

    def totarget(i):
        if i < 0:
            result = -1
        else:
            result = 1
        return result

    # train=np.array(train)


    print 'initial size:train: %d unlabeled: %d' % (X_train.shape[0], X_U.shape[0])
    start_size = X_train.shape[0]
    # print 'Start train set size: %d' % start_size
    iters = 50
    scores = []  #keep track of how it changes according to the development set
    scores.append(clf.score(X_test, y_test))
    for i in range(iters):
        # print clf.predict(i)
'''