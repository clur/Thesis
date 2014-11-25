__author__ = 'claire'
from sklearn.metrics import f1_score, classification_report
import codecs
import cPickle
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as tf
from sklearn.svm import LinearSVC as svc
from sklearn.linear_model import LogisticRegression as lr
from scipy.stats import logistic
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import StratifiedKFold, cross_val_score, ShuffleSplit
from sklearn.utils import shuffle


"""
use params learned from running autoencoder on unlabeled data to get representations for train and test
then classify
representation_train = s (dot(W,train_tfidf) +b)
"""


def print_top_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%f\t%-15s\t\t%f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)


def sigmoid(x):
    return logistic.cdf(x)


def load_twitter_2class(fname):
    """
    open and read fname, split into data and target
    :param fname: filename to read
    :return: data, list and target, list
    """
    raw = codecs.open(fname, 'r', 'utf-8').readlines()  # load and split data into reviews
    raw = [r for r in raw if r.split('\t')[2] != u'neutral']
    target = [r.split('\t')[2] for r in raw]  # target is pos,neg,neutral
    data = [r.split('\t')[3] for r in raw]  # review text
    data = [d.lower().strip() for d in data]
    target = [t for t in target if t != 0]
    mapping = {u'positive': 1, u'negative': - 1}
    target = [mapping[t] for t in target]
    return data, target


def get_rep(X):
    return sigmoid(X.dot(W) + b)


train_data, y_train = load_twitter_2class(
    '/Users/claire/Dropbox/PycharmProjects/Thesis/Scripts/Data/twitter/twitter.train')
test_data, y_test = load_twitter_2class('/Users/claire/Dropbox/PycharmProjects/Thesis/Scripts/Data/twitter/twitter.dev')
print 'train samples:', len(y_train)
print 'test samples:', len(y_test)

old_vec = cPickle.load(open('vec'))
vec = tf(vocabulary=old_vec.vocabulary_)
X_train = vec.fit_transform(train_data)
X_test = vec.transform(test_data)
print X_train.shape
print X_test.shape

# load params
W = cPickle.load(open('W_corr0.3_batchsize20_epochs100'))
b = cPickle.load(open('b_corr0.3_batchsize20_epochs100'))
print 'W:', W.shape, 'b:', b.shape

X_train = get_rep(X_train)
X_test = get_rep(X_test)

clf = svc()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print_top_features(vec, clf)
print classification_report(y_train, clf.predict(X_train))
print confusion_matrix(y_test, y_pred)
print classification_report(y_test, y_pred)
print f1_score(y_test, y_pred, pos_label=None, average='macro')

scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1')

print scores