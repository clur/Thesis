import cPickle
from sklearn.metrics import f1_score

__author__ = 'claire'
from sklearn.svm import LinearSVC as svc
from sklearn.linear_model import LogisticRegression as lr
from sklearn.feature_extraction.text import TfidfVectorizer as tf
from sklearn.feature_extraction.text import CountVectorizer as cv
import random
import codecs
import matplotlib.pyplot as plt
import numpy as np
import sys


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


def load_unlabeled_twitter(fname):
    raw = codecs.open(fname, 'r', 'utf8')  # load and split data into reviews
    return [''.join(r.split('\t')[1:]) for r in raw]


def totarget(i):
    if i < 0:
        result = -1
    else:
        result = 1
    return result


# Parameters to tune
# Threshold: range
#Vectorizer: cv or tf (and vectorizer params)
#Refit: vec.fit at each iteration or not
#Classifier: svc or lr (and clf params)
#Stopping criteria: patience?
#NPoints: how many to add, 1 at a time vs. range of points vs. all above threshold vs. balanced



def run(threshold, v, c):
    i = -1  # first iteration, controls fitting vec

    train_f = '/Users/claire/Dropbox/PycharmProjects/Thesis/Scripts/Data/twitter/twitter.train'
    test_f = '/Users/claire/Dropbox/PycharmProjects/Thesis/Scripts/Data/twitter/twitter.dev'
    unlabeled_f = '/Users/claire/Dropbox/PycharmProjects/Thesis/Scripts/Data/twitter_CST/englishtweets.both'

    train, y_train = load_twitter_2class(train_f)
    test, y_test = load_twitter_2class(test_f)
    unlabeled = load_unlabeled_twitter(unlabeled_f)

    name = test_f.split('/')[-1].replace('.', '-')
    unlabeled = unlabeled[:5000]
    # random.shuffle(unlabeled)
    unlabeled_size = len(unlabeled)

    # Fit transform data
    vec = v
    X_train = vec.fit_transform(train)
    X_test = vec.transform(test)
    X_U = vec.transform(unlabeled)

    # train classifier on labeled data
    clf = c
    clf.fit(X_train, y_train)
    print 'initial score:', f1_score(y_test, clf.predict(X_test), pos_label=None, average='macro')
    # print 'initial size:train: %d unlabeled: %d' % (X_train.shape[0], X_U.shape[0])
    start_size = X_train.shape[0]
    iters = 2
    scores = []  # keep track of how it changes according to the development set
    scores.append(f1_score(y_test, clf.predict(X_test), pos_label=None, average='macro'))
    for i in range(iters):
        # print
        # find points above threshold to add to training data
        # print 'unlabeled shape', X_U.shape
        distance = clf.decision_function(X_U)  # the distance (- or +) from the hyperplane
        idx = np.where(abs(distance) > threshold)[0]  #indices above threshold
        target = map(totarget, distance[idx])  #convert the targets to -1,1
        y_train += target  #add to train
        train += np.array(unlabeled)[idx]

        # remove those points from unlabeled
        unlabeled = [unlabeled[x] for x in range(len(unlabeled)) if x not in idx]

        # re-transform
        X_train = vec.transform(train)
        X_test = vec.transform(test)
        X_U = vec.transform(unlabeled)
        clf.fit(X_train, y_train)
        # if i % 10 == 0:
        scores.append(f1_score(y_test, clf.predict(X_test), pos_label=None, average='macro'))
        # print 'added: %d data points' % (len(idx))
        # print 'Iteration %d : accuracy: %f ' % (i, scores[-1])

    res = f1_score(y_test, clf.predict(X_test), pos_label=None, average='macro')
    print 'f1 macro:', res
    print
    # color = cm(1. * i / NUM_COLORS)  # color will now be an RGBA tuple
    # cm = plt.get_cmap('gist_rainbow')
    # fig = plt.figure(figsize=(8.0, 5.0))
    # ax = fig.add_subplot(111)
    # # ax.set_color_cycle([cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
    # ax.plot(range(len(scores)), scores, label=str(threshold))
    # ax.text(len(scores) - 1, scores[len(scores) - 1], threshold, fontsize='smaller')
    # plt.show()
    print name
    return res


vec_list = [tf(), cv()]
clf_list = [svc(), lr()]
threshold_list = np.arange(0.5, 3, 0.5)
print len(threshold_list)
# results_size = (len(vec_list), len(clf_list),len(threshold_list))
# results = np.zeros(results_size, dtype = np.float)
# a, b, c = range(3), range(3), range(3)
# def my_func(x, y, z):
#     return (x + y + z) / 3.0, x * y * z, max(x, y, z)

grids = np.vectorize(run)(*np.ix_(threshold_list, vec_list, clf_list))
# mean_grid, product_grid, max_grid = grids
print len(grids)
try:
    print grids.shape
except:
    print type(grids)
np.save('temp', grids)
# print mean_grid, product_grid, max_grid
# from scipy.optimize import brute
# results = brute(run,[threshold_list,vec_list,clf_list], full_output = True)


'''
plt.xlabel('iters')
plt.ylabel('F1 macro')
plt.legend(loc=1, fontsize='x-small')
fig.suptitle('Grid search of threshold values\n unlabeled data size = %d' % unlabeled_size)
fig.savefig('threshold_plots/' + name + '_GRID_' + str(threshold).replace('.', '_') + '_thresholds')
plt.show()
'''