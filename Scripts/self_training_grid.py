import cPickle

__author__ = 'claire'
from utility import load_twitter_2class, load_amazon
from sklearn.svm import LinearSVC as svc
from sklearn.feature_extraction.text import TfidfVectorizer as tf
import random
import codecs
import matplotlib.pyplot as plt
import numpy as np

# WRONG i'm adding points over and over again from unlabeled

def load_unlabeled(fname):
    raw = codecs.open(fname, 'r', 'utf8')  # load and split data into reviews
    unlabeled = []
    for r in raw:
        try:
            unlabeled.append(' '.join(r.split('\t')[6:]).strip())
        except:
            pass
    raw.close()
    return unlabeled


def load_unlabeled_twitter(fname):
    raw = codecs.open(fname, 'r', 'utf8')  # load and split data into reviews
    return [''.join(r.split('\t')[1:]) for r in raw]


def totarget(i):
        if i < 0:
            result = -1
        else:
            result = 1
        return result


start = 0.1
stop = 2.2
threshold_range = np.arange(start, stop, step=0.1)
NUM_COLORS = len(threshold_range)
cm = plt.get_cmap('gist_rainbow')
fig = plt.figure(figsize=(8.0, 5.0))
ax = fig.add_subplot(111)
ax.set_color_cycle([cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

for threshold in np.arange(start, stop, 0.1):
    # Load datasets
    train, y_train = load_twitter_2class('Data/twitter/twitter.train')
    test, y_test = load_twitter_2class('Data/twitter/twitter.dev')
    unlabeled = load_unlabeled_twitter('Data/twitter_CST/englishtweets.both')
    unlabeled = unlabeled[:5000]
    # random.shuffle(unlabeled)
    unlabeled_size = len(unlabeled)

    # Fit vectorizer with training data and transform datasets
    vec = tf()
    X_train = vec.fit_transform(train)
    X_test = vec.transform(test)
    X_U = vec.transform(unlabeled)

    # train classifier on labeled data
    clf = svc()
    clf.fit(X_train, y_train)
    print 'initial score:', clf.score(X_test, y_test)
    print 'initial size:train: %d unlabeled: %d' % (X_train.shape[0], X_U.shape[0])
    start_size = X_train.shape[0]
    iters = 10
    scores = []  #keep track of how it changes according to the development set
    scores.append(clf.score(X_test, y_test))
    for i in range(iters):
        print
        print 'iteration %d' % i
        # find points above threshold to add to training data
        print 'unlabeled shape', X_U.shape
        distance = clf.decision_function(X_U)  # the distance (- or +) from the hyperplane
        idx = np.where(abs(distance) > threshold)[0]
        target = map(totarget, distance[idx])
        y_train += target
        train += np.array(unlabeled)[idx]

        # remove those points from unlabeled
        unlabeled = [unlabeled[x] for x in range(len(unlabeled)) if x not in idx]
        X_train = vec.fit_transform(train)
        X_test = vec.transform(test)
        X_U = vec.transform(unlabeled)
        clf.fit(X_train, y_train)
        # if i % 10 == 0:
        scores.append(clf.score(X_test, y_test))
        print 'added: %d data points' % (len(idx))
        print 'Iteration %d : accuracy: %f ' % (i, scores[-1])

    print clf.score(X_test, y_test)

    color = cm(1. * i / NUM_COLORS)  # color will now be an RGBA tuple
    ax.plot(range(len(scores)), scores, label=str(threshold))
    ax.text(len(scores) - 1, scores[len(scores) - 1], threshold, fontsize='smaller')

    # plt.show()
plt.xlabel('iters')
plt.ylabel('accuracy')
plt.legend(loc=1, fontsize='x-small')
fig.suptitle('Grid search of threshold values\n unlabeled data size = %d' % unlabeled_size)
fig.savefig(str(threshold).replace('.', '_') + '_thresholds')
plt.show()