__author__ = 'claire'
from utility import load_twitter_2class, load_amazon
from sklearn.svm import LinearSVC as svc
from sklearn.feature_extraction.text import TfidfVectorizer as tf
import random
import codecs
import matplotlib.pyplot as plt
import numpy as np


def load_unlabeled_twitter(fname):
    raw = codecs.open(fname, 'r', 'utf8')  # load and split data into reviews
    return [''.join(r.split('\t')[1:]) for r in raw]


def totarget(i):
    if i < 0:
        result = -1
    else:
        result = 1
    return result

# Load datasets
train, y_train = load_twitter_2class('Data/twitter/twitter.train')
test, y_test = load_twitter_2class('Data/twitter/twitter.dev')
unlabeled = load_unlabeled_twitter('Data/twitter_CST/englishtweets.both')
# unlabeled=unlabeled[:5000]
random.shuffle(unlabeled)
# Fit vectorizer with training data and transform datasets
vec = tf()
X_train = vec.fit_transform(train)
X_test = vec.transform(test)
X_U = vec.transform(unlabeled)


# train classifier on labeled data
clf = svc()
clf.fit(X_train, y_train)
# print clf.score(X_test,y_test)


threshold = 1
added = 0
scores = []  # keep track of how it changes according to the development set
iters = 50
for i in range(iters):
    print
    print 'iteration %d' % i
    # find points above threshold to add to training data
    print 'unlabeled shape', X_U.shape
    distance = clf.decision_function(X_U)  # the distance (- or +) from the hyperplane
    idx = np.where(abs(distance) > threshold)[0]  # the indices above the threshold distance
    # take the 50 highest
    top = (-(abs(distance)[idx])).argsort()[:30]
    idx = idx[top]  #to remove
    print idx
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
    print 'added %d unlabeled datapoints' % len(idx)
    print 'Iteration %d : accuracy: %f ' % (i, scores[-1])
print clf.score(X_test, y_test)
print added
plt.plot(range(len(scores)), scores)

plt.xlabel('iters')
plt.ylabel('accuracy')
plt.savefig('threshold=' + str(threshold).replace('.', '_') + ' iters=' + str(iters))
plt.show()
