import cPickle
from sklearn.metrics import f1_score
import sys

'''
ADD ALL THE POINTS ABOVE THRESHOLD TO TRAIN ON EACH PASS
'''
__author__ = 'claire'
from utility import load_twitter_2class, load_amazon
from sklearn.svm import LinearSVC as svc
from sklearn.linear_model import LogisticRegression as log
from sklearn.feature_extraction.text import TfidfVectorizer as tf
from sklearn.feature_extraction.text import CountVectorizer as cv
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
train_f = 'Data/twitter/twitter.train'
test_f = 'Data/twitter/twitter.dev'
unlabeled_f = 'Data/twitter_CST/englishtweets.both'

train, y_train = load_twitter_2class(train_f)
test, y_test = load_twitter_2class(test_f)
unlabeled = load_unlabeled_twitter(unlabeled_f)
random.shuffle(unlabeled)

name = test_f.split('/')[-1].replace('.', '-')


# Fit vectorizer with training data and transform datasets
vec = cv()
X_train = vec.fit_transform(train)
X_test = vec.transform(test)
X_U = vec.transform(unlabeled)


# train classifier on labeled data
clf = log()
clf.fit(X_train, y_train)
# threshold = float(sys.argv[1])
threshold = 1
iters = 10
scores = []  # keep track of how it changes according to the development set
scores.append(f1_score(y_test, clf.predict(X_test), pos_label=None, average='macro'))
print scores
start_size = X_train.shape[0]
for i in range(iters):
    # find points above threshold to add to training data
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
    scores.append(f1_score(y_test, clf.predict(X_test), pos_label=None, average='macro'))
    # print 'added: %d data points' % (len(idx))
    # print 'Iteration %d : accuracy: %f ' % (i, scores[-1])

with open(name + 'threshold_results.txt', 'a') as f:
    f.write('all_threshold=' + str(threshold).replace('.', '_') + 'iters=' + str(iters) + '\n')
    f.write('best: %f iter: %d' % (np.max(scores), np.argmax(scores)))
    f.write('\n')

print f1_score(y_test, clf.predict(X_test), pos_label=None, average='macro')
plt.plot(range(len(scores)), scores)
plt.xlabel('iters')
plt.ylabel('F1 macro')
plt.title(name + '_all_threshold=' + str(threshold).replace('.', '_') + 'iters=' + str(iters))
plt.savefig('threshold_plots/' + name + 'all_threshold_' + str(threshold).replace('.', '_') + 'iters=' + str(iters))