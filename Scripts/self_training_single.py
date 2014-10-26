from sklearn.metrics import f1_score

__author__ = 'claire'
from utility import load_twitter_2class, load_amazon
from sklearn.svm import LinearSVC as svc
from sklearn.feature_extraction.text import TfidfVectorizer as tf
import random
import codecs
import matplotlib.pyplot as plt
import numpy as np
import sys

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

name = test_f.split('/')[-1].replace('.', '-')

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

iters = 30
threshold = float(sys.argv[1])
added = 0
scores = []  #keep track of how it changes according to the development set
for i in range(iters):
    # find points above threshold to add to training data
    distance = clf.decision_function(X_U)  # the distance (- or +) from the hyperplane
    idx = np.where(abs(distance) > threshold)[0]  # the indices above the threshold distance
    new = np.random.choice(idx)  # pick one to add
    target = map(totarget, [distance[new]])
    y_train += target
    train.append(np.array(unlabeled)[new].tolist())

    # remove from unlabeled
    unlabeled.pop(new)
    X_train = vec.fit_transform(train)
    X_test = vec.transform(test)
    X_U = vec.transform(unlabeled)
    clf = svc()
    clf.fit(X_train, y_train)
    scores.append(f1_score(y_test, clf.predict(X_test), pos_label=None, average='macro'))
    # print 'added %d unlabeled datapoints' % added
    print 'Iteration %d : accuracy: %f ' % (i, scores[-1])

with open(name + 'threshold_results.txt', 'a') as f:
    f.write('threshold_plots/single_threshold=' + str(threshold).replace('.', '_') + 'iters=' + str(iters) + '\n')
    f.write('best: %f iter: %d' % (np.max(scores), np.argmax(scores)))
    f.write('\n')

plt.plot(range(len(scores)), scores)
plt.title(name + '_single_threshold=' + str(threshold).replace('.', '_') + 'batches=' + str(iters))
plt.xlabel('iters')
plt.ylabel('F1 macro')
# plt.show()
plt.savefig('threshold_plots/' + name + '_single_threshold=' + str(threshold).replace('.', '_') + 'iters=' + str(iters))