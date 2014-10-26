__author__ = 'claire'
from utility import load_twitter_2class, load_amazon
from sklearn.svm import LinearSVC as svc
from sklearn.feature_extraction.text import TfidfVectorizer as tf
import random
import codecs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import sys
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
print 'F1 score', f1_score(y_test, clf.predict(X_test), pos_label=None, average='macro')

threshold = float(sys.argv[1])
added = 0
scores = []  # keep track of how it changes according to the development set
iters = 30
num_top = 30
for i in range(iters):
    # find points above threshold to add to training data
    distance = clf.decision_function(X_U)  # the distance (- or +) from the hyperplane
    idx = np.where(abs(distance) > threshold)[0]  # the indices above the threshold distance
    # take the 50 highest
    top = (-(abs(distance)[idx])).argsort()[:num_top]
    idx = idx[top]  #to remove
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
    print 'added %d unlabeled datapoints' % len(idx)
    print 'Iteration %d : accuracy: %f ' % (i, scores[-1])

with open(name + 'threshold_results.txt', 'a') as f:
    f.write('threshold_plots/top_threshold=' + str(threshold).replace('.', '_') + 'iters=' + str(iters) + '\n')
    f.write('best: %f iter: %d' % (np.max(scores), np.argmax(scores)))
    f.write('\n')

print f1_score(y_test, clf.predict(X_test), pos_label=None, average='macro')
plt.plot(range(len(scores)), scores)
plt.xlabel('iters')
plt.ylabel('F1 macro')
plt.title(name + '_top_threshold=' + str(threshold).replace('.', '_') + ' iters=' + str(iters) + 'top ' + str(num_top))
plt.savefig('threshold_plots/' + name + 'top_threshold=' + str(threshold).replace('.', '_') + ' iters=' + str(
    iters) + 'top ' + str(num_top))

