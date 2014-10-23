import cPickle

__author__ = 'claire'
from utility import load_twitter_2class, load_amazon
from sklearn.svm import LinearSVC as svc
from sklearn.feature_extraction.text import TfidfVectorizer as tf
import random
import codecs
import matplotlib.pyplot as plt
import numpy as np



# Load datasets
train, y_train = load_twitter_2class('Data/twitter/twitter.train')
test, y_test = load_twitter_2class('Data/twitter/twitter.dev')
unlabeled_file = 'Data/amazon_embeddings/short.txt'
raw = codecs.open(unlabeled_file, 'r', 'utf8')  # load and split data into reviews
unlabeled = []
for r in raw:
    try:
        unlabeled.append(' '.join(r.split('\t')[6:]).strip())
    except:
        pass
raw.close()
# random.shuffle(X_U)

unlabeled = unlabeled[:20]
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


print 'initial shapes (train, unlabeled)', X_train.shape, len(y_train), X_U.shape
threshold = 0.5
iters = 2
scores = []  # keep track of how it changes according to the development set
scores.append(clf.score(X_test, y_test))
start_size = X_train.shape[0]
print 'Start train set size: %d' % start_size
for i in range(iters):
    print
    # find points above threshold to add to training data
    print 'unlabeled shape', X_U.shape
    distance = clf.decision_function(X_U)  # the distance (- or +) from the hyperplane
    print distance
    print threshold
    print (abs(distance) > threshold)
    print 'greater than threshold?', len([i for i in distance if abs(i) > threshold])
    print np.where(abs(distance > threshold))

    idx = np.where(abs(distance) > threshold)[0]
    print 'idx', idx
    target = map(totarget, distance[idx])
    print 'target', target
    y_train += target
    train += np.array(unlabeled)[idx]

    # remove those points from unlabeled
    unlabeled = [unlabeled[i] for i in range(len(unlabeled)) if i not in idx]
    print 'new unlabeled', len(unlabeled)
    X_train = vec.fit_transform(train)
    X_test = vec.transform(test)
    X_U = vec.transform(unlabeled)
    clf.fit(X_train, y_train)
    if i % 10 == 0:
        scores.append(clf.score(X_test, y_test))
        print 'added: %d data points' % X_train.shape[0] - start_size
        print 'Iteration %d : accuracy: %f ' % (i, scores[-1])

print clf.score(X_test, y_test)
# plt.plot(range(len(scores)), scores)
# plt.xlabel('iters')
# plt.ylabel('accuracy')
# plt.show()

