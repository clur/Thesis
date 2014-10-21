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
unlabeled = 'Data/amazon_embeddings/short.txt'
raw = codecs.open(unlabeled, 'r', 'utf8')  # load and split data into reviews
U = []
for r in raw:
    try:
        U.append(' '.join(r.split('\t')[6:]).strip())
    except:
        pass
raw.close()
# random.shuffle(X_U)

# Fit vectorizer with training data and transform datasets
vec = tf()
X_train = vec.fit_transform(train)
X_test = vec.transform(test)
X_U = vec.transform(U)


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
iters = 50
scores = []  # keep track of how it changes according to the development set
scores.append(clf.score(X_test, y_test))
start_size = X_train.shape[0]
print 'Start train set size: %d' % start_size
for i in range(iters):
    # print clf.predict(i)
    distance = clf.decision_function(X_U)  # the distance (- or +) from the hyperplane
    toadd = abs(distance) > threshold
    temp = np.array(U)[toadd].tolist()
    train = train + temp
    target = map(totarget, distance)
    y_train = y_train + np.array(target)[toadd].tolist()
    X_train = vec.fit_transform(train)
    X_test = vec.transform(test)
    X_U = vec.transform(U)
    clf.fit(X_train, y_train)
    if i % 10 == 0:
        scores.append(clf.score(X_test, y_test))
        print 'added: %d data points' % X_train.shape[0] - start_size
        print 'Iteration %d : accuracy: %f ' % (i, scores[-1])

print clf.score(X_test, y_test)
plt.plot(range(len(scores)), scores)
plt.xlabel('iters')
plt.ylabel('accuracy')
plt.show()

