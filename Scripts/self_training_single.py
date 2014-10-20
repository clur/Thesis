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


threshold = 0.5
added = 0
scores = []  #keep track of how it changes according to the development set
for i in range(X_U.shape[0]):
    # print clf.predict(i)
    distance = clf.decision_function(X_U[i])  # the distance (- or +) from the hyperplane
    if abs(distance) > threshold:
        # print 'adding',U[i]
        added += 1
        train.append(U[i])
        if distance > 0:
            y_train.append(1)
        else:
            y_train.append(-1)
        vec = tf()
    X_train = vec.fit_transform(train)
    X_test = vec.transform(test)
    X_U = vec.transform(U)
    clf.fit(X_train, y_train)
    if i % 10 == 0:
        scores.append(clf.score(X_test, y_test))
        print 'added %d unlabeled datapoints' % added
        print 'Iteration %d : accuracy: %f ' % (i, scores[-1])
print clf.score(X_test, y_test)
print added
plt.plot(range(len(scores)), scores)

plt.xlabel('iters')
plt.ylabel('accuracy')
plt.show()

