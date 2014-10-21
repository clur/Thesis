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

for threshold in np.arange(0.1, 1.5, 0.1):
    # Load datasets
    train, y_train = load_twitter_2class('Data/twitter/twitter.train')
    test, y_test = load_twitter_2class('Data/twitter/twitter.dev')
    unlabeled = 'Data/amazon_embeddings/vshort.txt'
    raw = codecs.open(unlabeled, 'r', 'utf8')  # load and split data into reviews
    U = []
    for r in raw:
        try:
            U.append(' '.join(r.split('\t')[6:]).strip())
        except:
            pass
    raw.close()
    # random.shuffle(X_U)
    U_total = U
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


    print 'initial size:train: %d unlabeled: %d' % (X_train.shape[0], X_U.shape[0])
    start_size = X_train.shape[0]
    # print 'Start train set size: %d' % start_size
    iters = 50
    scores = []  #keep track of how it changes according to the development set
    scores.append(clf.score(X_test, y_test))
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
            print 'added: %d data points' % (len(temp))
            print 'Iteration %d : accuracy: %f ' % (i, scores[-1])

    print clf.score(X_test, y_test)
    plt.plot(range(len(scores)), scores, label=str(threshold))
    plt.xlabel('iters')
    plt.ylabel('accuracy')
    # plt.show()
plt.legend(loc=1)
plt.savefig('thresholds')
