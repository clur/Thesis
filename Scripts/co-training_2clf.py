"""
Shouldn't I refit the vectorizer? otherwise, what is the point of adding new data???

Using decision boundary
"""
from __future__ import division
from sklearn.metrics import f1_score

__author__ = 'claire'
from utility import load_twitter_2class, load_amazon
from sklearn.svm import LinearSVC as svc
from sklearn.naive_bayes import BernoulliNB as nb
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


def validate(tr, te):
    v = cv()
    X_tr = v.fit_transform(tr)
    X_te = v.transform(te)
    c = svc()
    c.fit(X_tr, y_train)
    return f1_score(y_test, c.predict(X_te), pos_label=None, average='macro')

# LOAD
# Load datasets
train_f = 'Data/twitter/twitter.train'
test_f = 'Data/twitter/twitter.dev'
unlabeled_f = 'Data/twitter_CST/englishtweets.both'
train, y_train = load_twitter_2class(train_f)
test, y_test = load_twitter_2class(test_f)
unlabeled = load_unlabeled_twitter(unlabeled_f)
name = test_f.split('/')[-1].replace('.', '-')
# random.shuffle(unlabeled)
# unlabeled = unlabeled[:5000]

#VECTORIZE
#vectorize based on initial training data
vec = cv()
X_train = vec.fit_transform(train)
X_test = vec.transform(test)
X_U = vec.transform(unlabeled)
print 'initial train size:', X_train.shape[0]

#INITIAL TRAIN
#train two classifiers on initial training data
clf1 = svc()
clf1.fit(X_train, y_train)
print 'initial clf1 on train:', f1_score(y_train, clf1.predict(X_train), pos_label=None, average='macro')
print 'initial clf1 on dev:', f1_score(y_test, clf1.predict(X_test), pos_label=None, average='macro')

clf2 = log()
clf2.fit(X_train, y_train)
print 'initial clf2 on train:', f1_score(y_train, clf2.predict(X_train), pos_label=None, average='macro')
print 'initial clf2 on dev:', f1_score(y_test, clf2.predict(X_test), pos_label=None, average='macro')

#SETTINGS
threshold = 2
iters = 1000
n_pos = int(round(float(y_train.count(1) / len(y_train)), 1) * 100)
n_neg = int(round(float(y_train.count(-1) / len(y_train)), 1) * 100)
print n_pos
print n_neg
scores1 = []  # keep track of f1 scores over iterations
scores2 = []


#TRAINING LOOP
for i in range(iters):
    X_U = vec.transform(unlabeled)
    #CLF1
    # get confidence above threshold
    dist = clf1.decision_function(X_U)
    idx = np.where(abs(dist) > threshold)[0]  # the indices above the threshold distance
    #get most neg and most pos samples
    top_neg = dist[idx].argsort()[:n_neg]
    top_pos = dist[idx].argsort()[-(n_pos):]
    idx_neg = idx[top_neg]
    idx_pos = idx[top_pos]
    # print 'CLF1'
    # print 'pos:',dist[idx_pos]
    # print 'neg:',dist[idx_neg]
    # add to train
    target = map(totarget, dist[idx_neg]) + map(totarget, dist[idx_pos])
    y_train += target
    train += np.array(unlabeled)[idx_neg]
    train += np.array(unlabeled)[idx_pos]
    toremove = idx_neg.tolist() + idx_pos.tolist()

    #CLF2
    # get confidence above threshold
    dist = clf2.decision_function(X_U)
    idx = np.where(abs(dist) > threshold)[0]  # the indices above the threshold distance
    #get most neg and most pos samples to add
    top_neg = dist[idx].argsort()[:n_neg]
    top_pos = dist[idx].argsort()[-(n_pos):]
    idx_neg = idx[top_neg]
    idx_pos = idx[top_pos]
    # print 'CLF2'
    # print 'pos:',dist[idx_pos]
    # print 'neg:',dist[idx_neg]
    target = map(totarget, dist[idx_neg]) + map(totarget, dist[idx_pos])
    y_train += target
    train += np.array(unlabeled)[idx_neg]
    train += np.array(unlabeled)[idx_pos]
    toremove += idx_neg.tolist() + idx_pos.tolist()

    # remove those points from unlabeled
    unlabeled = [unlabeled[x] for x in range(len(unlabeled)) if x not in toremove]
    #RE-TRAIN CLFS
    X_train = vec.transform(train)
    clf1 = svc()
    clf1.fit(X_train, y_train)
    clf2 = log()
    clf2.fit(X_train, y_train)
    # SCORE
    if i % 10 == 0:
        print 'iteration: %d (of %d)' % (i, iters)
        score1 = f1_score(y_test, clf1.predict(X_test), pos_label=None, average='macro')
        score2 = f1_score(y_test, clf2.predict(X_test), pos_label=None, average='macro')
        print 'f1 score, clf1:', score1
        print 'f1 score, clf2:', score2
        scores1.append(score1)
        scores2.append(score2)

score1 = f1_score(y_test, clf1.predict(X_test), pos_label=None, average='macro')
score2 = f1_score(y_test, clf2.predict(X_test), pos_label=None, average='macro')
print 'point %d of %d' % (i, len(unlabeled))
print 'final f1 score, clf1:', score1
print 'final f1 score, clf2:', score2
scores1.append(score1)
scores2.append(score2)

plt.plot(range(len(scores1)), scores1, label='clf1')
plt.plot(range(len(scores2)), scores2, label='clf2')
plt.xlabel('Unlabeled data-points seen')
plt.ylabel('f1-score')
plt.legend()
plt.title('Co-training')
plt.show()

# plt.savefig(
#     'threshold_plots/' + name + 'cotrainclfsingle_threshold_' + str(threshold).replace('.', '_') + 'iters=' + str(
#         iters))
# # retrain the resulting training set with full vocab, test on test set
# vec = cv()
# X_train = vec.fit_transform(train)
# X_test = vec.transform(test)
# clf = svc()
# clf.fit(X_train, y_train)
# print 'final train size:', X_train.shape[0]
# print 'final (all features):', f1_score(y_test, clf.predict(X_test), pos_label=None, average='macro')

# with open(name + 'threshold_results.txt', 'a') as f:
#     f.write('cotrainclfsingle_threshold=' + str(threshold).replace('.', '_') + 'iters=' + str(iters) + '\n')
#     f.write('best: %f iter: %d' % (np.max(scores), np.argmax(scores)))
#     f.write('\n')
