"""
Shouldn't I refit the vectorizer? otherwise, what is the point of adding new data???

Using probability instead of distance
"""
from __future__ import division
from sklearn.metrics import f1_score
from collections import Counter as C

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


def totarget_distance(i):
    if i < 0:
        result = -1
    else:
        result = 1
    return result


def totarget(i):
    if i.argmax() == 0:
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
random.shuffle(unlabeled)
# unlabeled = unlabeled[:5000]
name = test_f.split('/')[-1].replace('.', '-')


# VECTORIZE
# vectorize based on initial training data
vec = cv()
X_train = vec.fit_transform(train)
X_test = vec.transform(test)
X_U = vec.transform(unlabeled)
print 'initial train size:', X_train.shape[0]

# INITIAL TRAIN
# train two classifiers on initial training data
clf1 = svc()
clf1.fit(X_train, y_train)
# print 'initial clf1 on train:', f1_score(y_train, clf1.predict(X_train), pos_label=None, average='macro')
print 'initial clf1 on dev:', f1_score(y_test, clf1.predict(X_test), pos_label=None, average='macro')

clf2 = log()
clf2.fit(X_train, y_train)
# print 'initial clf2 on train:', f1_score(y_train, clf2.predict(X_train), pos_label=None, average='macro')
print 'initial clf2 on dev:', f1_score(y_test, clf2.predict(X_test), pos_label=None, average='macro')

#SETTINGS
threshold = 0.7
iters = 1000
n_pos = int(round(float(y_train.count(1) / len(y_train)), 1) * 100)
n_neg = int(round(float(y_train.count(-1) / len(y_train)), 1) * 100)
print 'pos to add:', n_pos
print 'neg to add:', n_neg
print 'threshold:', threshold

#initial scores
scoreclf1 = []
scoreclf2 = []
scores = []
probs1 = clf1.predict_proba(X_test)
probs2 = clf2.predict_proba(X_test)
y_pred = map(totarget, probs1 * probs2)
score = f1_score(y_test, y_pred, pos_label=None, average='macro')
scoreclf1.append(f1_score(y_test, clf1.predict(X_test), pos_label=None, average='macro'))
scoreclf2.append(f1_score(y_test, clf2.predict(X_test), pos_label=None, average='macro'))
scores.append(score)

print clf1.classes_
print clf2.classes_

#TRAINING LOOP
for i in range(iters):
    X_U = vec.transform(unlabeled)
    #CLF1
    # get confidence above threshold
    # clf1.classes_: array([-1,  1])
    pred1 = clf1.predict_proba(X_U)
    neg_idx = np.where(pred1[:, 0] > threshold)[0]  # the indices above the threshold distance
    pos_idx = np.where(pred1[:, 1] > threshold)[0]
    #get most neg and most pos samples
    top_neg = pred1[:, 0][neg_idx].argsort()[-n_neg:]
    top_pos = pred1[:, 1][pos_idx].argsort()[-n_pos:]
    top_negidx1 = neg_idx[top_neg]
    top_posidx1 = pos_idx[top_pos]
    # print 'CLF1'
    # print 'clf1 neg:', top_negidx1,pred1[top_negidx1]
    # print 'clf1 pos:', top_posidx1,pred1[top_posidx1]

    #CLF2
    # get confidence above threshold
    # clf2.classes_: array([-1,  1])
    pred2 = clf2.predict_proba(X_U)
    neg_idx = np.where(pred2[:, 0] > threshold)[0]  # the indices above the threshold distance
    pos_idx = np.where(pred2[:, 1] > threshold)[0]
    #get most neg and most pos samples
    top_neg = pred2[:, 0][neg_idx].argsort()[-n_neg:]
    top_pos = pred2[:, 1][pos_idx].argsort()[-n_pos:]
    top_negidx2 = neg_idx[top_neg]
    top_posidx2 = pos_idx[top_pos]
    # print 'clf2 neg:', top_negidx1,pred1[top_negidx1]
    # print 'clf2 pos:', top_posidx1,pred1[top_posidx1]

    # add to train
    poses = [top_posidx1.tolist() + top_posidx2.tolist()][0]
    negs = [top_negidx1.tolist() + top_negidx2.tolist()][0]
    if [u for u in poses if u in negs] != []:
        break  #if they don't agree, this should never happen
    # print 'negs:', negs
    # print 'poses:', poses
    negs = list(set(negs))
    poses = list(set(poses))
    target = [-1] * len(negs) + [1] * len(poses)
    # print 'target:', target
    if target == []:  #no more points are above the threshold
        print 'no more unlabeled points above threshold'
        break
    y_train += target
    train += np.array(unlabeled)[np.array(negs + poses)]
    # train += np.array(unlabeled)[np.array(poses)]
    toremove = negs + poses
    # print 'toremove:', toremove
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
        probs1 = clf1.predict_proba(X_test)
        probs2 = clf2.predict_proba(X_test)
        # print probs1[0], probs2[0]
        # print probs1 * probs2
        # print map(totarget, probs1 * probs2)
        y_pred = map(totarget, probs1 * probs2)
        score = f1_score(y_test, y_pred, pos_label=None, average='macro')
        scoreclf1.append(f1_score(y_test, clf1.predict(X_test), pos_label=None, average='macro'))
        scoreclf2.append(f1_score(y_test, clf2.predict(X_test), pos_label=None, average='macro'))
        print 'f1 score, product of clfs:', score
        scores.append(score)


#STATS ON TRAIN
print C(y_train)
print 'iterations: %d of %d' % (i, iters)
probs1 = clf1.predict_proba(X_test)
probs2 = clf2.predict_proba(X_test)
# print probs1[0], probs2[0]
# print probs1 * probs2
# print map(totarget, probs1 * probs2)
y_pred = map(totarget, probs1 * probs2)
score = f1_score(y_test, y_pred, pos_label=None, average='macro')
scores.append(score)
scoreclf1.append(f1_score(y_test, clf1.predict(X_test), pos_label=None, average='macro'))
scoreclf2.append(f1_score(y_test, clf2.predict(X_test), pos_label=None, average='macro'))
print 'final f1 score, product of clfs:', score

plt.plot(range(len(scores)), scores, label='combined clfs')
plt.plot(range(len(scoreclf1)), scoreclf1, label='clf1')
plt.plot(range(len(scoreclf2)), scoreclf2, label='clf2')

plt.xlabel('Unlabeled data-points seen (*10)')
plt.ylabel('f1-score')
plt.legend()
plt.title('Co-training')
plt.show()

# plt.savefig(
#     'threshold_plots/' + name + 'cotrainclfsingle_threshold_' + str(threshold).replace('.', '_') + 'iters=' + str(
#         iters))

# with open(name + 'threshold_results.txt', 'a') as f:
#     f.write('cotrainclfsingle_threshold=' + str(threshold).replace('.', '_') + 'iters=' + str(iters) + '\n')
#     f.write('best: %f iter: %d' % (np.max(scores), np.argmax(scores)))
#     f.write('\n')
