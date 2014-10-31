from sklearn.metrics import f1_score

__author__ = 'claire'
from utility import load_twitter_2class, load_amazon
from sklearn.svm import LinearSVC as svc
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


def make_feature_split(vocabulary):
    """
    given the vocabulary of documents, return two dicts that are a random split of this vocabulary
    """
    vocab_list = [i for i in vocabulary]
    feat1 = vocab_list[:len(vocab_list) / 2]
    feat2 = vocab_list[len(vocab_list) / 2:]
    return feat1, feat2


def validate(tr, te):
    v = cv()
    X_tr = v.fit_transform(tr)
    X_te = v.transform(te)
    c = svc()
    c.fit(X_tr, y_train)
    return f1_score(y_test, c.predict(X_te), pos_label=None, average='macro')


# Load datasets

train_f = 'Data/twitter/twitter.train'
test_f = 'Data/twitter/twitter.dev'
unlabeled_f = 'Data/twitter_CST/englishtweets.both'

train, y_train = load_twitter_2class(train_f)
test, y_test = load_twitter_2class(test_f)
unlabeled = load_unlabeled_twitter(unlabeled_f)
random.shuffle(unlabeled)

name = test_f.split('/')[-1].replace('.', '-')
unlabeled = unlabeled[:5000]
random.shuffle(unlabeled)
threshold = 1
iters = 5


# initial score
vec = cv()
X_train = vec.fit_transform(train)
print 'initial train size:', X_train.shape[0]
X_test = vec.transform(test)
clf = svc()
clf.fit(X_train, y_train)
print 'initial (all features):', f1_score(y_test, clf.predict(X_test), pos_label=None, average='macro')

# do feature split to get views for co-training
# use only features from train, keep vocab fixed over iterations
view1, view2 = make_feature_split(vec.vocabulary_)
vec1 = cv(vocabulary=view1)
# vec1.fit(train)
vec2 = cv(vocabulary=view2)
# vec2.fit(train)

# initial results on different views
X_train1 = vec1.fit_transform(train)
_X_test = vec1.transform(test)
clf1 = svc()
clf1.fit(X_train1, y_train)
print 'initial view1 on train:', f1_score(y_train, clf1.predict(X_train1), pos_label=None, average='macro')
print 'initial view1 on dev:', f1_score(y_test, clf1.predict(_X_test), pos_label=None, average='macro')
X_train2 = vec2.fit_transform(train)
_X_test = vec2.transform(test)
clf2 = svc()
clf2.fit(X_train2, y_train)
print 'initial view2 on train:', f1_score(y_train, clf2.predict(X_train2), pos_label=None, average='macro')
print 'initial view2 on dev:', f1_score(y_test, clf2.predict(_X_test), pos_label=None, average='macro')

scores = []

for i in range(len(unlabeled)):
    # get confidence for both
    X_U1 = vec1.transform([unlabeled[i]])
    dist1 = clf1.decision_function(X_U1)[0]
    X_U2 = vec2.transform([unlabeled[i]])
    dist2 = clf2.decision_function(X_U2)[0]
    # check if either above threshold
    if abs(dist1) > threshold or abs(dist2) > threshold:
        # print 'dist1 %f dist2 %f' % (dist1, dist2)
        # pick the largest absolute confidence
        dist = [dist1, dist2][np.argmax([abs(dist1), abs(dist2)])]
        print 'dist:', dist
        # add to train
        target = totarget(dist)
        y_train += [target]
        train += [np.array(unlabeled)[i]]
        # re fit and train vec and clfs
        print 'len train %d' % len(train)
        X_train1 = vec1.fit_transform(train)
        clf1 = svc()
        clf1.fit(X_train1, y_train)
        X_train2 = vec2.fit_transform(train)
        clf2 = svc()
        clf2.fit(X_train2, y_train)
    if i % 100 == 0:
        # score = validate(train, test)
        v = cv()
        X_tr = v.fit_transform(train)
        X_te = v.transform(test)
        c = svc()
        c.fit(X_tr, y_train)
        score = f1_score(y_test, c.predict(X_te), pos_label=None, average='macro')
        print 'point %d of %d' % (i, len(unlabeled))
        print 'f1 score:', score
        scores.append(score)

plt.plot(range(len(scores)), scores)
plt.xlabel('Unlabeled data-points seen')
plt.ylabel('f1-score')
plt.title('Co-training')
plt.show()
plt.savefig(
    'threshold_plots/' + name + 'cotrainviewssingle_threshold_' + str(threshold).replace('.', '_') + 'iters=' + str(
        iters))
# retrain the resulting training set with full vocab, test on test set
vec = cv()
X_train = vec.fit_transform(train)
X_test = vec.transform(test)
clf = svc()
clf.fit(X_train, y_train)
print 'final train size:', X_train.shape[0]
print 'final (all features):', f1_score(y_test, clf.predict(X_test), pos_label=None, average='macro')

with open(name + 'threshold_results.txt', 'a') as f:
    f.write('cotrainviewssingle_threshold=' + str(threshold).replace('.', '_') + 'iters=' + str(iters) + '\n')
    f.write('best: %f iter: %d' % (np.max(scores), np.argmax(scores)))
    f.write('\n')

