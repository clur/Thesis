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
    feat_dict1 = dict((k, v) for (k, v) in zip(feat1, range(len(feat1))))
    feat_dict2 = dict((k, v) for (k, v) in zip(feat1, range(len(feat2))))
    return feat_dict1, feat_dict2

# Load datasets
train, y_train = load_twitter_2class('Data/twitter/twitter.train')
test, y_test = load_twitter_2class('Data/twitter/twitter.dev')
unlabeled = load_unlabeled_twitter('Data/twitter_CST/englishtweets.both')
# unlabeled=unlabeled[:5000]
random.shuffle(unlabeled)
# Fit vectorizer with training data and transform datasets

# build the feature split for co-training
vec = cv()
vec.fit(train)
view1, view2 = make_feature_split(vec.vocabulary_)

'''
vec = tf()
X_train = vec.fit_transform(train)
X_test = vec.transform(test)
X_U = vec.transform(unlabeled)


# train classifier on labeled data
clf = svc()
clf.fit(X_train, y_train)

iters=30
threshold = 1
added = 0
scores = []  #keep track of how it changes according to the development set
for i in range(iters):
    print 'iteration %d' % i
    # find points above threshold to add to training data
    distance = clf.decision_function(X_U)  # the distance (- or +) from the hyperplane
    idx= np.where(abs(distance)> threshold)[0] # the indices above the threshold distance
    new=np.random.choice(idx) #pick one to add
    target = map(totarget, [distance[new]])
    y_train+=target
    train.append(np.array(unlabeled)[new].tolist())

    # remove from unlabeled
    unlabeled.pop(new)
    X_train = vec.fit_transform(train)
    X_test = vec.transform(test)
    X_U = vec.transform(unlabeled)
    clf=svc()
    clf.fit(X_train, y_train)
    if i % 10 == 0:
        scores.append(f1_score(y_test, clf.predict(X_test), pos_label=None, average='macro'))
        # print 'added %d unlabeled datapoints' % added
        print 'Iteration %d : accuracy: %f ' % (i, scores[-1])
print scores.append(f1_score(y_test, clf.predict(X_test), pos_label=None, average='macro'))

with open('threshold_results.txt','a') as f:
    f.write('threshold_plots/single_threshold='+str(threshold).replace('.','_')+'iters='+str(iters)+'\n')
    f.write('best: %f iter: %d' %(np.max(scores), np.argmax(scores)*10))
    f.write('\n')

plt.plot([i*10 for i in range(len(scores))], scores)
plt.title('single_threshold='+str(threshold).replace('.','_')+'batches='+str(iters))
plt.xlabel('iters')
plt.ylabel('accuracy')
# plt.show()
plt.savefig('threshold_plots/single_threshold='+str(threshold).replace('.','_')+'iters='+str(iters))
'''