import cPickle

__author__ = 'claire'

# from numpy.random import random_sample
from sklearn.naive_bayes import BernoulliNB as nb
from sklearn.feature_extraction.text import TfidfVectorizer as tf
from sklearn.feature_extraction.text import CountVectorizer as cv
from sklearn.linear_model import LogisticRegression as log
from sklearn.svm import LinearSVC as svm
import numpy as np
from sklearn.metrics import classification_report, f1_score
import gensim
import re
from utility import load_twitter_2class
from collections import Counter
import codecs


def load_amazon(fname):
    """
    open and read fname, split into data and target
    :param fname: filename to read
    :return: data, list and target, list
    """
    target = []
    data = []
    raw = codecs.open(fname, 'r', 'utf8')  # load and split data into reviews
    # random.shuffle(raw)
    for r in raw:
        try:
            target.append(int(float(r.split('\t')[5])))
            data.append(''.join(r.split('\t')[6:]))
        except:
            pass
    raw.close()
    mapping = {1: -1, 2: -1, 4: 1, 5: 1}
    target = [mapping[t] for t in target]
    assert len(target) == len(data)
    return data, target


def load_twitter_2class(fname):
    """
    open and read fname, split into data and target
    :param fname: filename to read
    :return: data, list and target, list
    """
    raw = codecs.open(fname, 'r', 'utf-8').readlines()  # load and split data into reviews
    raw = [r for r in raw if r.split('\t')[2] != u'neutral']
    target = [r.split('\t')[2] for r in raw]  # target is pos,neg
    data = [r.split('\t')[3] for r in raw]  # review text
    data = [d.lower().strip() for d in data]
    mapping = {'negative': -1, 'positive': 1}
    target = [mapping[t] for t in target]

    return data, target


if __name__ == "__main__":
    trainfile = 'Data/amazon_embeddings/pos_neg.txt'
    testfile = 'Data/twitter/twitter.test'
    tr_data, tr_target = load_amazon(trainfile)
    te_data, te_target = load_twitter_2class(testfile)
    # print tr_data[0]
    vec = tf(ngram_range=(1, 1), stop_words='english')  # basic tfidf vectorizer
    print vec
    print 'TFIDF FITTING'
    vec.fit(tr_data)
    print len(vec.vocabulary_)
    cPickle.dump(vec.vocabulary_, open('tf.vocabulary', 'wb'))
    print 'TFIDF FIT'
    print 'TFIDF TRANSFORMING'
    X_train = vec.transform(tr_data)
    X_test = vec.transform(te_data)
    print 'TRANSFORMED'
    for i in [log, svm]:
        clf = i()
        print clf
        clf.fit(X_train, tr_target)
        print 'data:\ntrain size: %s test size: %s' % (str(len(tr_target)), str(len(te_target)))
        print 'train set class representation:' + str(Counter(tr_target))
        print 'test set class representation: ' + str(Counter(te_target))
        print '\nclassifier\n----'
        print 'Accuracy on train:', clf.score(X_train, tr_target)
        print 'Accuracy on test:', clf.score(X_test, te_target)
        print '\nReport\n', classification_report(te_target, clf.predict(X_test))
        print 'F1 score', f1_score(te_target, clf.predict(X_test), pos_label=None, average='macro')



        # anders twitter text 200k
        # train=open('Data/twitter_emoticon_embeddings/posneg_200k.labeled','r').readlines()
        # tr_target=[int(i.split()[0]) for i in train]
        # tr_data=[' '.join(i.split()[3:]) for i in train]