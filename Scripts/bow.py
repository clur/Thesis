__author__ = 'claire'
import codecs
from sklearn.linear_model.stochastic_gradient import SGDClassifier as sgd
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
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


def load_twitter(fname):
    """
    open and read fname, split into data and target
    :param fname: filename to read
    :return: data, list and target, list
    """
    raw = codecs.open(fname, 'rb', 'utf-8').readlines()  # load and split data into reviews
    target = [r.split('\t')[2] for r in raw]  # target is pos,neg,neutral
    data = [r.split('\t')[3] for r in raw]  # review text
    data = [d.lower().strip() for d in data]
    return data, target


def load_amazon(fname):
    """
    open and read fname, split into data and target
    :param fname: filename to read
    :return: data, list and target, list
    """
    raw = open(fname, 'r').readlines()  # load and split data into reviews
    # random.shuffle(raw)
    target = [int(float((r.split('\t')[5]))) for r in raw]  # target is pos,neg,neutral
    data = [r.split('\t')[6:] for r in raw]  # title and review text
    return data, target


def print_top10(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    s = ''
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        s += "%s: %s\n" % (class_label, ", ".join(feature_names[j] for j in top10))
    return s


def bow_clf_twitter(trainfile, testfile, clf):
    '''read in train and test file and perform classification experiment with sklearn bow features and clf'''

    mapping = {u'positive': 1, u'negative': -1}

    tr_data, tr_target = load_twitter_2class('Data/twitter/twitter.train')
    te_data, te_target = load_twitter_2class('Data/twitter/twitter.test')


    tr_target = [mapping[t] for t in tr_target]
    te_target = [mapping[t] for t in te_target]
    # print tr_data[0]
    vec = cv(ngram_range=(1, 1))  # basic tfidf vectorizer
    print vec
    vec.fit(tr_data)
    X_train = vec.transform(tr_data)
    X_test = vec.transform(te_data)
    clf = clf()
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


def bow_clf_amazon(trainfile, testfile):
    '''read in train and test file and perform classification experiment with sklearn bow features and clf'''

    tr_data, tr_target = load_amazon(trainfile)
    te_data, te_target = load_amazon(testfile)
    mapping = {1: -1, 2: -1, 3: 0, 4: 1, 5: 1}
    tr_target = [mapping[t] for t in tr_target]
    te_target = [mapping[t] for t in te_target]
    # print tr_data[0]
    vec = tf(ngram_range=(1, 1), stop_words='english')  # basic tfidf vectorizer
    print vec
    print 'TFIDF FITTING'
    vec.fit(tr_data)
    print 'TFIDF FIT'
    print 'TFIDF TRANSFORMING'
    X_train = vec.transform(tr_data)
    X_test = vec.transform(te_data)
    print 'TRANSFORMED'
    clf = log()
    print clf
    clf.fit(X_train, tr_target)
    print 'data:\ntrain size: %s test size: %s' % (str(len(tr_target)), str(len(te_target)))
    print 'train set class representation:' + str(Counter(tr_target))
    print 'test set class representation: ' + str(Counter(te_target))
    print '\nclassifier\n----'
    print 'Accuracy on train:', clf.score(X_train, tr_target)
    print 'Accuracy on test:', clf.score(X_test, te_target)
    print '\nReport\n', classification_report(te_target, clf.predict(X_test))


def bow_clf_twitter_grid(trainfile, testfile):
    '''read in train and test file and perform classification experiment with sklearn bow features and clf'''

    tr_data, tr_target = load_twitter(trainfile)
    te_data, te_target = load_twitter(testfile)
    mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
    tr_target = [mapping[t] for t in tr_target]
    te_target = [mapping[t] for t in te_target]
    # print tr_data[0]

    pipeline = Pipeline([
        ('tf', tf()),
        ('clf', nb()),
    ])

    params = {
        'tf__ngram_range': ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5)),
        'tf__use_idf': (True, False),
        'tf__norm': ('l1', 'l2', None),
        # 'clf__class_weight':('auto',None)
    }
    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, params, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print params
    import time

    t0 = time.time()
    grid_search.fit(tr_data, tr_target)

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def bow_clf_twitter_grid2(trainfile, testfile):
    '''read in train and test file and perform classification experiment with sklearn bow features and clf'''

    tr_data, tr_target = load_twitter(trainfile)
    te_data, te_target = load_twitter(testfile)
    mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
    tr_target = [mapping[t] for t in tr_target]
    te_target = [mapping[t] for t in te_target]
    # print tr_data[0]

    pipeline = Pipeline([
        ('tf', tf()),
        ('clf', nb()),
    ])

    params = {
        # 'tf__ngram_range':((1,1),(1,2),(1,3),(1,4),(1,5)),
        'tf__use_idf': (True, False),
        'tf__norm': ('l1', 'l2', None),
        # 'clf__class_weight':('auto',None)
    }
    # find the best parameters for both the feature extraction and the
    # classifier
    clf = GridSearchCV(pipeline, params, n_jobs=-1, verbose=1)
    clf.fit(tr_data, tr_target)
    print "Best parameters set found on development set:"
    print
    print clf.best_estimator_

    print "Grid scores on development set:"
    print
    for params, mean_score, scores in clf.grid_scores_:
        print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params)
    print
    '''
    print "Detailed classification report:"
    print
    print "The model is trained on the full development set."
    print "The scores are computed on the full evaluation set."
    print
    y_true, y_pred = y_test, clf.predict X_test
    print classification_report(y_true, y_pred)
    print
    '''


def scorer(true, pred):
    """
    use classification report method on files, format is -1,0,1
    :param true: e.g. 'Data/twitter/twitter.test.true'
    :param pred: e.g. 'Data/twitter/amherst.test.pred'
    :return:
    """

    print 'TEST:', true
    print 'PRED:', pred
    true = open(true).readlines()
    pred = open(pred).readlines()
    print '\nReport\n'
    print classification_report(true, pred)


def bow_clf_general(trainfile, testfile):
    '''read in train and test file and perform classification experiment with sklearn bow features and clf'''

    tr_data, tr_target = load_amazon(trainfile)
    te_data, te_target = load_amazon(testfile)
    mapping = {1: -1, 2: -1, 3: 0, 4: 1, 5: 1}
    tr_target = [mapping[t] for t in tr_target]
    te_target = [mapping[t] for t in te_target]
    # print tr_data[0]
    vec = tf(ngram_range=(1, 1), stop_words='english')  # basic tfidf vectorizer
    print vec
    print 'TFIDF FITTING'
    vec.fit(tr_data)
    print 'TFIDF FIT'
    print 'TFIDF TRANSFORMING'
    X_train = vec.transform(tr_data)
    X_test = vec.transform(te_data)
    print 'TRANSFORMED'
    clf = log()
    print clf
    clf.fit(X_train, tr_target)
    print 'data:\ntrain size: %s test size: %s' % (str(len(tr_target)), str(len(te_target)))
    print 'train set class representation:' + str(Counter(tr_target))
    print 'test set class representation: ' + str(Counter(te_target))
    print '\nclassifier\n----'
    print 'Accuracy on train:', clf.score(X_train, tr_target)
    print 'Accuracy on test:', clf.score(X_test, te_target)
    print '\nReport\n', classification_report(te_target, clf.predict(X_test))

if __name__ == "__main__":

    pass
    for clf in [log,svm]:
        bow_clf_twitter('Data/twitter/twitter.train', 'Data/twitter/twitter.test', clf)
        print '-' * 10