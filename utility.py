__author__ = 'claire'
import codecs
import random

from numpy.random import random_sample
from sklearn.feature_extraction.text import TfidfVectorizer as tf
from sklearn.linear_model import LogisticRegression as log
import numpy as np
from sklearn.metrics import classification_report
import gensim
import re
from collections import defaultdict
import Words


def prob_sample(values, probabilities, size):
    """returns a *random* sample of length size of the values based on probabilities
    values=list of length N
    probabilities=list of length N
    size=integer
    """
    bins = np.add.accumulate(probabilities)
    return values[np.digitize(random_sample(size), bins)]


def load_twitter(fname):
    """
    open and read fname, split into data and target
    :param fname: filename to read
    :return: data, list and target, list
    """
    raw = codecs.open(fname, 'r', 'utf-8').readlines()  # load and split data into reviews
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
    raw = codecs.open(fname, 'rb', 'latin1').readlines()  # load and split data into reviews
    random.shuffle(raw)
    target = [r.split('\t')[5] for r in raw]  # target is pos,neg,neutral
    data = [' '.join(r.split('\t')[6:]) for r in raw]  # review text
    data = [d.lower().strip() for d in data]
    return data, target


def print_top10(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    s = ''
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        s += "%s: %s\n" % (class_label, ", ".join(feature_names[j] for j in top10))
    return s


def bow_clf(trainfile, testfile):
    '''read in train and test file and perform classification experiment with sklearn bow features and clf'''

    tr_data, tr_target = load_amazon(trainfile)
    te_data, te_target = load_amazon(testfile)
    vec = tf()  # basic tfidf vectorizer
    vec.fit(tr_data)
    X_train = vec.transform(tr_data)
    X_test = vec.transform(te_data)
    clf = log()
    clf.fit(X_train, tr_target)  # fit classifier to training data
    print '\nclassifier\n----'
    print 'Accuracy on train:', clf.score(X_train, tr_target)
    print 'Accuracy on test:', clf.score(X_test, te_target)
    print '\nReport\n', classification_report(te_target, clf.predict(X_test))


def tokenize(text):
    """
    return list of tokens from string text based on token pattern
    :param text: string
    :return:list
    """
    token_pattern = r"(?u)\b\w\w+\b\'*\w*"  # from sklearn.text.py
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall(text)


def vectorize(data, modelname):
    """
    :param data: data is a list of strings, each string is a document
    :return: a sparse array of vectors representing data
    """
    model = gensim.models.Word2Vec.load(modelname)
    new_data = []
    # oovdic=defaultdict(int)
    blank = np.ndarray(model.layer1_size)  # TODO check this makes sense for comparisons
    for i in data:
        new_vec = []
        for j in tokenize(i):
            try:
                new_vec.append(model[j])
            except:
                new_vec.append(blank)  # to ensure document vectors are same size
                # oovdic[j]+=1
        new_data.append(new_vec)
    # print oovdic
    return new_data


def avg_rating(text, rating_dict):
    '''
    calculates the average sentiment score of a text as the average over ratings of unigrams in text based on
    amherst rating table.
    :param fname: textfile name to be scored
    :param rating_dict: Words().dic
    :return: the average rating over words in text
    '''
    score = []
    for i in tokenize(text):
        print 'word:', i
        print 'rating of word:', rating_dict[i]
        try:
            print rating_dict[i]
            score.append(rating_dict[i])
        except:
            print i
    print np.mean(score)


if __name__ == "__main__":
    # values = np.array([1.1, 2.2, 3.3])
    # probabilities = np.array([0.1, 0.9, 0.0])
    #
    # print prob_sample(values, probabilities, 10)
    #
    # print load_twitter('Data/twitter/twitter.train')[1][:100]

    # bow_clf('Data/amazon/review.train','Data/amazon/review.test')

    tr_data, tr_target = load_amazon('Data/amazon/review.train')
    te_data, te_target = load_amazon('Data/amazon/review.test')

    # X_train=vectorize(tr_data,'review-uni_10.model')
    # X_test=vectorize(te_data,'review-uni_10.model')
    # np.save('temp',X_test[0])
    # clf = log()
    # clf.fit(X_train, tr_target)  #fit classifier to training data
    # print '\nclassifier\n----'
    # print 'Accuracy on train:', clf.score(X_train, tr_target)
    # print 'Accuracy on test:', clf.score(X_test, te_target)
    # print '\nReport\n', classification_report(te_target, clf.predict(X_test))
    W = Words.Words()
    W.build_dicts()
    for i in W.rating_dict.keys():
        print len(W.rating_dict[i])

    # inv_rating = {v:k for k, v in W.rating_dict.items()}

    # avg_rating(tr_data[0],inv_rating) #todo invert list so keys are words and ratings are values

    print tr_data[0]
    # pass
