import glob
import cPickle

__author__ = 'claire'

import codecs
from utility import load_twitter_2class, tokenize
import numpy as np
from sklearn.naive_bayes import BernoulliNB as nb
from sklearn.feature_extraction.text import TfidfVectorizer as tf
from sklearn.linear_model import LogisticRegression as log
from sklearn.svm import LinearSVC as svm
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import codecs


def vec_dict(fname):
    """
    reads in embedding file (in word2vec format) and returns dictionary where words are keys and embeddings are values
    :param fname: the file to read
    :return:dictionary of word embeddings
    """
    text = open(fname)
    d = {}
    for line in text:
        d[line.split()[0]] = np.array(line.split()[1:], dtype='float32')
    return d


def vectorize(data, model):
    """
    :param data: data is a list of strings, each string is a document
    :return: a sparse array of vectors representing word embeddings for words in word2vec model according to the modelname
    """
    new_data = []
    for i in data:  # for each document
        new_vec = []
        for j in tokenize(i.lower()):  # for each word in the document
            # print 'word:',j,
            try:
                new_vec.append(model[j.lower()])  # add the model representation of the word => the embedding
            except:
                pass
        new_vec = np.array(new_vec)
        avg = np.mean(new_vec.T, axis=1)
        max = np.max(new_vec.T, axis=1)
        min = np.max(new_vec.T, axis=1)
        all = np.concatenate((avg, max, min))
        new_data.append(all)
    return new_data


if __name__ == "__main__":

    tr_data, tr_target = load_twitter_2class('Data/twitter/twitter.train')
    te_data, te_target = load_twitter_2class('Data/twitter/twitter.dev')
    model_names = glob.glob('embeddings/*')
    print model_names
    for m in model_names:
        X_train = np.array(vectorize(tr_data, vec_dict(m)))
        X_test = np.array(vectorize(te_data, vec_dict(m)))
        for i in [log, svm]:
            print m + '\t',
            print i, '\t',
            clf = i()
            clf.fit(X_train, tr_target)  # fit classifier to training data
            # print '\nclassifier:', clf
            # print 'Accuracy on train:', clf.score(X_train, tr_target)
            # print 'Accuracy on test:', clf.score(X_test, te_target)
            # print '\nReport\n', classification_report(te_target, clf.predict(X_test))
            print 'F1 score', f1_score(te_target, clf.predict(X_test), pos_label=None, average='macro')
