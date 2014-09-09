# coding: utf-8
__author__ = 'claire'

import gensim
from sklearn.metrics import f1_score
import codecs
import time
from utility import load_amazon
from utility import load_twitter_2class
from utility import tokenize
import numpy as np
from sklearn.naive_bayes import BernoulliNB as nb
from sklearn.feature_extraction.text import TfidfVectorizer as tf
from sklearn.linear_model import LogisticRegression as log
from sklearn.svm import LinearSVC as svm
from sklearn.metrics import classification_report


class MySentences(object):
    def __iter__(self, fname):
        for line in codecs.open(fname, 'r', 'utf-8'):
            # assume there's one document per line, tokens separated by whitespace
            yield line.lower().split()


def vectorize_gensim(data, modelname):
    """
    :param data: data is a list of strings, each string is a document
    :return: a sparse array of vectors representing word embeddings for words in word2vec model according to the modelname
    """
    model = gensim.models.Word2Vec.load(modelname)
    new_data = []
    blank = np.ndarray(model.layer1_size)  # TODO check this makes sense for comparisons
    # print model.layer1_size
    for i in data:  # for each document
        new_vec = []
        for j in tokenize(i):  # for each word in the document
            # print 'word:',j,
            try:
                new_vec.append(model[j.lower()])  #add the model representation of the word => the embedding
            except:
                pass
        new_data.append(np.mean(new_vec, axis=0))

    return new_data


def train(fname):
    start = time.time()
    print 'training model on ' + fname
    sentences = MySentences(fname)
    model = gensim.models.Word2Vec(sentences, min_count=1, size=100, workers=4)
    print 'trained model, took %1.4f minutes' % ((time.time() - start) / 60)
    model.save(fname + '.model')
    print 'saved model, took %1.4f minutes' % ((time.time() - start) / 60)
    print time.time() - start

# model=gensim.models.Word2Vec.load('review-uni_10.model')


if __name__ == "__main__":
    # train('Data/generated/10milsentences.txt')
    mapping = {u'positive': 1, u'negative': -1}

    tr_data, tr_target = load_twitter_2class('Data/twitter/twitter.train')
    te_data, te_target = load_twitter_2class('Data/twitter/twitter.test')
    # print te_target
    tr_target = [mapping[t] for t in tr_target]
    te_target = [mapping[t] for t in te_target]
    print te_target



    # embedding stuff
    #review-uni_10.model is trained from unigrams, 10 million lines of words generated from amherst data
    X_train = vectorize_gensim(tr_data, 'Data/models/10milsentences.model')

    X_test = vectorize_gensim(te_data, 'Data/models/10milsentences.model')
    print 'train size:', len(X_train), 'test size:', len(X_test)
    clf = log()
    # clf=svm()
    clf.fit(X_train, tr_target)  #fit classifier to training data
    print '\nclassifier\n----'
    print 'Accuracy on train:', clf.score(X_train, tr_target)
    print 'Accuracy on test:', clf.score(X_test, te_target)
    print '\nReport\n', classification_report(te_target, clf.predict(X_test))
    print 'F1 score', f1_score(te_target, clf.predict(X_test), pos_label=None, average='macro')
