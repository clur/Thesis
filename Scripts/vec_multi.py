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




def vectorize_multi(data, models):
    """
    models is a list of models
    """
    # model = gensim.models.Word2Vec.load(modelname)
    # print 'vocab of model:',len(model.vocab)
    new_data = []
    for i in data:  # for each document
        # print 'length of tweet:',len(tokenize(i.lower()))
        new_vec = []
        for j in tokenize(i.lower()):  # for each word in the document
            try:
                w_embed = [m[j] for m in models]  # list of embeddings from each model, word has to be in both models
                new_vec.append(w_embed)  # add embedding of each word, represents 1 tweet, w_embed is 2*100
            except:
                pass
        new_vec = np.array(new_vec)
        avg = np.mean(new_vec, axis=0).flatten()  # concatenate the means of both pos and neg word embeddings
        max = np.max(new_vec, axis=0).flatten()
        min = np.max(new_vec, axis=0).flatten()
        all = np.concatenate((avg, max, min))
        print all.shape
        new_data.append(all)
    return new_data


if __name__ == "__main__":

    tr_data, tr_target = load_twitter_2class('Data/twitter/twitter.train')
    te_data, te_target = load_twitter_2class('Data/twitter/twitter.dev')
    model_names = glob.glob('embeddings/twitter*')
    print model_names
    models = map(vec_dict, model_names)

    X_train = np.array(vectorize_multi(tr_data, models))
    X_test = np.array(vectorize_multi(te_data, models))
    for i in [log, svm]:
        print model_names, '\t',
        print i, '\t',
        clf = i()
        clf.fit(X_train, tr_target)  # fit classifier to training data
        # print '\nclassifier:', clf
        # print 'Accuracy on train:', clf.score(X_train, tr_target)
        # print 'Accuracy on test:', clf.score(X_test, te_target)
        # print '\nReport\n', classification_report(te_target, clf.predict(X_test))
        print f1_score(te_target, clf.predict(X_test), pos_label=None, average='macro')