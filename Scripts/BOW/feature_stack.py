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
from sklearn.pipeline import FeatureUnion
from collections import Counter
import codecs


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


def vectorize_gensim(data, model):
    """
    :param data: data is a list of strings, each string is a document
    :return: a sparse array of vectors representing word embeddings for words in word2vec model according to the modelname
    """
    new_data = []
    for i in data:  # for each document
        new_vec = []
        for j in i.split():  # for each word in the document
            try:
                new_vec.append(model[j.lower()])  # add the model representation of the word => the embedding
            except:
                pass
        avg = (np.mean(new_vec, axis=0))
        max = (np.max(new_vec, axis=0))
        min = (np.max(new_vec, axis=0))
        all = np.concatenate((avg, max, min))
        new_data.append(all)
    return new_data


if __name__ == "__main__":
    datafolder = '/Users/claire/Dropbox/PycharmProjects/Thesis/Scripts/Data/'
    trainfile = datafolder + 'twitter/twitter.train'
    testfile = datafolder + 'twitter/twitter.dev'
    tr_data, tr_target = load_twitter_2class(trainfile)
    te_data, te_target = load_twitter_2class(testfile)

    vec1 = cv(analyzer='word', ngram_range=(1, 4))
    vec2 = cv(analyzer='char_wb', ngram_range=(1, 4))

    combined_features = FeatureUnion([("word", vec1), ("char", vec2)])
    print combined_features

    # Use combined features to transform dataset:
    print 'Fit transform data'
    X_train = combined_features.fit_transform(tr_data)
    print X_train.shape
    X_test = combined_features.transform(te_data)
    # X_train = vec2.fit_transform(tr_data)
    # X_test = vec2.transform(te_data)

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
        print



        # anders twitter text 200k
        # train=open('Data/twitter_emoticon_embeddings/posneg_200k.labeled','r').readlines()
        # tr_target=[int(i.split()[0]) for i in train]
        # tr_data=[' '.join(i.split()[3:]) for i in train]