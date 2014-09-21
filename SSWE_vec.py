__author__ = 'claire'
import codecs
from sklearn.linear_model.stochastic_gradient import SGDClassifier as sgd
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
# from numpy.random import random_sample
from sklearn.metrics import f1_score
from sklearn.naive_bayes import BernoulliNB as nb
from sklearn.feature_extraction.text import TfidfVectorizer as tf
from sklearn.linear_model import LogisticRegression as log
from sklearn.svm import LinearSVC as svm
import numpy as np
from sklearn.metrics import classification_report
# import gensim
# import re
# from collections import Counter
from utility import tokenize, load_amazon, load_twitter, load_twitter_2class


def build_embeddings_dict(f):
    # f='Data/duyu_tang_embedding-results/sswe-h.txt'
    data = [i.strip() for i in open(f).readlines()]
    words = [i.split('\t')[0] for i in data]
    vecs = [i.split('\t')[1:] for i in data]
    vecs = [np.array(i, np.dtype('float')) for i in vecs]
    d = {}
    for i in range(len(words)):
        d[words[i]] = vecs[i]
    return d


def vectorize_sswe(data, embedding_dict):
    '''
    The returned array is the average of the SSWE vectors for words contained in the example

    :param data: the data to turn into vectors, list, each element an example
    :param embedding_dict: dict created in build_embeddings_dict
    :return: 1 * 150 np float array
    '''
    vectors = []
    for example in data:
        score = []
        for i in tokenize(example.lower()):
            try:
                score.append(embedding_dict[i])
                # print i,':',embedding_dict[i][:2]
            except:
                pass
        avg = (np.mean(score, axis=0))
        # max = (np.max(score, axis=0))
        # min = (np.max(score, axis=0))
        # all = np.concatenate((avg, max, min))
        # print len(all)
        vectors.append(avg)
    return vectors


def embed_clf_twitter(trainfile, testfile, embedding_dict_file, clf):
    '''read in train and test file and perform classification experiment with sklearn bow features and clf'''

    tr_data, tr_target = load_twitter_2class(trainfile)
    te_data, te_target = load_twitter_2class(testfile)
    mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
    tr_target = [mapping[t] for t in tr_target]
    te_target = [mapping[t] for t in te_target]
    # print 'BUILDING EMBEDDING DICT'
    embedding_dict = build_embeddings_dict(embedding_dict_file)
    # print tr_data[0]
    # vec = tf(ngram_range=(1, 1), max_df=0.9, min_df=0.01, norm=None, use_idf=False)  # basic tfidf vectorizer
    # print 'AVERAGING EMBEDDINGS'
    X_train = vectorize_sswe(tr_data, embedding_dict)
    X_test = vectorize_sswe(te_data, embedding_dict)
    # clf = log()
    # clf=nb()
    # clf=svm()
    clf = clf()
    print embedding_dict_file.split('/')[-1]
    print clf
    clf.fit(X_train, tr_target)
    # print 'data:\ntrain size: %s test size: %s' % (str(len(tr_target)), str(len(te_target)))
    # print 'train set class representation:' + str(Counter(tr_target))
    # print 'test set class representation: ' + str(Counter(te_target))
    # print '\nclassifier\n----'
    print 'Accuracy on train:', clf.score(X_train, tr_target)
    print 'Accuracy on test:', clf.score(X_test, te_target)
    y_pred = clf.predict((X_test))
    print '\nReport\n', classification_report(te_target, y_pred)
    print 'F1 Macro:', f1_score(te_target, y_pred, pos_label=None, average='macro')
    print


if __name__ == "__main__":

    # pass

    for clf in [nb, log, svm]:
        embed_clf_twitter('Data/twitter/twitter.train', 'Data/twitter/twitter.test',
                          'Data/duyu_tang_embedding-results/sswe-h.txt', clf)

        print '-' * 10
        embed_clf_twitter('Data/twitter/twitter.train', 'Data/twitter/twitter.test',
                          'Data/duyu_tang_embedding-results/sswe-r.txt', clf)
        print '-' * 10
        embed_clf_twitter('Data/twitter/twitter.train', 'Data/twitter/twitter.test',
                          'Data/duyu_tang_embedding-results/sswe-u.txt', clf)
        print '-' * 10
        print '-' * 10
