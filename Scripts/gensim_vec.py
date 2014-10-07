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
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class MySentences(object):
    def __init__(self, fname):
        self.file = fname
        self.token = re.compile(r"(?u)\b\w\w+\b\'*\w*")

    def __iter__(self):
        for line in codecs.open(self.file, 'r'):
            # assume there's one document per line, tokens separated by whitespace

            # yield line.lower().split()
            # print ''.join(line.split('\t')[6:])
            yield self.token.findall(''.join(line.split('\t')[6:]).lower())


def vectorize_gensim_posneg2(data, pos_model, neg_model):
    """
    :param data: data is a list of strings, each string is a document
    :return: a sparse array of vectors representing word embeddings for words in word2vec model according to the modelname
    """
    # model = gensim.models.Word2Vec.load(modelname)
    # print 'vocab of model:',len(model.vocab)
    new_data = []
    for i in data:  # for each document
        new_vec = []
        for j in tokenize(i):  # for each word in the document
            # print 'word:',j,
            j = j.lower()
            if j in pos_model and j in neg_model:  # word is oov for both datasets
                if j in pos_model and j not in neg_model:  # word is oov for neg dataset
                    w_embed = np.array([pos_model[j], n_oov])

                else:
                    w_embed = np.array([p_oov, neg_model[j]])

                new_vec.append(w_embed)  # add embedding of each word, represents 1 tweet

        avg = np.mean(new_vec, axis=0).flatten()  # concatenate the means of both pos and neg word embeddings
        # max = (np.max(new_vec, axis=0)).flatten()
        # min = (np.max(new_vec, axis=0)).flatten()
        # all = np.concatenate((avg,max, min))
        new_data.append(avg)

    return new_data


def vectorize_gensim_posneg(data, pos_model, neg_model):
    """
    :param data: data is a list of strings, each string is a document
    :return: len(data) * 200 array composed of avg(pos)+avg(neg) embeddings for each word in tweet.
    """
    # model = gensim.models.Word2Vec.load(modelname)
    # print 'vocab of model:',len(model.vocab)
    new_data = []
    for i in data:  # for each document
        new_vec = []
        for j in tokenize(i):  # for each word in the document
            # print 'word:',j,
            j = j.lower()
            if j not in pos_model and j not in neg_model:  # word is oov for both datasets
                w_embed = pn_oov
                # print j
                # print w_embed
            elif j in pos_model and j not in neg_model:  # word is oov for neg dataset
                w_embed = np.array([pos_model[j], n_oov])

            else:
                w_embed = np.array([p_oov, neg_model[j]])

            new_vec.append(w_embed)  # add embedding of each word, represents 1 tweet, w_embed is 2*100
        # avg=np.mean(new_vec,axis=0).flatten() # concatenate the means of both pos and neg word embeddings
        max = (np.max(new_vec, axis=0)).flatten()
        min = (np.max(new_vec, axis=0)).flatten()
        all = np.concatenate((max, min))
        new_data.append(all)

    return new_data


def vectorize_gensim_posnegboth(data, pos_model, neg_model, both_model):
    """
    embeddings from pos, neg and both texts
    """
    # model = gensim.models.Word2Vec.load(modelname)
    # print 'vocab of model:',len(model.vocab)
    new_data = []
    for i in data:  # for each document
        new_vec = []
        for j in tokenize(i):  # for each word in the document
            # print 'word:',j,
            j = j.lower()
            if j not in both_model:  # word is oov, since both_model gets words from both the negative and positive combined text
                w_embed = pnboth_oov
                # print j
                # print w_embed
            elif j not in pos_model and j not in neg_model:  # TODO for some reason, words are in both that are not in pos or neg???
                w_embed = np.array([p_oov, n_oov, both_model[j]])
            elif j in pos_model and j not in neg_model:  # word is oov for neg dataset
                w_embed = np.array([pos_model[j], n_oov, both_oov])
            else:
                w_embed = np.array([p_oov, neg_model[j], both_oov])  # word is oov for pos dataset
            new_vec.append(w_embed)  # add embedding of each word, represents 1 tweet, w_embed is 2*100
        avg = np.mean(new_vec, axis=0).flatten()  # concatenate the means of both pos and neg word embeddings
        # max = (np.max(new_vec, axis=0)).flatten()
        # min = (np.max(new_vec, axis=0)).flatten()
        # all = np.concatenate((avg, max, min))
        new_data.append(avg)

    return new_data


def vectorize_gensim(data, model):
    """
    :param data: data is a list of strings, each string is a document
    :return: a sparse array of vectors representing word embeddings for words in word2vec model according to the modelname
    """
    new_data = []
    for i in data:  # for each document
        new_vec = []
        for j in tokenize(i):  # for each word in the document
            # print 'word:',j,
            try:
                new_vec.append(model[j.lower()])  #add the model representation of the word => the embedding
            except:
                pass
        avg = (np.mean(new_vec, axis=0))
        # max = (np.max(new_vec, axis=0))
        # min = (np.max(new_vec, axis=0))
        # all = np.concatenate((avg, max, min))
        new_data.append(avg)
    return new_data


def plot_words(words, model):
    # TODO fix
    w = [k for k in model.vocab.keys()]
    allvec = np.array([model[word] for word in w])
    idx = [w.index(i) for i in words]
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(allvec)
    data = np.array([reduced[i] for i in idx])
    labels = words
    plt.scatter(data[:, 0], data[:, 1])

    for label, x, y in zip(labels, data[:, 0], data[:, 1]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(-2, 2),
            textcoords='offset points', ha='right', va='bottom')
    plt.show()


def train(fname):
    start = time.time()
    print 'training model on ' + fname
    sentences = MySentences(fname)
    model = gensim.models.Word2Vec(sentences, min_count=10, size=100, workers=4)
    print 'trained model, took %1.4f minutes' % ((time.time() - start) / 60)
    model.save(fname + '.model')
    print 'saved model, took %1.4f minutes' % ((time.time() - start) / 60)

# model=gensim.models.Word2Vec.load('review-uni_10.model')


if __name__ == "__main__":

    # train('Data/generated/10milsentences.txt')
    mapping = {u'positive': 1, u'negative': -1}

    tr_data, tr_target = load_twitter_2class('Data/twitter/twitter.train')
    te_data, te_target = load_twitter_2class('Data/twitter/twitter.test')
    # print te_target
    tr_target = [mapping[t] for t in tr_target]
    te_target = [mapping[t] for t in te_target]


    pos = 'Data/models/out_pos_equal.txt.model'
    neg = 'Data/models/out_neg.txt.model'
    both = 'Data/models/pos_neg.txt.model'


    pos_model = gensim.models.Word2Vec.load(pos)
    neg_model = gensim.models.Word2Vec.load(neg)
    both_model = gensim.models.Word2Vec.load(both)
    print 'vocab of pos model:', len(pos_model.vocab)
    print 'vocab of neg model:', len(neg_model.vocab)
    print 'vocab of both model:', len(both_model.vocab)
    # compute oov (means of each model)

    w = [k for k in pos_model.vocab.keys()]
    allvec = [pos_model[word] for word in w]
    p_oov = np.mean(allvec, axis=0)  # value when word has no pos embedding
    w = [k for k in neg_model.vocab.keys()]
    allvec = [neg_model[word] for word in w]
    n_oov = np.mean(allvec, axis=0)  # value when word has no neg embedding

    pn_oov = np.array(
        [p_oov, n_oov])  # value when word has no embedding, used in the case when looking at pos and neg model together

    w = [k for k in both_model.vocab.keys()]
    allvec = [both_model[word] for word in w]
    both_oov = np.mean(allvec, axis=0)  # value when word has no embedding

    pnboth_oov = np.array([p_oov, n_oov,
                           both_oov])  #todo need to check if this makes sense or can just use pn_oov here? embedding is different learned from the combined text so i think this might be necessary

    # embedding stuff
    #review-uni_10.model is trained from unigrams, 10 million lines of words generated from amherst data
    # X_train = vectorize_gensim(tr_data, model)
    # X_test = vectorize_gensim(te_data, model)

    # X_train = np.array(vectorize_gensim_posneg2(tr_data, pos_model, neg_model))
    # X_test = np.array(vectorize_gensim_posneg2(te_data, pos_model, neg_model))

    X_train = np.array(vectorize_gensim_posnegboth(tr_data, pos_model, neg_model, both_model))
    X_test = np.array(vectorize_gensim_posnegboth(te_data, pos_model, neg_model, both_model))

    print X_train.shape
    print 'train size:', len(X_train), 'test size:', len(X_test)

    for i in [log, svm]:
        clf = i(C=1e5)
        clf.fit(X_train, tr_target)  # fit classifier to training data
        print '\nclassifier:', clf
        print 'Accuracy on train:', clf.score(X_train, tr_target)
        print 'Accuracy on test:', clf.score(X_test, te_target)
        print '\nReport\n', classification_report(te_target, clf.predict(X_test))
        print 'F1 score', f1_score(te_target, clf.predict(X_test), pos_label=None, average='macro')



