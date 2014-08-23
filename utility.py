__author__ = 'claire'
import codecs
import random

from sklearn.svm import LinearSVC as svm
from sklearn.naive_bayes import BernoulliNB as nb
# from numpy.random import random_sample
from sklearn.feature_extraction.text import TfidfVectorizer as tf
from sklearn.linear_model import LogisticRegression as log
import numpy as np
from sklearn.metrics import classification_report
import gensim
import re
from collections import defaultdict, Counter
import Words


def labelstext_012(textfile):
    '''
    FOR TWITTER
    convert labels text to 012
    :param textfile:
    :return:
    '''
    text = codecs.open(textfile, 'r', 'utf8').readlines()
    target = [r.split('\t')[2] for r in text]
    map = {'positive': 0, 'neutral': 1, 'negative': 2}
    with open(textfile + '012.true', 'w') as f:
        for t in target:
            f.write(str(map[t]))
            f.write('\n')


def labels012_text(textfile):
    pass


# todo write more simple mapping functions from words to numbers
# todo integrate scorer



def prob_sample(values, probabilities, size):
    """returns a *random* sample of length size of the values based on probabilities
    values=list of length N
    probabilities=list of length N
    size=integer
    """
    bins = np.add.accumulate(probabilities)
    return values[np.digitize(random_sample(size), bins)]


def target2int(target):
    """converts list of words in target array into list of numbers.
    0='positive', 1= 'neutral', 2= 'negative'
    """
    classes = list(set(target))
    print [[i, classes[i]] for i in range(len(classes))]
    new = []
    for i in target:
        for j in range(len(classes)):
            if i == classes[j]:
                new.append(j)
    return new


def target2tri(target):
    """
    target is list with possible values 1-5, convert to list with POS-0=4,5, NEUT-1=3
    and NEG-2=1,2
    :param target: list
    :return:list
    """
    mapping = {5: 0, 4: 0, 3: 1, 2: 2, 1: 2}
    return map(lambda x: mapping[x], target)


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
    raw = codecs.open(fname, 'r', 'utf-8').readlines()  # load and split data into reviews
    random.shuffle(raw)
    target = [int(float((r.split('\t')[5]))) for r in raw]  # target is pos,neg,neutral
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


def bow_clf(tr_data, tr_target, te_data, te_target):
    '''read in train and test file and perform classification experiment with sklearn bow features and clf'''

    # tr_data, tr_target = load_amazon(trainfile)
    # te_data, te_target = load_amazon(testfile)
    vec = tf(ngram_range=(1, 4), max_df=0.9, min_df=0.01)  # basic tfidf vectorizer
    vec.fit(tr_data)
    print 'TFIDF FIT'
    X_train = vec.transform(tr_data)
    X_test = vec.transform(te_data)
    print 'TRANSFORMED'
    clf = svm(class_weight='auto')
    clf.fit(X_train, tr_target)  # fit classifier to training data
    # x=clf.predict(X_test)
    # with open('twitter.train_predict','w') as f:
    # for i in x:
    #         f.write(str(i))
    #         f.write('\n')
    print 'data:\ntrain size: %s test size: %s' % (str(len(tr_target)), str(len(te_target)))
    print 'train set class representation:' + str(Counter(tr_target))
    print 'test set class representation: ' + str(Counter(te_target))
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


def vectorize_gensim(data, modelname):
    """
    :param data: data is a list of strings, each string is a document
    :return: a sparse array of vectors representing word embeddings for words in word2vec model according to the modelname
    """
    model = gensim.models.Word2Vec.load(modelname)
    new_data = []
    # oovdic=defaultdict(int)
    blank = np.ndarray(model.layer1_size)  # TODO check this makes sense for comparisons
    for i in data:  #for each document
        new_vec = []
        for j in tokenize(i):  #for each word in the document
            try:
                new_vec.append(model[j])  #add the model representation of the word/ the embedding
            except:
                new_vec.append(blank)  # to ensure document vectors are same size
                # oovdic[j]+=1
        new_data.append(new_vec)
    # print oovdic
    return new_data


def avg_rating_amherst(text, word_dict):
    '''
    calculates the average sentiment score of a text as the average over ratings of unigrams in text based on
    amherst rating table.
    :param fname: textfile name to be scored
    :param word_dict: Words().dic
    :return: the average rating over words in text
    '''
    score = []
    oov = []
    iv = []
    l = 0
    for i in tokenize(text):
        l += 1
        try:
            score.append(word_dict[i])
            iv.append(i)
        except:
            oov.append(i)
            #todo put in a counter here to see how many words are oov
    # print 'in:',iv
    # print 'out:',oov
    # print '% oov:',round((1.0*len(oov)/l)*100,0)
    return np.mean(score)


def amherst_average_score(word_dict, outf):
    '''
    write to file
    '''
    with open(outf, 'w') as f:
        for i in te_data:
            avg = avg_rating_amherst(i, word_dict)
            avg = int(round(avg, 0))  #round the average to the nearest integer so it can be mapped to class labels
            print target2tri([avg])[
                0]  #map to tri class labels (make a list so it works with code above, then back to int
            f.write(str(target2tri([avg])[0]))
            f.write('\n')


def get_senti_vocab(senti_word_dict):
    vocab = []  # holds ngrams that are features
    for i in senti_word_dict:
        vocab.append(' '.join(i.split('_')))  # split words that are 2,3..grams


if __name__ == "__main__":
    # BOW baseline
    tr_data, tr_target = load_twitter('Data/twitter/twitter.train')
    # tr_data, tr_target = load_amazon('Data/amazon/review10.train')
    te_data, te_target = load_twitter('Data/twitter/twitter.test')
    # te_data, te_target = load_amazon('Data/amazon/review.train')


    # convert AMAZON data to 3
    # tr_target = target2tri(tr_target)
    # te_target=target2tri(te_target)
    # convert TWITTER data to int
    # tr_target=target2int(tr_target)
    # te_target = target2int(te_target)
    # bow_clf(tr_data, tr_target, te_data, te_target)

    #embedding stuff
    #review-uni_10.model is trained from unigrams, 10 million lines of words generated from amherst data

    # X_train=vectorize_gensim(tr_data,'Data/models/review-uni_10.model')
    # X_test=vectorize_gensim(te_data,'Data/models/review-uni_10.model')
    # np.save('temp',X_test[0])
    # print len(X_test), len(X_train)
    # print len(X_test[0]), len(X_train[0])
    # clf = log()
    # clf.fit(X_train, tr_target)  #fit classifier to training data
    # print '\nclassifier\n----'
    # print 'Accuracy on train:', clf.score(X_train, tr_target)
    # print 'Accuracy on test:', clf.score(X_test, te_target)
    # print '\nReport\n', classification_report(te_target, clf.predict(X_test))

    W = Words.Words()
    W.build_dict_amherst()  #creates the rating dictionary
    W.build_dict_sentiword()
    print W.amherst_dict
    print W.sentiword_dict
    #TODO some mapping of sentiworddict values to pos neut neg etc

    # amherst_average_score(W.amherst_dict,'Data/twitter/amherst.test.pred')
