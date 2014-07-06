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
    # BOW baseline
    # tr_data, tr_target = load_twitter('Data/twitter/twitter.test')
    tr_data, tr_target = load_amazon('Data/amazon/review10.train')
    te_data, te_target = load_twitter('Data/twitter/twitter.train')
    # te_data, te_target = load_amazon('Data/amazon/review.train')


    # convert AMAZON data to 3
    tr_target = target2tri(tr_target)
    # te_target=target2tri(te_target)
    # convert TWITTER data to int
    # tr_target=target2int(tr_target)
    te_target = target2int(te_target)
    bow_clf(tr_data, tr_target, te_data, te_target)





    #embedding stuff

    # X_train=vectorize(tr_data,'review-uni_10.model')
    # X_test=vectorize(te_data,'review-uni_10.model')
    # np.save('temp',X_test[0])
    # clf = log()
    # clf.fit(X_train, tr_target)  #fit classifier to training data
    # print '\nclassifier\n----'
    # print 'Accuracy on train:', clf.score(X_train, tr_target)
    # print 'Accuracy on test:', clf.score(X_test, te_target)
    # print '\nReport\n', classification_report(te_target, clf.predict(X_test))
    # W = Words.Words()
    # W.build_dicts()
    # for i in W.rating_dict.keys():
    # print len(W.rating_dict[i])

    # inv_rating = {v:k for k, v in W.rating_dict.items()}

    # avg_rating(tr_data[0],inv_rating) #todo invert list so keys are words and ratings are values

    # print tr_data[0]
    # pass
