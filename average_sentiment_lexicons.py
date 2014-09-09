__author__ = 'claire'
import codecs
# from numpy.random import random_sample
import numpy as np
from sklearn.metrics import classification_report, f1_score
import re
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import Words


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


def load_twitter_2class(fname):
    """
    open and read fname, split into data and target
    :param fname: filename to read
    :return: data, list and target, list
    """
    raw = codecs.open(fname, 'r', 'utf-8').readlines()  # load and split data into reviews
    raw = [r for r in raw if r.split('\t')[2] != u'neutral']
    target = [r.split('\t')[2] for r in raw]  # target is pos,neg,neutral
    data = [r.split('\t')[3] for r in raw]  # review text
    data = [d.lower().strip() for d in data]
    target = [t for t in target if t != 0]
    return data, target


def tokenize(text):
    """
    return list of tokens from string text based on token pattern
    :param text: string
    :return:list
    """
    token_pattern = r"(?u)\b\w\w+\b\'*\w*"  # from sklearn.text.py
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall(text)


def senti_score_2class(text, word_dict):
    """
    as below using senti word dict values
    :param text: senti pos tagged text (first stanford, then use postag replace
    :param word_dict: sentiworddict, can be loaded with W.load_sentiword('sentiword_dict.json')
    :return: ? 0 if more postive, 1 if more negative, 2 if a tie an average over the 3 scores from sentiworddict (pos,neg,neutral)
    """
    scores = []
    # print 'type', type(text), text
    for i in tokenize(text.lower()):
        try:
            k = tuple(i.split('_'))
            val = word_dict[k]
            val = val[:2]  # ignore the objective score
            scores.append(val)
            # print i,val
        except:
            pass

    avg = np.mean(scores, axis=0)
    return np.argmax(avg)


def senti_score(text, word_dict):
    """
    as below using senti word dict values
    :param text: senti pos tagged text (first stanford, then use postag replace
    :param word_dict: sentiworddict, can be loaded with W.load_sentiword('sentiword_dict.json')
    :return: ? 0 if more postive, 1 if more negative, 2 if a tie an average over the 3 scores from sentiworddict (pos,neg,neutral)
    """
    scores = []
    # print 'type', type(text), text
    for i in tokenize(text.lower()):
        try:
            k = tuple(i.split('_'))
            val = word_dict[k]
            val = val[:2]  # ignore the objective score
            scores.append(val)
        except:
            pass
    if len(scores) > 0:
        avg = np.mean(scores, axis=0)
        if avg[0] == avg[1] or avg[np.argmax(avg)] < 0.1:  # lower than a threshold gets marked as neutral
            result = 2
        else:
            result = np.argmax(avg)
    else:
        result = 2  # all words are oov, assign to neutral
    return result


def senti_avg_2class(test_data, word_dict):
    """
    """
    pred = []
    for i in test_data:
        avg = senti_score_2class(i, word_dict)
        map = {0: 1, 1: -1, 2: 0}  # tuple is ordered(pos,neg,neutral)
        avg = map[avg]
        # print 'mapped:',avg
        # print
        pred.append(avg)
    return pred


def senti_avg(test_data, word_dict):
    """
    """
    pred = []
    for i in test_data:
        # print i,':'
        avg = senti_score(i, word_dict)
        map = {0: 1, 1: -1, 2: 0}  # tuple is ordered(pos,neg,neutral)
        avg = map[avg]
        # print 'mapped:',avg
        # print
        pred.append(avg)
    return pred


def amherst_score(text, word_dict):
    """
    calculates the average sentiment score of a text as the average over ratings of unigrams in text based on
    amherst rating table.
    :param fname: textfile name to be scored
    :param word_dict: Words().dic
    :return: the average rating over words in text
    """
    score = []

    for i in tokenize(text.lower()):
        try:
            score.append(word_dict[i])
            # print i,':',word_dict[i]
        except:
            pass

    # print score
    # print np.mean(score)
    # print text
    # print
    return np.mean(score)


def amherst_avg(test_data, word_dict):
    """
    binary case
    """
    pred = []
    for i in test_data:
        # print i
        avg = amherst_score(i, word_dict)
        avg = int(round(avg, 0))  # round the average to the nearest integer so it can be mapped to class labels
        map = {1: -1, 2: -1, 3: 0, 4: 1, 5: 1}
        avg = map[avg]  # map it to class values
        pred.append(avg)
    return pred


def amherst_avg_2class(test_data, word_dict):
    """
    binary case
    """
    pred = []
    for i in test_data:
        # print i
        avg = amherst_score(i, word_dict)
        avg = int(round(avg, 0))  # round the average to the nearest integer so it can be mapped to class labels
        if avg > 2.99999:  # split neutral in the middle of 2.49 - 3.49
            avg = 4
        else:
            avg = 2
        map = {1: -1, 2: -1, 4: 1, 5: 1}
        avg = map[avg]  # map it to class values
        pred.append(avg)
    return pred


def scorer(true, pred):
    """
    use classification report method on files, format is -1,0,1
    :return:
    """
    print classification_report(true, pred)
    print 'Macro F1', f1_score(true, pred, average='micro')


def postag_replace(file):
    """
    read file into memory, replace stanford tags with senti tags, write to file _senti_tagged
    :param file: a Stanford pos tagged file
    :return: None
    """
    # nouns
    text = codecs.open(file, 'r', 'utf8').read()
    text = re.sub('_NN[A-Z]*', '_n', text)
    #adverbs
    text = re.sub('_RB[R|S]*', '_r', text)
    #verbs
    text = re.sub('_VB[D|G|N|P|Z]*', '_v', text)
    #adjectives
    text = re.sub('_JJ[R|S]*', '_a', text)
    #punctuation junk
    text = re.sub('_[^A-Za-z]', '', text)
    #other tags
    text = re.sub('_[A-Z]+', '', text)
    with codecs.open(file + '_senti', 'w', 'utf8') as f:
        f.write(text)


if __name__ == "__main__":
    # postag_replace('Data/twitter/twitter.binary.test_POS-tagged')
    # TODO write postag part for senti in here for all in one-ness


    #Binary
    print 'amherst'
    f = 'Data/twitter/twitter.binary.test'
    te_data, te_target = load_twitter_2class(f)
    mapping = {u'positive': 1, u'neutral': 0, u'negative': -1}
    te_target = [mapping[t] for t in te_target]
    W = Words.Words()
    W.load_amherst('amherst_dict_5.json')
    y_hat = amherst_avg_2class(te_data, W.amherst_dict)
    print Counter(y_hat), Counter(te_target)
    print classification_report(te_target, y_hat)
    print 'Macro F1', f1_score(te_target, y_hat, pos_label=None, average='macro')
    print '-' * 10
    print 'senti'
    W.load_sentiword('sentiword_dict.json')

    te_data = codecs.open(f + '_POS-tagged_senti').readlines()
    y_hat = senti_avg_2class(te_data, W.sentiword_dict)
    print classification_report(te_target, y_hat)
    print 'Macro F1', f1_score(te_target, y_hat, pos_label=None, average='macro')

    print Counter(y_hat)
    print '-' * 10
    print '-' * 10


    #3class
    print 'amherst'
    f = 'Data/twitter/twitter.test'
    te_data, te_target = load_twitter(f)
    mapping = {u'positive': 1, u'neutral': 0, u'negative': -1}
    te_target = [mapping[t] for t in te_target]
    W = Words.Words()
    W.load_amherst('amherst_dict_5.json')
    y_hat = amherst_avg(te_data, W.amherst_dict)
    print Counter(y_hat), Counter(te_target)
    print classification_report(te_target, y_hat)
    print 'Macro F1', f1_score(te_target, y_hat, pos_label=None, average='macro')

    print '-' * 10
    print 'senti'
    W.load_sentiword('sentiword_dict.json')

    te_data = codecs.open(f + '_POS-tagged_senti').readlines()
    y_hat = senti_avg(te_data, W.sentiword_dict)
    print classification_report(te_target, y_hat)
    print 'Macro F1', f1_score(te_target, y_hat, pos_label=None, average='macro')
    print Counter(y_hat)
    print '-' * 10
    print '-' * 10