__author__ = 'claire'
from collections import defaultdict
import operator
import codecs
import re
import numpy as np
import cPickle
from collections import OrderedDict
from operator import itemgetter
import os
from scipy import stats


class MyTokens(object):
    def __init__(self, fname):
        self.file = fname
        tokens = r"(?<![http:\/\/t.co/\w+]|@)[\w'#]+\b"
        self.token = re.compile(tokens)

    def __iter__(self):
        for line in codecs.open(self.file, 'r'):
            yield self.token.findall(line.lower())


def write_vocab(file, v):
    """
    extract vocabulary from file capped at v most frequent words
    :returns a dictionary of word:index pairs where index of
    most frequent word is 0, 2nd most frequent is 1, and so on
    """
    total = 0
    s = MyTokens(file)
    words = defaultdict(int)
    for tokens in s:
        for t in tokens:
            words[t] += 1
            total += 1
    print 'Total', total
    top_sorted = sorted(words.items(), key=operator.itemgetter(1), reverse=True)
    outf = 'top_' + str(v) + '.txt'
    with open(outf, 'w') as f:
        print 'writing to:' + os.path.abspath(f.name)
        for i in top_sorted:
            f.write(i[0] + '\t' + str(i[1]))
            f.write('\n')


def mapping(list, vocab):
    res = []
    for i in list:
        if i in vocab:
            res.append(vocab[i])
        else:
            res.append(vocab['UNK'])
    return res


def sanity_check(x, y, inv_vocab):
    """
    return the string formed by the context x and term y
    :param x: context, 4 element long list
    :param y: term, string
    :return: string of above words from vocabulary
    """
    words = []
    for i in x:
        words.append(inv_vocab[i])
    words.insert(2, inv_vocab[y])
    print ' '.join(words)


if __name__ == "__main__":

    file = '/Users/claire/Dropbox/PycharmProjects/Thesis/Scripts/Data/CW_data/wikipedia/full.txt'
    # file = '/Users/claire/Dropbox/PycharmProjects/Thesis/Scripts/Data/CW_data/wikipedia/5000.txt'
    top = 30000
    # write_vocab(file, top)
    # use list already extracted to speed things up
    text = open('all_vocab_no_single_occ.txt').readlines()
    idx = 0
    vocab = defaultdict()
    for i in range(top):
        vocab[text[i].split('\t')[0]] = idx
        idx += 1
    vocab['UNK'] = len(vocab)
    vocab['BLANK'] = len(vocab)
    # do unigram dist for noise

    Y = []
    X = []
    window = 5
    N = 200000  # sentences to consider
    padding = ['BLANK']
    text = MyTokens(file)  # text is an iterator over a list of list of tokens
    s = 0

    for sentence in text:
        if s % 10000 == 0:
            print 'processed %d' % s
        if s >= N:
            break
        s += 1
        L = len(sentence)
        if L > 2:
            sentence = padding * window + sentence + padding * window
            for idx in range(window, len(sentence) - window):
                Y.append(mapping([sentence[idx]], vocab))
                X.append(mapping(sentence[idx - window:idx] + sentence[idx + 1:idx + 1 + window], vocab))

    print np.array(X).shape
    print np.array(Y).shape

    # make a directory to put things to avoid confusion
    #dir contains x, y, vocab and inv_vocab
    import os
    import datetime

    dirname = 'Preprocessed_window_' + str(window)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    os.chdir(dirname)
    with open('X.pickle', 'wb') as handle:
        cPickle.dump(np.array(X), handle, protocol=-1)
    with open('Y.pickle', 'wb') as handle:
        cPickle.dump(np.array(Y), handle, protocol=-1)
    with open('vocab' + '.pickle', 'wb') as handle:
        cPickle.dump(vocab, handle)
    inv_vocab = {v: k for k, v in vocab.items()}
    with open('inv_vocab' + '.pickle', 'wb') as handle:
        cPickle.dump(inv_vocab, handle)
    # with open('unigram' + '.pickle', 'wb') as handle:
    # cPickle.dump(unigram, handle)
    with open('Readme.txt', 'w') as handle:
        handle.write('Created: ' + str(datetime.datetime.now()))
        handle.write('\nWindow: ' + str(window))
        handle.write('\nSamples: ' + "{:,}".format(len(X)))
        handle.write('\nVocab size: ' + str(len(vocab)))