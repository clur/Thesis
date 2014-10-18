__author__ = 'claire'
from collections import defaultdict
import operator
import codecs
import re
import numpy as np
import cPickle


class MyTokens(object):
    def __init__(self, fname):
        self.file = fname
        self.token = re.compile(r"(?u)\b\w+\b\'*\w*")

    def __iter__(self):
        for line in codecs.open(self.file, 'r'):
            yield self.token.findall(line.lower())


def get_vocab(file, v=20000):
    """
    extract vocabulary from file capped at v most frequent words
    :returns a dictionary of word:index pairs where index of
    most frequent word is 0, 2nd most frequent is 1, and so on
    """
    s = MyTokens(file)
    d = defaultdict(int)
    for tokens in s:
        for t in tokens:
            d[t] += 1
    top_sorted = sorted(d.items(), key=operator.itemgetter(1), reverse=True)[:v]
    print top_sorted
    vocab = defaultdict()
    id = 0
    for i in top_sorted:
        vocab[i[0]] = id
        id += 1
    vocab['UNK'] = len(vocab) + 1
    vocab['zpaddingz'] = len(vocab) + 2
    return vocab


def mapping(list):
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

    file = '/Users/claire/Dropbox/PycharmProjects/Thesis/Scripts/Data/CW_data/wikipedia/padded_full.txt'
    # vocab=get_vocab(file, v=20000)

    # use list already extracted to speed things up
    x = open('top words.txt').readlines()
    idx = 0
    vocab = defaultdict()
    for i in x:
        vocab[i.split(':')[0]] = idx
        idx += 1
    vocab['UNK'] = len(vocab)
    vocab['zpaddingz'] = len(vocab)
    with open('vocab.pickle','wb') as handle:
        cPickle.dump(vocab,handle)
    text = MyTokens(file)  #text is list of list of tokens
    Y = []
    X = []
    window = 2
    N = 200000  # sentences to consider

    s = 0
    for sentence in text:
        if s%10000==0:
            print '%s sentences processed' % s
        if s >= N:
            break
        s += 1
        if len(sentence) > window * 2 + 4:  # must have at least 2*window+1 length
            for idx in range(window, len(sentence) - window):
                Y.append(mapping(sentence[idx]))
                X.append(mapping(sentence[idx - window:idx] + sentence[idx + 1:idx + 1 + window]))

    with open('X.pickle', 'wb') as handle:
        cPickle.dump(np.array(X),handle)
    with open('Y.pickle', 'wb') as handle:
        cPickle.dump(np.array(Y),handle)

    #perform sanity check
    # for i in range(len(Y)):
    #     inv_vocab = {v: k for k, v in vocab.items()}
    #     sanity_check(X[i], Y[i][0], inv_vocab)

