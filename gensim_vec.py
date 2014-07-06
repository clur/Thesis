# coding: utf-8
__author__ = 'claire'

import gensim
import codecs
import time

fname = '2mil.txt'


class MySentences(object):
    def __iter__(self):
        for line in codecs.open(fname, 'r', 'utf-8'):
            # assume there's one document per line, tokens separated by whitespace
            yield line.lower().split()


start = time.time()

print 'training model on ' + fname
sentences = MySentences()
model = gensim.models.Word2Vec(sentences, min_count=1, size=100, workers=4)
print 'trained model, took %1.4f minutes' % ((time.time() - start) / 60)
model.save('10mil.model')
print 'saved model, took %1.4f minutes' % ((time.time() - start) / 60)
print time.time() - start

# model=gensim.models.Word2Vec.load('review-uni_10.model')

from utility import load_amazon

tr_data, tr_target = load_amazon('Data/amazon/review.train')
te_data, te_target = load_amazon('Data/amazon/review.test')

