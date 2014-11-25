import datetime

__author__ = 'claire'
# http://deeplearning.net/tutorial/dA.html#autoencoders

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import time
import gzip
import cPickle
import os
import PIL.Image as Image
import sys
import codecs
from sklearn.feature_extraction.text import CountVectorizer as cv
from sklearn.feature_extraction.text import TfidfVectorizer as tf
import random
import theano.sparse as SP


# (?<!@)\b\w[\w|']+\b

class dA(object):
    def __init__(self, learning_rate, corruption, numpy_rng, n_visible, n_hidden, theano_rng=None, input=None, W=None,
                 bhid=None,
                 bvis=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        if not W:
            initial_W = np.asarray(numpy_rng.uniform(low=-1 * np.sqrt(6. / (n_hidden + n_visible)),
                                                     high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                                                     size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)
        if not bvis:
            bvis = theano.shared(value=np.zeros(n_visible, dtype=theano.config.floatX), name='bprime', borrow=True)
        if not bhid:
            bhid = theano.shared(value=np.zeros(n_hidden, dtype=theano.config.floatX), name='b', borrow=True)
        self.W = W
        self.b = bhid  # bias of hidden
        self.b_prime = bvis  # bias of visible
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        self.x = input
        self.params = [self.W, self.b, self.b_prime]
        self.learning_rate = learning_rate
        self.corruption = corruption
        self.cost = self.get_cost()
        self.update = self.get_update()

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(SP.dot(input, self.W) + self.b)

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1, p=1 - corruption_level) * input

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost(self):
        tilde_x = self.get_corrupted_input(self.x, self.corruption)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        L = - T.sum(SP.dense_from_sparse(self.x) * T.log(z) + (1 - SP.dense_from_sparse(self.x)) * T.log(1 - z), axis=1)
        # L = T.sum((SP.dense_from_sparse(self.x) - z) **2, axis = 1)
        cost = T.mean(L)
        # print 'learningrate:', self.learning_rate
        return cost

    def get_update(self):
        gparams = T.grad(self.cost, self.params)
        updates = [(param, param - self.learning_rate * gparam) for param, gparam in zip(self.params, gparams)]
        return updates


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
    mapping = {u'positive': 1, u'negative': - 1}
    target = [mapping[t] for t in target]
    return data, target


# for dumping
def dump_params():
    cPickle.dump(da.W.get_value(borrow=True),
                 open('cv_W_corr' + str(corruption) + '_batchsize' + str(batch_size) + '_epochs' + str(training_epochs),
                      'wb'))
    cPickle.dump(da.b.get_value(borrow=True),
                 open('cv_b_corr' + str(corruption) + '_batchsize' + str(batch_size) + '_epochs' + str(training_epochs),
                      'wb'))
    cPickle.dump(da.b_prime.get_value(borrow=True), open(
        'cv_bprime_corr' + str(corruption) + '_batchsize' + str(batch_size) + '_epochs' + str(training_epochs), 'wb'))


# For Debugging
def inspect_inputs(i, node, fn):
    print i, node, "input(s) value(s):\n", [i[0] for i in fn.inputs],


def inspect_outputs(i, node, fn):
    print "output(s) value(s):\n", [output[0] for output in fn.outputs]

# mode = theano.compile.MonitorMode(post_func=detect_nan).excluding('local_elemwise_fusion', 'inplace)

# mode = theano.compile.MonitorMode(pre_func=inspect_inputs, post_func=inspect_outputs).excluding('local_elemwise_fusion', 'inplace')

# load data
train_f = '/Users/claire/Dropbox/PycharmProjects/Thesis/Scripts/Data/twitter_CST/englishtweets.both'
# train_f = 'Data/twitter/twitter.train'
# train, _ = load_twitter_2class(train_f)
train = open(train_f).readlines()
train = [t.split('\t')[1:] for t in train]
train = [t for inner in train for t in inner]
random.shuffle(train)
tokens = r"(?<![http:\/\/t.co/\w+]|@)[\w'#]+\b"
# tokens = r" (?<![http:\/\/t.co/\w+]|@)\b\w{2,20}[\w']\b"
# tokens = r"(?<!@)\b\w[\'\w\-]+"

vec = cv(token_pattern=tokens)
# vec = cv(token_pattern=tokens, min_df=0.01)
train_set = vec.fit_transform(train)
print type(train_set)
# vec = tf(token_pattern=tokens, min_df=0.01)
# train_set = vec.fit_transform(train)
# print type(train_set)

vocab = vec.vocabulary_
now = str(datetime.datetime.now())
cPickle.dump(vec, open('AE_unlabeled_vec_cv' + str(now), 'w'))

#variables for dA object
n_vis = train_set.shape[1]
n_hid = 500
train_set_x = SP.shared(train_set)
batch_size = 20
n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
index = T.lscalar()  # minibatch index
x = SP.csr_matrix(name='xdata', dtype='int64')  # data
rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))
# lr = theano.shared(learning_rate)
# learning_rate = 0.1  # 1e-3
learning_rate = theano.shared(0.1, 'lr')
corruption = 0.3

print
print 'train size:', len(train)
print 'train_batches:', n_train_batches
print 'batch_size', batch_size
print 'n_visible:', n_vis
print 'n_hidden:', n_hid
print 'vocabulary:', len(vocab)
print 'initial learning rate:', learning_rate.get_value()
print 'corruption level:', corruption
print '----------------------'

da = dA(learning_rate, corruption, rng, n_vis, n_hid, theano_rng=theano_rng, input=x)
# cost, updates = da.get_cost_updates(corruption_level=corruption, learning_rate=learning_rate)

train_da = theano.function(inputs=[index], outputs=[da.cost], updates=da.update,
                           givens={x: train_set_x[index * batch_size: (index + 1) * batch_size]})  #, mode = mode)

# tempfunc = theano.function(inputs=[index],outputs=[da.x.shape, da.x], givens={x: train_set_x[index * batch_size: (index + 1) * batch_size]})

start_time = time.clock()

# train epochs
training_epochs = 100
print 'Starting training at %s' % datetime.datetime.now()
for epoch in xrange(training_epochs):
    print 'epoch %d (of %d)' % (epoch, training_epochs)
    c = []
    for batch_index in xrange(n_train_batches):
        # print 'lr', da.learning_rate.get_value()
        cost = train_da(batch_index)[0]
        c.append(cost)
        if batch_index % 100 == 0:
            print '(%d%% done)\tcost:%.4f' % (batch_index / float(n_train_batches) * 100, cost)
    da.learning_rate.set_value(da.learning_rate.get_value() * 0.95)
    print 'Training epoch %d, mean cost (so far)' % epoch, np.mean(c)
    print '%.2fm so far...' % ((time.clock() - start_time) / 60.)
    print 'dumping W, b and b_prime'
    dump_params()

#
# print da.y.eval()
end_time = time.clock()
training_time = end_time - start_time

print 'The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % (training_time / 60.)
