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
from utility import load_twitter_2class, load_amazon
from sklearn.feature_extraction.text import CountVectorizer as cv


class dA(object):
    def __init__(self, numpy_rng, n_visible, n_hidden, theano_rng=None, input=None, W=None, bhid=None,
                 bvis=None):

        """
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None
        """
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
        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]


    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1, p=1 - corruption_level) * input

    def get_cost_updates(self, corruption_level, learning_rate):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        cost = T.mean(L)
        gparams = T.grad(cost, self.params)
        updates = [(param, param - learning_rate * gparam) for param, gparam in zip(self.params, gparams)]
        return (cost, updates)


# load data
train_f = 'Data/twitter/twitter.train'
train, _ = load_twitter_2class(train_f)
vec = cv(binary=True, min_df=0.001)
train_set = vec.fit_transform(train)

n_vis = train_set.shape[1]
n_hid = 50
train_set = train_set.A  # will give you a dense array constructed from your sparse matrix.
train_set_x = theano.shared(np.asarray(train_set, dtype=theano.config.floatX), borrow=True)

batch_size = 10
n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

index = T.lscalar()  # minibatch index
x = T.matrix('x')  # data

rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

print 'train_batches:', n_train_batches
print 'n_visible:', n_vis
print 'n_hidden:', n_hid

da = dA(rng, n_vis, n_hid, theano_rng=theano_rng, input=x)
learning_rate = 1e-3
corruption = 0.5

cost, updates = da.get_cost_updates(corruption_level=corruption, learning_rate=learning_rate)

train_da = theano.function([index], cost, updates=updates,
                           givens={x: train_set_x[index * batch_size: (index + 1) * batch_size]})

start_time = time.clock()

# train
training_epochs = 20

for epoch in xrange(training_epochs):
    c = []
    for batch_index in xrange(n_train_batches):
        # print batch_index
        c.append(train_da(batch_index))
    print 'Training epoch %d, mean cost (so far)' % epoch, np.mean(c)

end_time = time.clock()
training_time = end_time - start_time

print >> sys.stderr, ('The code for file ' +
                      os.path.split(__file__)[1] +
                      ' ran for %.2fm' % ((training_time) / 60.))

cPickle.dump(da.W.get_value(borrow=True),
             open('W_2corr' + str(corruption) + '_batchsize' + str(batch_size) + '_epochs' + str(training_epochs),
                  'wb'))
