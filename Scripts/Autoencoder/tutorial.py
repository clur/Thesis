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
import matplotlib.pyplot as plt


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
  Transform an array with one flattened image per row, into an array in
  which images are reshaped and layed out like tiles on a floor.

  This function is useful for visualizing datasets whose rows are images,
  and also columns of matrices for transforming those rows
  (such as the first layer of a neural net).

  :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
  be 2-D ndarrays or None;
  :param X: a 2-D array in which every row is a flattened image.

  :type img_shape: tuple; (height, width)
  :param img_shape: the original shape of each image

  :type tile_shape: tuple; (rows, cols)
  :param tile_shape: the number of images to tile (rows, cols)

  :param output_pixel_vals: if output should be pixel values (i.e. int8
  values) or floats

  :param scale_rows_to_unit_interval: if the values need to be scaled before
  being plotted to [0,1] or not


  :returns: array suitable for viewing as an image.
  (See:`Image.fromarray`.)
  :rtype: a 2-d array with same dtype as X.

  """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape = [0,0]
    # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
    # tile_spacing[0]
    # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
    # tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                 in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        # colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                out_array[:, :, i] = np.zeros(out_shape,
                                              dtype='uint8' if output_pixel_vals else out_array.dtype) + \
                                     channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing,
                                                        scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        out_array = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                    else:
                        this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    out_array[
                    tile_row * (H + Hs): tile_row * (H + Hs) + H,
                    tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] \
                        = this_img * (255 if output_pixel_vals else 1)
        return out_array


class dA(object):
    def __init__(self, numpy_rng, theano_rng=None, input=None, n_visible=1, n_hidden=1, W=None, bhid=None,
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
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30), name='theano_rng')

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
        # L = 0.5*((self.x - z) ** 2)
        cost = T.mean(L)
        gparams = T.grad(cost, self.params)
        updates = [(param, param - learning_rate * gparam) for param, gparam in zip(self.params, gparams)]
        return (cost, updates)


# load data
# datasets = load_data('mnist.pkl.gz')


# resize images
import glob


def preprocess_images(fldr, w, h):
    ims = glob.glob(fldr + '/*.jpg')
    print ims
    ims = [Image.open(i) for i in ims]
    ims = [im.resize((w, h)) for im in ims]
    ims = [np.array(i) for i in ims]
    ims = [i[:, :, 0] for i in ims if len(i.shape) > 2]
    print '%d images in %s folder' % (len(ims), fldr)
    plt.imshow(ims[3], cmap=plt.gray())
    # plt.show()
    ims = [x.flatten() for x in ims]
    return ims


fo = 'brain'
# original shape is 800*600
w, h = 100, 100
hands = preprocess_images(fo, w, h)
# pizzas=preprocess_images('pizza')
# joint = hands+pizzas
# print 'total', len(joint)
# print 'visible:',pizzas[0].shape
hands = np.asarray(hands)
print 'type(hands)', type(hands)
print 'hands.shape', hands.shape
train_set_x = theano.shared(np.asarray(hands, dtype=theano.config.floatX), borrow=True)
print type(train_set_x)
batch_size = 1
n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
index = T.lscalar(name='index')  # minibatch index
x = T.matrix('x')  # data

rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

da = dA(numpy_rng=rng, theano_rng=theano_rng, input=x, n_visible=w * h, n_hidden=1000)
learning_rate = 0.1
corruption = 0.3

cost, updates = da.get_cost_updates(corruption_level=corruption, learning_rate=learning_rate)

train_da = theano.function([index], cost, updates=updates,
                           givens={x: train_set_x[index * batch_size: (index + 1) * batch_size]})

theano.printing.pydotprint(train_da, outfile=fo + '_graph_train_da', var_with_name_simple=True, )
start_time = time.clock()

# train
training_epochs = 100

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
image = Image.fromarray(
    tile_raster_images(X=da.W.get_value(borrow=True).T,
                       img_shape=(w, h), tile_shape=(10, 10),
                       tile_spacing=(1, 1)))
plt.imshow(image, cmap=plt.gray())
plt.show()
image.save(
    'hands_filters_corr' + str(corruption) + '_batchsize' + str(batch_size) + '_epochs' + str(training_epochs) + '.png')

print 'Took %f minutes' % ((end_time - start_time) / 60.)
