from __future__ import print_function

from keras import activations, initializers
from keras import regularizers
from keras.engine import Layer
from keras.layers import Dropout

import keras.backend as K


class GraphConvolution(Layer):
    def __init__(self, output_dim, support=1, featureless=False,
                 kernel_initializer='glorot_uniform', activation='linear',
                 weights=None, kernel_regularizer=None, num_bases=-1,
                 bias_regularizer=None, use_bias=False, dropout=0.,input_dim=None,
                 **kwargs):
        self.kernel_initializer = kernel_initializer
        self.activation = activations.get(activation)
        self.output_dim = output_dim  # number of features per node
        self.support = support  # filter support / number of weights
        self.featureless = featureless  # use/ignore input features
        self.dropout = dropout

        assert support >= 1

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.use_bias = use_bias
        self.initial_weights = weights
        self.num_bases = num_bases

        # these will be defined during build()
        self.input_dim = None
        self.kernel = None
        self.kernel_comp = None
        self.bias = None
        self.num_nodes = None

        super(GraphConvolution, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.output_dim)
        return output_shape  # (batch_size, output_dim)

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        if self.featureless:
            self.num_nodes = features_shape[1]  # NOTE: Assumes featureless input (i.e. square identity mx)
        assert len(features_shape) == 2
        self.input_dim = features_shape[1]

        if self.num_bases > 0:
            self.kernel = self.add_weight(shape=(self.input_dim*self.num_bases, self.output_dim),
                                                    initializer=self.kernel_initializer,
                                                    name='{}_kernel'.format(self.name),
                                                    regularizer=self.kernel_regularizer)

            self.kernel_comp = self.add_weight(shape=(self.support, self.num_bases),
                                          initializer=self.kernel_initializer,
                                          name='{}_kernel_comp'.format(self.name),
                                          regularizer=self.kernel_regularizer)
        else:
            self.kernel = self.add_weight(shape=(self.input_dim*self.support, self.output_dim),
                                                    initializer=self.kernel_initializer,
                                                    name='{}_kernel'.format(self.name),
                                                    regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                     initializer='zero',
                                     name='{}_bias'.format(self.name),
                                     regularizer=self.bias_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, mask=None):
        features = inputs[0]
        A = inputs[1:]  # list of basis functions

        # convolve
        supports = list()
        for i in range(self.support):
            if not self.featureless:
                supports.append(K.dot(A[i], features))
            else:
                supports.append(A[i])
        supports = K.concatenate(supports, axis=1)

        if self.num_bases > 0:
            self.kernel = K.reshape(self.kernel,
                               (self.num_bases, self.input_dim, self.output_dim))
            self.kernel = K.permute_dimensions(self.kernel, (1, 0, 2))
            V = K.dot(self.kernel_comp, self.kernel)
            V = K.reshape(V, (self.support*self.input_dim, self.output_dim))
            output = K.dot(supports, V)
        else:
            output = K.dot(supports, self.kernel)

        # if featureless add dropout to output, by elementwise multiplying with column vector of ones,
        # with dropout applied to the vector of ones.
        if self.featureless:
            tmp = K.ones(self.num_nodes)
            tmp_do = Dropout(self.dropout)(tmp)
            output = K.transpose(K.transpose(output) * tmp_do)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'kernel_initializer': self.kernel_initializer,
                  'activation': self.activation.__name__,
                  'kernel_regularizer': self.kernel_regularizer.get_config() if self.kernel_regularizer else None,
                  'bias_regularizer': self.bias_regularizer.get_config() if self.bias_regularizer else None,
                  'num_bases': self.num_bases,
                  'use_bias': self.use_bias,
                  'input_dim': self.input_dim}
        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
