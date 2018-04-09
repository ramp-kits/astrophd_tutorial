from __future__ import division

from math import ceil

import numpy as np

from sklearn.utils import Bunch

from keras.models import Sequential
from keras.layers import Conv2D, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.layers.noise import GaussianNoise


class ObjectDetector(object):
    """Object detector.

    Parameters
    ----------
    batch_size : int, optional
        The batch size used during training. Set by default to 32 samples.

    epoch : int, optional
        The number of epoch for which the model will be trained. Set by default
        to 50 epochs.

    model_check_point : bool, optional
        Whether to create a callback for intermediate models.

    Attributes
    ----------
    model_ : object
        The DNN model.

    params_model_ : Bunch dictionary
        All hyper-parameters to build the DNN model.

    """

    def __init__(self, batch_size=32, epoch=10, model_check_point=True):
        self.model_, self.params_model_ = self._build_model()
        self.batch_size = batch_size
        self.epoch = epoch
        self.model_check_point = model_check_point

    def fit(self, X, y, pretrained=False):

        if pretrained:
            # for showcase load weights (this is not possible
            # for an actual submission)
            self.model_.load_weights(
                'submissions/keras_fcnn/fcnn_weights_best.h5')
            return

        # build the box encoder to later encode y to make usable in the model
        train_dataset = BatchGeneratorBuilder(X, y)
        train_generator, val_generator, n_train_samples, n_val_samples = \
            train_dataset.get_train_valid_generators(
                batch_size=self.batch_size)

        # create the callbacks to get during fitting
        callbacks = self._build_callbacks()

        # fit the model
        self.model_.fit_generator(
            generator=train_generator,
            steps_per_epoch=ceil(n_train_samples / self.batch_size),
            epochs=self.epoch,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=ceil(n_val_samples / self.batch_size))

    def predict(self, X):
        return self.model_.predict(np.expand_dims(X, -1))

    ###########################################################################
    # Setup model

    @staticmethod
    def _init_params_model():
        params_model = Bunch()

        # image and class parameters
        params_model.img_rows = 128
        params_model.img_cols = 128
        params_model.img_channels = 1

        # architecture params
        params_model.output_channels = 1            # size of the output in depth
        params_model.depth = 16                     # depth of all hidden layers
        params_model.n_layers = 6                   # number of layers before last
        params_model.conv_size0 = (3, 3)            # kernel size of first layer
        params_model.conv_size = (3, 3)             # kernel size of intermediate layers
        params_model.last_conv_size = (3, 3)        # kernel size of last layer
        params_model.activation = 'relu'            # activation between layers
        params_model.last_activation = 'sigmoid'    # final activation (sigmoid nice if binary objective)
        params_model.initialization = 'he_normal'   # weight initialization
        params_model.constraint = None              # kernel constraints (None, nonneg, unitnorm, maxnorm)
        params_model.dropout_rate = 0.0             # percentage of weights not updated (0 = no dropout)
        params_model.sigma_noise = 0.01             # random noise added before last layer (0 = no noise added)

        # optimizer parameters
        params_model.lr = 1e-4
        params_model.beta_1 = 0.9
        params_model.beta_2 = 0.999
        params_model.epsilon = 1e-08
        params_model.decay = 5e-05

        # loss parameters
        params_model.keras_loss = 'binary_crossentropy'

        # callbacks parameters
        params_model.early_stopping = True
        params_model.es_patience = 12
        params_model.es_min_delta = 0.001

        params_model.reduce_learning_rate = True
        params_model.lr_patience = 5
        params_model.lr_factor = 0.5
        params_model.lr_min_delta = 0.001
        params_model.lr_cooldown = 2

        return params_model

    def _build_model(self):

        # load the parameter for the SSD model
        params_model = self._init_params_model()

        model = fcnn_model(
            input_shape=(params_model.img_rows,
                         params_model.img_cols,
                         params_model.img_channels),
            output_channels=params_model.output_channels,
            depth=params_model.depth,
            n_layers=params_model.n_layers,
            conv_size0=params_model.conv_size0,
            conv_size=params_model.conv_size,
            last_conv_size=params_model.last_conv_size,
            activation=params_model.activation,
            last_activation=params_model.last_activation,
            dropout_rate=params_model.dropout_rate,
            sigma_noise=params_model.sigma_noise,
            initialization=params_model.initialization,
            constraint=params_model.constraint)

        optimizer = Adam(lr=params_model.lr)

        model.compile(optimizer=optimizer, loss=params_model.keras_loss)

        return model, params_model

    def _build_callbacks(self):
        callbacks = []

        if self.model_check_point:
            callbacks.append(
                ModelCheckpoint('./fcnn_weights_best.h5',
                                monitor='val_loss',
                                save_best_only=True,
                                save_weights_only=True,
                                period=1,
                                verbose=1))
        # add early stopping
        if self.params_model_.early_stopping:
            callbacks.append(
                EarlyStopping(monitor='val_loss',
                              min_delta=self.params_model_.es_min_delta,
                              patience=self.params_model_.es_patience,
                              verbose=1))

        # reduce learning-rate when reaching plateau
        if self.params_model_.reduce_learning_rate:
            callbacks.append(
                ReduceLROnPlateau(monitor='val_loss',
                                  factor=self.params_model_.lr_factor,
                                  patience=self.params_model_.lr_patience,
                                  cooldown=self.params_model_.lr_cooldown,
                                  # min_delta=self.params_model_.lr_min_delta,
                                  verbose=1))

        return callbacks


def fcnn_model(input_shape, output_channels, depth, n_layers,
               conv_size0, conv_size, last_conv_size,
               activation, last_activation,
               dropout_rate, sigma_noise,
               initialization, constraint):
    model = Sequential()
    model.add(Conv2D(depth, conv_size0,
                     input_shape=input_shape,
                     activation=activation,
                     padding='same',
                     name='conv0',
                     kernel_initializer=initialization,
                     kernel_constraint=constraint))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    for layer_n in range(1, n_layers):
        model.add(Conv2D(depth, conv_size,
                         activation=activation,
                         padding='same',
                         name="conv{}".format(layer_n),
                         kernel_initializer=initialization,
                         kernel_constraint=constraint))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    if sigma_noise > 0:
        model.add(GaussianNoise(sigma_noise))

    model.add(Conv2D(output_channels, last_conv_size,
                     activation=last_activation,
                     padding='same',
                     name='last',
                     kernel_initializer=initialization,
                     kernel_constraint=constraint))

    return model


###############################################################################
# Batch generator


class BatchGeneratorBuilder(object):
    """A batch generator builder for generating batches of images on the fly.

    This class is a way to build training and
    validation generators that yield each time a tuple (X, y) of mini-batches.
    The generators are built in a way to fit into keras API of `fit_generator`
    (see https://keras.io/models/model/).

    The fit function from `Classifier` should then use the instance
    to build train and validation generators, using the method
    `get_train_valid_generators`

    Parameters
    ==========

    X_array : ArrayContainer of int
        vector of image data to train on
    y_array : vector of int
        vector of object labels corresponding to `X_array`

    """
    def __init__(self, X_array, y_array):
        self.X_array = X_array
        self.y_array = y_array
        self.nb_examples = len(X_array)

    def get_train_valid_generators(self, batch_size=256, valid_ratio=0.1):
        """Build train and valid generators for keras.

        This method is used by the user defined `Classifier` to o build train
        and valid generators that will be used in keras `fit_generator`.

        Parameters
        ==========

        batch_size : int
            size of mini-batches
        valid_ratio : float between 0 and 1
            ratio of validation data

        Returns
        =======

        a 4-tuple (gen_train, gen_valid, nb_train, nb_valid) where:
            - gen_train is a generator function for training data
            - gen_valid is a generator function for valid data
            - nb_train is the number of training examples
            - nb_valid is the number of validation examples
        The number of training and validation data are necessary
        so that we can use the keras method `fit_generator`.
        """
        nb_valid = int(valid_ratio * self.nb_examples)
        nb_train = self.nb_examples - nb_valid
        indices = np.arange(self.nb_examples)
        train_indices = indices[0:nb_train]
        valid_indices = indices[nb_train:]
        gen_train = self._get_generator(
            indices=train_indices, batch_size=batch_size)
        gen_valid = self._get_generator(
            indices=valid_indices, batch_size=batch_size)
        return gen_train, gen_valid, nb_train, nb_valid

    def _get_generator(self, indices=None, batch_size=256):
        if indices is None:
            indices = np.arange(self.nb_examples)
        # Infinite loop, as required by keras `fit_generator`.
        # However, as we provide the number of examples per epoch
        # and the user specifies the total number of epochs, it will
        # be able to end.
        while True:
            X = self.X_array[indices]
            y = self.y_array[indices]

            # converting to float needed?
            X = np.array(X, dtype='float32')
            y = np.array(y, dtype='float32')

            # Yielding mini-batches
            for i in range(0, len(X), batch_size):

                X_batch = [np.expand_dims(img, -1)
                           for img in X[i:i + batch_size]]
                y_batch = [np.expand_dims(seg, -1)
                           for seg in y[i:i + batch_size]]

                for j in range(len(X_batch)):

                    # flip images
                    if np.random.randint(2):
                        X_batch[j] = np.flip(X_batch[j], axis=0)
                        y_batch[j] = np.flip(y_batch[j], axis=0)

                    if np.random.randint(2):
                        X_batch[j] = np.flip(X_batch[j], axis=1)
                        y_batch[j] = np.flip(y_batch[j], axis=1)

                    # TODO add different data augmentation steps

                yield np.array(X_batch), np.array(y_batch)
