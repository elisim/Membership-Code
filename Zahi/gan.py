import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD


class Generator():
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build(self, input_shape, input_size, layer_size, dropout=1):
        input = Input(shape=input_shape, batch_size=self.batch_size)
        x = Dense(layer_size, activation='relu')(input)
        if dropout < 1:
            x = Dropout(dropout)(x)
        x = Dense(layer_size * 2, activation='relu')(x)
        if dropout < 1:
            x = Dropout(dropout)(x)
        x = Dense(layer_size * 4, activation='relu')(x)
        x = Dense(input_size)(x)
        return Model(inputs=input, outputs=x)


class Discriminator():
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build(self, input_shape, layer_size, dropout=.2):
        input = Input(shape=input_shape, batch_size=self.batch_size)
        x = Dense(layer_size * 4, activation='relu')(input)
        # if dropout < 1:
        #  x = Dropout(dropout)(x)
        x = Dense(layer_size * 2, activation='relu')(x)
        # if dropout < 1:
        #  x = Dropout(dropout)(x)
        x = Dense(layer_size, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        return Model(inputs=input, outputs=x)


class GAN:
    def __init__(self, batch_size=128, noise_size=32, input_size=128,
                 layer_size=128, optimizer=None, lr=1e-4, generator_dropout=1,
                 discriminator_dropout=.2, loss='binary_crossentropy',
                 metrics=['accuracy'], verbose=True):
        self.batch_size = batch_size
        self.noise_size = noise_size
        self.input_size = input_size
        self.layer_size = layer_size
        self.generator_dropout = generator_dropout
        self.discriminator_dropout = discriminator_dropout
        self.lr = lr
        self.loss = loss
        self.metrics = metrics
        self.verbose = verbose

        if optimizer == None or optimizer == 'Adam':
            self.optimizer = Adam(lr=self.lr)
        else:
            self.optimizer = SGD(lr=self.lr)

        self.generator = Generator(self.batch_size) \
            .build(input_shape=(self.noise_size,), layer_size=self.layer_size,
                   input_size=self.input_size, dropout=self.generator_dropout)

        self.discriminator = Discriminator(self.batch_size) \
            .build(input_shape=(self.input_size,), layer_size=self.layer_size,
                   dropout=self.discriminator_dropout)

        # Compile discriminator
        self.discriminator.compile(loss=self.loss, optimizer=self.optimizer,
                                   metrics=self.metrics)

        # The Generator takes some noise as its input and generate new data
        z = Input(shape=(self.noise_size,))
        record = self.generator(z)

        # We will train only the generator for now ()
        self.discriminator.trainable = False

        # the discriminator takes the generated data as input and decided validity
        valid_rate = self.discriminator(record)

        # The combined model  (stacked generator and discriminator)
        # We will train the generator to fool the discriminator
        self.model = Model(z, valid_rate)
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

    def train(self, data, path_prefix='', epochs=3000, step_interval=100):
        self.path_prefix = path_prefix
        self.epochs = epochs
        self.step_interval = step_interval

        data_columns = data.columns
        history = {'d_loss': [], 'g_loss': []}
        # Adversarial ground truths
        valids = np.ones((self.batch_size, 1))
        non_valids = np.zeros((self.batch_size, 1))

        for epoch in range(self.epochs):

            # Setting batch of real data and noise
            batch_data = get_batch(data, self.batch_size)
            noise = tf.random.normal((self.batch_size, self.noise_size))

            # generate a batch of new data
            gen_data = self.generator.predict(noise)

            # train the discriminator
            loss_real_discriminator = self.discriminator.train_on_batch(batch_data,
                                                                        valids)
            loss_fake_discriminator = self.discriminator.train_on_batch(gen_data,
                                                                        non_valids)
            total_loss_discriminator = 0.5 * np.add(loss_real_discriminator,
                                                    loss_fake_discriminator)

            # train the generator - to have the discriminator label samples from the
            # training of the discriminator as valid
            noise = tf.random.normal((self.batch_size, self.noise_size))
            total_loss_generator = self.model.train_on_batch(noise, valids)

            history['d_loss'].append(total_loss_discriminator[0])
            history['g_loss'].append(total_loss_generator)

            if self.verbose:
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                      (epoch, total_loss_discriminator[0],
                       100 * total_loss_discriminator[1], total_loss_generator))

            # if we came to step interval we need to save the generated data
            if epoch % self.step_interval == 0:
                model_checkpoint_path_d = os.path.join('weights/', self.path_prefix,
                                                       'discriminator_model_weights_step_{}'
                                                       .format(epoch))
                self.discriminator.save_weights(model_checkpoint_path_d)
                model_checkpoint_path_g = os.path.join('./weights/', self.path_prefix,
                                                       'generator_model_weights_step_{}'
                                                       .format(epoch))
                self.generator.save_weights(model_checkpoint_path_g)

                # generating some data
                z = tf.random.normal((432, self.noise_size))
                gen_data = self.generator(z)
                if self.verbose:
                    print('generated_data')
        return history

    def save(self, path):
        if not os.path.isdir(path):
            raise Exception('Please provide correct path - Path must be a directory!')
        self.generator.save_weights(os.path.join(path, 'generator_weights.h5'))  # Load the generator
        self.discrimnator.save_weights(os.path.join(path, 'disriminator_weights.h5'))

    def load(self, path):
        if not os.path.isdir(path):
            raise Exception('Please provide correct path - Path must be a directory!')
        self.generator = Generator(self.batch_size)
        self.generator = self.generator.load_weights(os.path.join(path, 'generator_weights.h5'))
        self.discriminator = Discriminator(self.batch_size)
        self.discriminator = self.discriminator.load_weights(os.path.join(path, 'discriminator_weights.h5'))
        return self.generator, self.discriminator


def get_batch(train, batch_size, seed=0):
    start_index = (batch_size * seed) % len(train)
    end_index = start_index + batch_size
    shuffle_read = (batch_size * seed) // len(train)
    np.random.seed(shuffle_read)
    train_index = np.random.choice(list(train.index), replace=False, size=len(train))
    train_index = list(train_index) + list(train_index)  # duplicate to cover ranges outside the end of the set
    x = train.loc[train_index[start_index:end_index]].values
    return np.reshape(x, (batch_size, -1))
