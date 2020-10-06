import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Embedding, multiply
from tensorflow.keras.optimizers import Adam, SGD


class Generator():
    def __init__(self, batch_size, num_classes):
        self.batch_size = batch_size
        self.num_classes = num_classes

    def build(self, input_shape, input_size, layer_size, dropout=1):
        noise = Input(shape=input_shape, batch_size=self.batch_size)
        label = Input(shape=(1,), batch_size=self.batch_size, dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, 1)(label))
        input = multiply([noise, label_embedding])
        x = Dense(layer_size, activation='relu')(input)
        if dropout < 1:
            x = Dropout(dropout)(x)
        x = Dense(layer_size * 2, activation='relu')(x)
        if dropout < 1:
            x = Dropout(dropout)(x)
        x = Dense(layer_size * 4, activation='relu')(x)
        x = Dense(input_size, activation='sigmoid')(x)
        return Model(inputs=[noise, label], outputs=x)


class Discriminator():
    def __init__(self, batch_size, num_classes):
        self.batch_size = batch_size
        self.num_classes = num_classes

    def build(self, input_shape, layer_size, dropout=.1):
        noise = Input(shape=input_shape, batch_size=self.batch_size)
        label = Input(shape=(1,), batch_size=self.batch_size, dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, 1)(label))
        noise_flat = Flatten()(noise)
        input = multiply([noise_flat, label_embedding])
        x = Dense(layer_size * 4, activation='relu')(input)
        if dropout < 1:
            x = Dropout(dropout)(x)
        x = Dense(layer_size * 2, activation='relu')(x)
        if dropout < 1:
            x = Dropout(dropout)(x)
        x = Dense(layer_size, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        return Model(inputs=[noise, label], outputs=x)


class CGAN():
    def __init__(self, batch_size=128, noise_size=32, input_size=128, num_classes=2, classes=[0, 1],
                 layer_size=128, optimizer=None, lr=1e-4, generator_dropout=1,
                 discriminator_dropout=.2, loss='binary_crossentropy',
                 metrics=['accuracy'], verbose=True):
        self.batch_size = batch_size
        self.noise_size = noise_size
        self.input_size = input_size - 1
        self.layer_size = layer_size
        self.generator_dropout = generator_dropout
        self.discriminator_dropout = discriminator_dropout
        self.lr = lr
        self.loss = loss
        self.metrics = metrics
        self.verbose = verbose
        self.num_classes = num_classes
        self.classes = classes

        if optimizer == None or optimizer == 'Adam':
            self.optimizer = Adam(lr=self.lr)
        else:
            self.optimizer = SGD(lr=self.lr)
        self.generator = Generator(self.batch_size, self.num_classes) \
            .build(input_shape=(self.noise_size,), layer_size=self.layer_size,
                   input_size=self.input_size, dropout=self.generator_dropout)

        self.discriminator = Discriminator(self.batch_size, self.num_classes) \
            .build(input_shape=(self.input_size,), layer_size=self.layer_size,
                   dropout=self.discriminator_dropout)

        # Compile discriminator
        self.discriminator.compile(loss=self.loss, optimizer=self.optimizer,
                                   metrics=self.metrics)

        # The Generator takes some noise as its input and generate new data
        z = Input(shape=(self.noise_size,), batch_size=self.batch_size)
        label = Input(shape=(1,), batch_size=self.batch_size)
        record = self.generator([z, label])

        # We will train only the generator for now ()
        self.discriminator.trainable = False

        # the discriminator takes the generated data as input and decided validity
        valid_rate = self.discriminator([record, label])

        # The combined model  (stacked generator and discriminator)
        # We will train the generator to fool the discriminator
        self.model = Model([z, label], valid_rate)
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

    def train(self, data, label_size, path_prefix='', epochs=3000, step_interval=100, is_Early_stopping=False,
              generate_data=True):
        self.path_prefix = path_prefix
        self.label_size = label_size
        self.epochs = epochs
        self.step_interval = step_interval
        data_columns = data.columns

        # Adversarial ground truths
        valids = np.ones((self.batch_size, 1))
        non_valids = np.zeros((self.batch_size, 1))

        # sample collector
        pass_samples = []
        not_pass_samples = []

        Early_stopping_counter = 0
        min_loss_g = 1000
        min_loss_d = 1000
        iterations_counter = 0
        history = {'d_loss': [], 'g_loss': []}
        fools_discriminator_samples = []
        catch_discriminator_samples = []
        for epoch in range(self.epochs):
            for X_batch, Y_batch in next_batch(data, 128):
                iterations_counter += 1
                # Setting batch of real data and noise
                label = Y_batch
                noise = tf.random.normal((self.batch_size, self.noise_size))

                # generate a batch of new data
                gen_records = self.generator.predict([noise, label])

                # train the discriminator
                loss_real_discriminator = self.discriminator.train_on_batch([X_batch, label],
                                                                            valids)
                loss_fake_discriminator = self.discriminator.train_on_batch([gen_records, label],
                                                                            non_valids)
                total_loss_discriminator = 0.5 * np.add(loss_real_discriminator,
                                                        loss_fake_discriminator)

                # train the generator - to have the discriminator label samples from the
                # training of the discriminator as valid
                noise = tf.random.normal((self.batch_size, self.noise_size))
                total_loss_generator = self.model.train_on_batch([noise, label], valids)

                # if we came to step interval we need to save the generated data
                if epoch % self.step_interval == 0:
                    model_checkpoint_path_d = os.path.join('weights/', self.path_prefix,
                                                           'cgan_discriminator_model_weights_step_{}'
                                                           .format(epoch))
                    self.discriminator.save_weights(model_checkpoint_path_d)
                    model_checkpoint_path_g = os.path.join('./weights/', self.path_prefix,
                                                           'cgan_generator_model_weights_step_{}'
                                                           .format(epoch))
                    self.generator.save_weights(model_checkpoint_path_g)

                history['g_loss'].append(total_loss_generator)
                history['d_loss'].append(total_loss_discriminator[0])
                # generating some data
                if generate_data:
                    if iterations_counter > 3000:
                        z = tf.random.normal((self.batch_size, self.noise_size))
                        label_z = np.random.randint(self.classes[0], self.classes[1] + 1, (self.batch_size, 1))
                        gen_data = self.generator.predict([z, label_z])
                        is_catch = True
                        is_fool = True
                        records = self.discriminator.predict_on_batch([gen_data, label_z])
                        for i, record in enumerate(records):
                            if record > 0.5 and is_fool:
                                is_fool = False
                                fool_sample = gen_data[i]
                                fools_discriminator_samples.append([np.append(fool_sample, [label_z[i]]), record])
                            elif len(catch_discriminator_samples) < 100 and is_catch:
                                is_catch = False
                                catch_sample = gen_data[i]
                                catch_discriminator_samples.append([np.append(catch_sample, [label_z[i]]), record])

                if self.verbose:
                    print("epoch: %d iterations: %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                          (epoch, iterations_counter, total_loss_discriminator[0],
                           100 * total_loss_discriminator[1], total_loss_generator))

                if epoch > 100 and is_Early_stopping:
                    Early_stopping_counter += 1

                    if min_loss_d > total_loss_discriminator[0] or min_loss_g > total_loss_generator:
                        Early_stopping_counter = 0
                        if min_loss_d > total_loss_discriminator[0]:
                            min_loss_d = total_loss_discriminator[0]
                        if min_loss_g > total_loss_generator:
                            min_loss_g = total_loss_generator

                    elif Early_stopping_counter > 100:
                        print('Early Stopping')
                        return history
        return history

    def get_discriminator(self):
        return self.discriminator

    def get_generator(self):
        return self.generator

    def generate_samples(self, num_samples):
        x_fake = tf.random.normal((num_samples, self.noise_size))
        y_fake = np.random.randint(self.classes[0], self.classes[1] + 1, (num_samples, 1))
        pred = self.generator.predict([x_fake, y_fake])
        res_df = pd.DataFrame(pred)
        res_df['y_fake'] = y_fake
        return res_df

    def save(self, path, name):
        if os.path.isdir(path) == False:
            raise Exception('Please provide correct path - Path must be a directory!')
        model_path = os.path.join(path, name)
        self.generator.save_weights(model_path)  # Load the generator

    def load(self, path):
        if os.path.isdir(path) == False:
            raise Exception('Please provide correct path - Path must be a directory!')
        self.generator = Generator(self.batch_size)
        self.generator = self.generator.load_weights(path)
        return self.generator


def next_batch(data, batchSize):
    # loop over our dataset X in mini-batches of size batchSize
    convert_data = data.iloc[:, :].to_numpy()
    for i in np.arange(0, data.shape[0], batchSize):
        # yield a tuple of the current batched data and labels
        random_batch = convert_data[np.random.choice(convert_data.shape[0], 128, replace=False)]
        X = random_batch[:, :-1]
        y = random_batch[:, -1]
        yield (X, y)


def get_batch(train, batch_size, seed=0):
    start_index = (batch_size * seed) % len(train)
    end_index = start_index + batch_size
    shuffle_read = (batch_size * seed) // len(train)
    np.random.seed(shuffle_read)
    train_index = np.random.choice(list(train.index), replace=False, size=len(train))
    train_index = list(train_index) + list(train_index)  # duplicate to cover ranges outside the end of the set
    x = train.loc[train_index[start_index:end_index]].values
    return np.reshape(x, (batch_size, -1))
