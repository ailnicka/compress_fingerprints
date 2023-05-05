import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
tfpl = tfp.layers
tfpd = tfp.distributions
# from tensorflow.python.keras.engine.sequential import Sequential


def build_encoder_multi(out_shape, in_shapes, units_common, units_ind, dropout, activation, latent_activation, **kwargs):
    inp = [None for _ in range(len(in_shapes))]
    x = [None for _ in range(len(in_shapes))]
    for i, in_shape in enumerate(in_shapes):
        inp[i] = layers.Input(shape=(in_shape,))
        x[i] = inp[i]
        for units in units_ind[i]:
            if dropout > 0:
                x[i] = layers.Dropout(dropout )(x[i])
            x[i] = layers.Dense(units, activation=activation)(x[i])
    x_concat = tf.concat(x, axis=-1)
    for units in units_common:
        if dropout > 0:
            x_concat = layers.Dropout(dropout )(x_concat)
        x_concat = layers.Dense(units, activation=activation)(x_concat)
    # different activation for the latent space
    x_concat = layers.Dense(out_shape, activation=latent_activation)(x_concat)
    return Model(inputs=inp, outputs=x_concat)

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_variational_encoder_multi(out_shape, in_shapes, units_common, units_ind, dropout, activation, latent_activation, **kwargs):
    prior = tfpd.Independent(tfpd.Normal(loc=tf.zeros(out_shape), scale=1), reinterpreted_batch_ndims=1)
    inp = [None for _ in range(len(in_shapes))]
    x = [None for _ in range(len(in_shapes))]
    for i, in_shape in enumerate(in_shapes):
        inp[i] = layers.Input(shape=(in_shape,))
        x[i] = inp[i]
        for units in units_ind[i]:
            if dropout > 0:
                x[i] = layers.Dropout(dropout )(x[i])
            x[i] = layers.Dense(units, activation=activation)(x[i])
    x_concat = tf.concat(x, axis=-1)
    for units in units_common:
        if dropout > 0:
            x_concat = layers.Dropout(dropout )(x_concat)
        x_concat = layers.Dense(units, activation=activation)(x_concat)
    x_concat = layers.Dense(tfpl.MultivariateNormalTriL.params_size(out_shape), activation=None)(x_concat)
    x_concat = tfpl.MultivariateNormalTriL(out_shape, activity_regularizer=tfpl.KLDivergenceRegularizer(prior))(x_concat)
    return Model(inputs=inp, outputs=x_concat)

def build_encoder(out_shape, in_shape, units_common, dropout, activation, latent_activation, **kwargs):
    mdl = Sequential()
    mdl.add(layers.Input(shape=(in_shape,)))
    for units in units_common:
        if dropout > 0:
            mdl.add(layers.Dropout(dropout ))
        mdl.add(layers.Dense(units, activation=activation))
    # different activation for the latent space
    mdl.add(layers.Dense(out_shape, activation=latent_activation))
    return mdl

def build_variational_encoder(out_shape, in_shape, units_common, dropout, activation, latent_activation, **kwargs):
    prior = tfpd.Independent(tfpd.Normal(loc=tf.zeros(out_shape), scale=1), reinterpreted_batch_ndims=1)
    mdl = Sequential()
    mdl.add(layers.Input(shape=(in_shape,)))
    for units in units_common:
        if dropout > 0:
            mdl.add(layers.Dropout(dropout ))
        mdl.add(layers.Dense(units, activation=activation))
    # different activation for the latent space
    mdl.add(layers.Dense(tfpl.MultivariateNormalTriL.params_size(out_shape), activation=None))
    mdl.add(tfpl.MultivariateNormalTriL(out_shape, activity_regularizer=tfpl.KLDivergenceRegularizer(prior)))
    #mdl.add(tfpl.KLDivergenceAddLoss(prior))
    return mdl

def build_decoder(out_shape, in_shape, units_common, dropout, activation, **kwargs):
    mdl = Sequential()
    mdl.add(layers.Input(shape=(out_shape,)))
    for units in units_common:
        if dropout > 0:
            mdl.add(layers.Dropout(dropout ))
        mdl.add(layers.Dense(units, activation=activation))
    # since FPs binary, sigmoid activation at the end
    mdl.add(layers.Dense(in_shape, activation='sigmoid', name='fp_0'))
    return mdl

def build_decoder_multi(encoded_shape, in_shapes, units_common, units_ind, dropout, activation, **kwargs):
    inp = layers.Input((encoded_shape,))
    x = inp
    for units in units_common:
        if dropout > 0:
            x = layers.Dropout(dropout )(x)
        x = layers.Dense(units, activation=activation)(x)
    x = [x for _ in range(len(in_shapes))]
    for i, in_shape in enumerate(in_shapes):
        for units in units_ind[i]:
            if dropout > 0:
                x[i] = layers.Dropout(dropout )(x[i])                
            x[i] = layers.Dense(units, activation=activation)(x[i])
        # since FPs binary, sigmoid activation at the end, in case props are also as input add linear activation there
        if kwargs.get('prop_activation') is not None and i == len(in_shapes)-1:
            x[i] = layers.Dense(in_shape, name='pred')(x[i])
        else:
            x[i] = layers.Dense(in_shape, activation='sigmoid', name='fp_'+str(i))(x[i])
    return Model(outputs=x, inputs=inp)


def build_predictor(encoded_shape, properties_shape, units_common, dropout, activation, batch_norm, **kwargs):
    mdl = Sequential()
    mdl.add(layers.Input((encoded_shape,)))
    if batch_norm[0]:
        mdl.add(layers.BatchNormalization())
    for units in units_common:
        if dropout > 0:
            mdl.add(layers.Dropout(dropout ))
        mdl.add(layers.Dense(units, activation=activation))
    if batch_norm[1]:
        mdl.add(layers.BatchNormalization())
    mdl.add(layers.Dense(properties_shape, name='pred'))
    return mdl


class Autoencoder_dense_multi(Model):
    def __init__(self, encoded_shape, in_shapes, with_properties=False, properties_shape=None, **kwargs):
        super().__init__()
        DEFAULTS = {
            'encoder': {'units_ind': [[100, 100], [100, 100]], 'units_common': [200, 100], 'dropout': 0.2, 'activation': 'relu', 'latent_activation': None, 'rnd_seed': None},
            'decoder': {'units_ind': [[200, 500], [200, 500]], 'units_common': [100, 200], 'dropout': 0.2, 'activation': 'relu', 'prop_activation': False, 'rnd_seed': None},
            'properties_predictor': {'units_common': [100, 100, 50, 50], 'dropout': 0.2, 'activation': 'relu', 'batch_norm': [False, False], 'rnd_seed': None}
        }
        self.config = {'encoded_shape': encoded_shape, 'in_shapes': in_shapes, 'with_properties': with_properties, 'properties_shape': properties_shape, **kwargs}

        self.encoder = build_encoder_multi(encoded_shape, in_shapes, **(kwargs.get('encoder', DEFAULTS['encoder'])))
        self.decoder = build_decoder_multi(encoded_shape, in_shapes, **(kwargs.get('decoder', DEFAULTS['decoder'])))
        self.proppred = with_properties
        if with_properties:
            self.predictor = build_predictor(encoded_shape, properties_shape, **(kwargs.get('properties_predictor', DEFAULTS['properties_predictor'])))

    def call(self, x):
        if self.proppred:
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            properties = self.predictor(encoded)
            return decoded + [properties]
        else:
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    def encode(self, x):
        return self.encoder(x)

    def summary(self):
        print('Encoder:')
        self.encoder.summary()
        print('Decoder:')
        self.decoder.summary()
        if self.proppred:
            print('Property_predictor:')
            self.predictor.summary()

    def get_config(self):
        return self.config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Variational_Autoencoder_dense_multi(Model):
    def __init__(self, encoded_shape, in_shapes, with_properties=False, properties_shape=None, **kwargs):
        super().__init__()
        DEFAULTS = {
            'encoder': {'units_ind': [[100, 100], [100, 100]], 'units_common': [200, 100], 'dropout': 0.2, 'activation': 'relu', 'latent_activation': None, 'rnd_seed': None},
            'decoder': {'units_ind': [[200, 500], [200, 500]], 'units_common': [100, 200], 'dropout': 0.2, 'activation': 'relu', 'prop_activation': False, 'rnd_seed': None},
            'properties_predictor': {'units_common': [100, 100, 50, 50], 'dropout': 0.2, 'activation': 'relu', 'batch_norm': [False, False], 'rnd_seed': None}
        }
        self.config = {'encoded_shape': encoded_shape, 'in_shapes': in_shapes, 'with_properties': with_properties, 'properties_shape': properties_shape, **kwargs}

        self.encoder = build_variational_encoder_multi(encoded_shape, in_shapes, **(kwargs.get('encoder', DEFAULTS['encoder'])))
        self.decoder = build_decoder_multi(encoded_shape, in_shapes, **(kwargs.get('decoder', DEFAULTS['decoder'])))
        self.proppred = with_properties
        if with_properties:
            self.predictor = build_predictor(encoded_shape, properties_shape, **(kwargs.get('properties_predictor', DEFAULTS['properties_predictor'])))

    def call(self, x):
        if self.proppred:
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            properties = self.predictor(encoded)
            return decoded + [properties]
        else:
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    def encode(self, x):
        return self.encoder(x)

    def summary(self):
        print('Encoder:')
        self.encoder.summary()
        print('Decoder:')
        self.decoder.summary()
        if self.proppred:
            print('Property_predictor:')
            self.predictor.summary()

    def get_config(self):
        return self.config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Autoencoder_dense(Model):
    def __init__(self, encoded_shape, in_shapes, with_properties=False, properties_shape=None, **kwargs):
        super().__init__()
        DEFAULTS = {
            'encoder': {'units_common': [500, 200], 'dropout': 0.2, 'activation': 'relu', 'latent_activation': None, 'rnd_seed': None},
            'decoder': {'units_common': [200, 500], 'dropout': 0.2, 'activation': 'relu', 'rnd_seed': None},
            'properties_predictor': {'units_common': [100, 100, 50, 50], 'dropout': 0.2, 'activation': 'relu', 'batch_norm': [False, False], 'rnd_seed': None}
        }
        self.config = {'encoded_shape': encoded_shape, 'in_shapes': in_shapes, 'with_properties': with_properties, 'properties_shape': properties_shape, **kwargs}

        self.encoder = build_encoder(encoded_shape, in_shapes, **(kwargs.get('encoder', DEFAULTS['encoder'])))
        self.decoder = build_decoder(encoded_shape, in_shapes, **(kwargs.get('decoder', DEFAULTS['decoder'])))
        self.proppred = with_properties
        if with_properties:
            self.predictor = build_predictor(encoded_shape, properties_shape, **(kwargs.get('properties_predictor', DEFAULTS['properties_predictor'])))

    def call(self, x):
        if self.proppred:
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            properties = self.predictor(encoded)
            return decoded, properties
        else:
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    def encode(self, x):
        return self.encoder(x)

    def summary(self):
        print('Encoder:')
        self.encoder.summary()
        print('Decoder:')
        self.decoder.summary()
        if self.proppred:
            print('Property_predictor:')
            self.predictor.summary()

    def get_config(self):
        return self.config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Variational_Autoencoder_dense(Model):
    def __init__(self, encoded_shape, in_shapes, with_properties=False, properties_shape=None, **kwargs):
        super().__init__()
        DEFAULTS = {
            'encoder': {'units_common': [500, 200], 'dropout': 0.2, 'activation': 'relu', 'latent_activation': None, 'rnd_seed': None},
            'decoder': {'units_common': [200, 500], 'dropout': 0.2, 'activation': 'relu', 'rnd_seed': None},
            'properties_predictor': {'units_common': [100, 100, 50, 50], 'dropout': 0.2, 'activation': 'relu', 'batch_norm': [False, False], 'rnd_seed': None}
        }

        self.config = {'encoded_shape': encoded_shape, 'in_shapes': in_shapes, 'with_properties': with_properties, 'properties_shape': properties_shape, **kwargs}

        self.encoder = build_variational_encoder(encoded_shape, in_shapes, **(kwargs.get('encoder', DEFAULTS['encoder'])))
        self.decoder = build_decoder(encoded_shape, in_shapes, **(kwargs.get('decoder', DEFAULTS['decoder'])))
        self.proppred = with_properties
        if with_properties:
            self.predictor = build_predictor(encoded_shape, properties_shape, **(kwargs.get('properties_predictor', DEFAULTS['properties_predictor'])))

    def call(self, x):
        if self.proppred:
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            properties = self.predictor(encoded)
            return decoded, properties
        else:
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    def encode(self, x):
        return self.encoder(x)

    def summary(self):
        print('Encoder:')
        self.encoder.summary()
        print('Decoder:')
        self.decoder.summary()
        if self.proppred:
            print('Property_predictor:')
            self.predictor.summary()

    def get_config(self):
        return self.config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



def build_reference_predictor(in_shapes, properties_shape):
    model = Sequential()
    model.add(layers.Input((in_shapes,)))
    for units in [200, 200, 100]:
        model.add(layers.Dense(units, activation='relu'))
    model.add(layers.Dense(properties_shape))
    return model

def build_reference_predictor_multi(in_shapes, properties_shape):
    inp = [None for _ in range(len(in_shapes))]
    x = [None for _ in range(len(in_shapes))]
    for i, in_shape in enumerate(in_shapes):
        inp[i] = layers.Input(shape=(in_shape,))
        x[i] = inp[i]
        for units in [500, 200]:
            x[i] = layers.Dense(units, activation='relu')(x[i])
    x_concat = tf.concat(x, axis=-1)
    x_concat = layers.Dense(100, activation='relu')(x_concat)
    x_concat = layers.Dense(2 * properties_shape, activation='relu')(x_concat)
    x_concat = layers.Dense(properties_shape)(x_concat)
    return Model(inputs= inp, outputs=x_concat)
