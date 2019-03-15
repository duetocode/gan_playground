import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, LeakyReLU, BatchNormalization, Lambda, Concatenate, Reshape

from models import build_discriminator

def build_generator(input_dim=512, batch_size=16):
    inputs = Input(shape=[input_dim], batch_size=batch_size)

    styles = Dense(4 * 4 * 3, use_bias=False)(inputs)
    styles = Reshape([4, 4, 3])(styles)
    outputs = SeedLayer()(styles)
    
    # 8|256, 16|128, 32|64, 64|32
    for i in range(4):
        channels = max(512 //  2**(i + 1), 32)
        styles = Dense(np.prod(K.int_shape(outputs)[1:]), use_bias=False)(inputs)
        styles = Reshape(K.int_shape(outputs)[1:])(styles)
        outputs = Concatenate(axis=-1)([outputs, styles])
        
        outputs = Conv2DTranspose(channels, 3, strides=2, padding='same')(outputs)
        outputs = BatchNormalization()(outputs)
        outputs = LeakyReLU()(outputs)

    outputs = Conv2D(3, 1, padding='same', activation='tanh')(outputs)

    return keras.Model(inputs, outputs)

class SeedLayer(keras.layers.Layer):
    
    def build(self, input_shape):
        self.seed = self.add_weight('seed',
                                        shape=[1, 4, 4, 3],
                                        initializer='uniform',
                                        trainable=True)
        super(SeedLayer, self).build(input_shape)
    
    def call(self, x):
        seed_expanded = K.ones([K.shape(x)[0], 4, 4, 3]) * self.seed
        return K.concatenate([seed_expanded, x], axis=-1)

    def compute_output_shape(self, **kwargs):
        return (4, 4, 3)

