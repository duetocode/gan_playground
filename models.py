import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Flatten, Reshape
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose

def build_generator(input_dim=512) -> Model:
    model = Sequential()

    # Map the inputs to seed
    model.add(Dense(4 * 4 * 512, input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((4, 4, 512)))

    for i in range(4):
        output_channels = 512 // 2**(i + 1)
        model.add(Conv2DTranspose(output_channels, 3, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

    assert (None, 64, 64, 32) == model.output_shape

    model.add(Conv2D(3, 1, strides=1, padding='same', activation='tanh'))

    return model

def build_discriminator() -> Model:
    model = Sequential()

    model.add(Conv2D(64, 1, strides=1, padding='same', input_shape=(64, 64, 3)))
    model.add(LeakyReLU())

    # 32, 16, 8, 4
    for i in range(4):
        output_channels = min(64 * 2**(i + 1), 512)
        model.add(Conv2D(output_channels, 3, strides=2, padding='same'))
        model.add(LeakyReLU())
    
    model.add(Flatten())
    model.add(Dense(1))

    return model


    