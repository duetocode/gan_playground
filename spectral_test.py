import unittest
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
from spectral import Conv2D, Conv2DTranspose, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K

class SpectralTest(unittest.TestCase):

    def setUp(self):
        K.clear_session()

    def test_Conv2D(self):
        inputs = K.random_uniform([4, 32, 32, 3])

        model = Sequential([Conv2D(32, 5, strides=2, padding='same', input_shape=K.int_shape(inputs)[1:])])
        self.assertEqual(model.output_shape, (None, 16, 16, 32))

        outputs = model(inputs)
        self.assertEqual(K.int_shape(outputs), (4, 16, 16, 32))

    def test_Conv2DTranspose(self):
        inputs = K.random_uniform([2, 4, 4, 64])

        model = Sequential([Conv2DTranspose(32, 5, strides=2, padding='same', input_shape=K.int_shape(inputs)[1:])])
        self.assertEqual(model.output_shape, (None, 8, 8, 32))

        outputs = model(inputs)
        self.assertEqual(K.int_shape(outputs), (2, 8, 8, 32))

    def test_Dense(self):
        inputs = K.random_uniform([2, 16])

        model = Sequential([Dense(32, input_shape=K.int_shape(inputs)[1:])])
        self.assertEqual(model.output_shape, (None, 32))

        outputs = model(inputs)
        self.assertEqual(K.int_shape(outputs), (2, 32))

if __name__ == "__main__":
    unittest.main()
