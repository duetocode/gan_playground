import unittest
import tensorflow as tf
from tensorflow.keras import backend as K
from models_style import build_generator

class ModelStyleTest(unittest.TestCase):

    def setUp(self):
        K.clear_session()

    def test_build_generator(self):
        m = build_generator()



if __name__ == "__main__":
    unittest.main()