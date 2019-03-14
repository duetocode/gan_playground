import unittest
from keras import backend as K
from models import build_discriminator, build_generator

class ModelsTest(unittest.TestCase):

    def setUp(self):
        K.clear_session()
    
    def test_generator(self):
        generator = build_generator()

        self.assertIsNotNone(generator)
        self.assertEqual((None, 64, 64, 3), generator.output_shape)

    def test_discriminator(self):
        discriminator = build_discriminator()

        self.assertIsNotNone(discriminator)
        self.assertEqual((None, 1), discriminator.output_shape)

if __name__ == "__main__":
    unittest.main()